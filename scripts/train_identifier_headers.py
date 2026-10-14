"""Matched code-prompt/header-only learning probe; not complete code generation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys


def header_records(records):
    """Preserve source prompts/order/styles; shorten only reference responses."""
    result = []
    for record in records:
        answer = record["answer"]
        header = answer.splitlines(keepends=True)[0] if answer else ""
        if not header.startswith("def " + record["function"] + "(") or not header.endswith(":\n") or "\r" in header:
            raise ValueError("requires an LF-terminated declared function header")
        result.append({**record, "answer": header})
    if not result:
        raise ValueError("empty header split")
    return result


def score_header(record, output, predicted_bytes):
    from scripts.diagnose_identifier_copy import header_name
    expected = list(record["function"].encode("utf-8"))
    if len(predicted_bytes) != len(expected):
        raise ValueError("prediction length mismatch")
    return {"id": record["id"], "prompt": record["prompt"],
            "requested_name": record["function"], "reference_header": record["answer"],
            "output": output, "exact_header_response": output == record["answer"],
            "autonomous_header_name_correct": header_name(output) == record["function"],
            "teacher_forced_expected_bytes": expected, "teacher_forced_argmax_bytes": predicted_bytes,
            "teacher_forced_complete_name": expected == predicted_bytes,
            "teacher_forced_correct_bytes": sum(a == b for a, b in zip(expected, predicted_bytes)),
            "teacher_forced_total_bytes": len(expected)}


def run(dataset, output, steps=6000, augment_prompts=False, position_jitter=False, augmentation_policy="four_forms"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.prompt_robustness import validate_training_policy
    validate_training_policy(augment_prompts, augmentation_policy)
    if not 1 <= steps <= 10000:
        raise ValueError("steps must be 1..10000")
    import torch
    import torch.nn.functional as F
    from train_nppai import NppAIModel, export_to_bin
    from scripts.train_broader_instruction_experiment import read_split, sample
    from scripts.train_instruction_experiment import generate
    from scripts.diagnose_identifier_copy import identifier_span
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.prompt_robustness import augment_training_records, training_variants

    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    root = Path(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    training = header_records(read_split(root, "train", manifest))
    source_count = len(training)
    if augment_prompts:
        training = augment_training_records(training, augmentation_policy)
    if len(training) < 4:
        raise ValueError("requires at least four training records")
    context = 256
    samples = [sample(record, context) for record in training]
    jitter = None
    if position_jitter:
        from scripts.position_jitter import PositionJitter
        jitter = PositionJitter()
    model = NppAIModel(256, 64, 128, 2, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    first_loss = None
    model.train()
    for step in range(steps):
        batch = [sample(jitter.apply(record), context) for record in random.sample(training, 4)] if jitter else random.sample(samples, 4)
        x = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        if step % 100 == 0 or step == steps - 1:
            print(f"step={step+1} loss={float(loss.detach()):.6f}", flush=True)
    model.eval()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "NppAI-identifier-headers.nppai"
    export_to_bin(model, str(checkpoint))
    reloaded = torch_model(load_weights(checkpoint))
    splits = {"train_first24": training[:24],
              "validation": header_records(read_split(root, "validation", manifest)),
              "test": header_records(read_split(root, "test", manifest))}
    report = {"seed": 42, "steps": steps, "batch_size": 4, "context": context,
              "architecture": {"vocabulary": 256, "dimension": 64, "hidden_dimension": 128, "layers": 2},
              "optimizer": {"name": "AdamW", "learning_rate": 0.001, "gradient_clip_norm": 1.0},
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "threads": 2},
              "source_manifest": manifest, "training_records": len(training),
              "derived_training_sha256": hashlib.sha256(json.dumps(training, sort_keys=True).encode("utf-8")).hexdigest(),
              "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "initial_loss": first_loss, "final_loss": float(loss.detach()), "reload_outputs_identical": True,
              "limits": "Partial header task, not valid complete functions or bodies. Frozen source names and splits unchanged; optional training-only prompt augmentation is recorded separately. Header responses shorten training sequences and change supervised targets/token budget relative to full code. Teacher forcing supplies preceding reference bytes. Seen first24 sample, single seed, previously exposed synthetic families. No native INT8 or production-model promotion.",
              "splits": {}}
    if augment_prompts:
        report["prompt_augmentation"] = {"source_records": source_count, "derived_records": len(training),
                                         "policy": augmentation_policy,
                                         "variants": list(training_variants(training[0], augmentation_policy))}
    if jitter:
        report["position_jitter"] = jitter.metadata()
    with torch.no_grad():
        for split, records in splits.items():
            cases = []
            for record in records:
                generated = generate(reloaded, record["prompt"], context)
                if generated != generate(model, record["prompt"], context):
                    raise RuntimeError("export/reload changed greedy output")
                sequence, start, end = identifier_span(record, context)
                logits = reloaded(torch.tensor([list(sequence[:-1])], dtype=torch.long))[0, start:end]
                cases.append(score_header(record, generated, logits.argmax(dim=-1).tolist()))
            keys = ("exact_header_response", "autonomous_header_name_correct", "teacher_forced_complete_name",
                    "teacher_forced_correct_bytes", "teacher_forced_total_bytes")
            summary = {"total": len(cases), **{key: sum(c[key] for c in cases) for key in keys}}
            print(json.dumps({"split": split, **summary}), flush=True)
            report["splits"][split] = {**summary, "cases": cases}
    (destination / "header-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--augment-prompts", action="store_true", help="expand training only with four fixed wording variants")
    parser.add_argument("--position-jitter", action="store_true", help="random training-only whitespace offsets; independent seed")
    parser.add_argument("--augmentation-policy", choices=("four_forms", "eight_forms"), default="four_forms")
    args = parser.parse_args()
    run(args.dataset, args.output, args.steps, args.augment_prompts, args.position_jitter, args.augmentation_policy)
