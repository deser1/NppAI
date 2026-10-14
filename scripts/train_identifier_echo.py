"""Fixed-format copying sanity check; not code generation or a deployed model."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys


def echo_records(records):
    result = []
    seen = set()
    for record in records:
        name = record["function"]
        if not re.fullmatch(r"fn_[a-z0-9]{6}", name):
            raise ValueError("echo probe requires frozen nine-byte V3 names")
        if name not in seen:
            result.append({"id": record["id"], "function": name,
                           "prompt": "Copy identifier: " + name,
                           "answer": name + "\n"})
            seen.add(name)
    if not result:
        raise ValueError("empty echo split")
    return result


def encode_echo(record, context=96):
    prefix = ("[USER]: " + record["prompt"] + "\n[AI]:\n").encode("utf-8")
    answer = record["answer"].encode("utf-8")
    sequence = prefix + answer + b"\0"
    if b"\0" in prefix + answer or len(sequence) > context:
        raise ValueError("NUL or context overflow in echo record")
    x, y = list(sequence[:-1]), list(sequence[1:])
    y[:len(prefix) - 1] = [-100] * (len(prefix) - 1)
    return x, y, len(prefix) - 1


def score_echo(record, output, predicted_name_bytes):
    expected = list(record["function"].encode("ascii"))
    if len(expected) != len(predicted_name_bytes):
        raise ValueError("identifier prediction length mismatch")
    return {"id": record["id"], "requested_name": record["function"],
            "prompt": record["prompt"], "output": output,
            "exact_response": output == record["answer"],
            "teacher_forced_expected_bytes": expected,
            "teacher_forced_argmax_bytes": predicted_name_bytes,
            "teacher_forced_correct_bytes": sum(a == b for a, b in zip(expected, predicted_name_bytes)),
            "teacher_forced_complete_name": expected == predicted_name_bytes}


def run(dataset, output, steps=3000, context=96, training_only=False):
    if not 1 <= steps <= 10000:
        raise ValueError("steps must be 1..10000")
    if context not in (96, 256):
        raise ValueError("context must be 96 or 256")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    import torch.nn.functional as F
    from train_nppai import NppAIModel, export_to_bin
    from scripts.train_broader_instruction_experiment import read_split
    from scripts.train_instruction_experiment import generate
    from scripts.check_dense_runtime_parity import load_weights, torch_model

    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    root = Path(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    training = echo_records(read_split(root, "train", manifest))
    if len(training) < 4:
        raise ValueError("requires at least four training names")
    samples = [encode_echo(record, context) for record in training]
    model = NppAIModel(256, 64, 128, 2, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    first_loss = None
    model.train()
    for step in range(steps):
        batch = random.sample(samples, 4)
        x = torch.tensor([b[0] for b in batch], dtype=torch.long)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
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
    checkpoint = destination / "NppAI-identifier-echo.nppai"
    export_to_bin(model, str(checkpoint))
    reloaded = torch_model(load_weights(checkpoint))
    # Held-out names are first read after the final optimizer update.
    splits = {"train_first16": training[:16]}
    if not training_only:
        splits.update(validation=echo_records(read_split(root, "validation", manifest)),
                      test=echo_records(read_split(root, "test", manifest)))
    report = {"seed": 42, "steps": steps, "batch_size": 4, "context": context,
              "architecture": {"vocabulary": 256, "dimension": 64, "hidden_dimension": 128, "layers": 2},
              "optimizer": {"name": "AdamW", "learning_rate": 0.001, "gradient_clip_norm": 1.0},
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "threads": 2},
              "source_manifest": manifest, "training_names": len(training),
              "echo_training_sha256": hashlib.sha256(json.dumps(training, sort_keys=True).encode("utf-8")).hexdigest(),
              "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "initial_loss": first_loss, "final_loss": float(loss.detach()), "reload_outputs_identical": True,
              "training_only": training_only,
              "limitations": "Fixed nine-byte names at a fixed prompt position; a synthetic copying sanity check only. No variable-length, shifted-position, code-generation or native INT8 claim. Training first16 is seen data. Teacher-forced predictions supply preceding true bytes and are not autonomous outputs. One seed; no held-out tuning or checkpoint promotion.",
              "splits": {}}
    with torch.no_grad():
        for split, records in splits.items():
            cases = []
            for record in records:
                generated = generate(reloaded, record["prompt"], context)
                if generated != generate(model, record["prompt"], context):
                    raise RuntimeError("export/reload changed greedy output")
                x, _, start = encode_echo(record, context)
                logits = reloaded(torch.tensor([x], dtype=torch.long))[0, start:start+9]
                cases.append(score_echo(record, generated, logits.argmax(dim=-1).tolist()))
            summary = {"total": len(cases),
                       "exact_response": sum(c["exact_response"] for c in cases),
                       "teacher_forced_complete_name": sum(c["teacher_forced_complete_name"] for c in cases),
                       "teacher_forced_correct_bytes": sum(c["teacher_forced_correct_bytes"] for c in cases),
                       "teacher_forced_total_bytes": 9 * len(cases)}
            print(json.dumps({"split": split, **summary}), flush=True)
            report["splits"][split] = {**summary, "cases": cases}
    (destination / "echo-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--context", type=int, choices=(96, 256), default=96)
    parser.add_argument("--training-only", action="store_true", help="do not read or evaluate held-out splits")
    args = parser.parse_args()
    run(args.dataset, args.output, args.steps, args.context, args.training_only)
