"""Fixed-budget training and staged replay for three PyTorch attention arms."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train(dataset, output, arm, task, steps=10000):
    if not 1 <= steps <= 10000:
        raise ValueError("steps must be 1..10000")
    import torch
    import torch.nn.functional as F
    from scripts.attention_probe import build_model, configuration, load_model, save_model
    from scripts.train_broader_instruction_experiment import read_split, sample
    from scripts.train_identifier_headers import header_records
    from scripts.prompt_robustness import augment_training_records
    from scripts.train_instruction_experiment import generate

    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    root = Path(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = read_split(root, "train", manifest)
    if task == "headers":
        records = header_records(records)
    records = augment_training_records(records)
    samples = [sample(record, 256) for record in records]
    config = configuration(arm, task)
    model = build_model(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    initial = None
    model.train()
    for step in range(steps):
        batch = random.sample(samples, 4)
        x = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if initial is None:
            initial = float(loss.detach())
        if step % 100 == 0 or step == steps - 1:
            print(f"step={step+1} loss={float(loss.detach()):.6f}", flush=True)
    model.eval()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "probe-model.pt"
    save_model(model, config, checkpoint)
    reloaded, loaded_config = load_model(checkpoint)
    assert loaded_config == config
    # A fixed seen-data reload probe; held-out records are used only by replay.
    reload_cases = []
    for record in records[:16]:
        before = generate(model, record["prompt"], 256)
        after = generate(reloaded, record["prompt"], 256)
        if before != after:
            raise RuntimeError("checkpoint reload changed greedy output")
        reload_cases.append({"id": record["id"], "prompt": record["prompt"], "output": after})
    report = {"config": config, "seed": 42, "steps": steps, "batch_size": 4,
              "optimizer": {"name": "AdamW", "learning_rate": 0.001, "clip": 1.0},
              "dataset_manifest": manifest, "training_records": len(records),
              "derived_training_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
              "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "active_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "stored_parameters": sum(p.numel() for p in model.parameters()),
              "initial_loss": initial, "final_loss": float(loss.detach()),
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "threads": 2},
              "reload_outputs_identical": True, "seen_reload_probe": reload_cases,
              "limits": "PyTorch-only experimental architecture, not native NPPAI format. Same four-form training corpus/update budget/record sampling across arms. Rotary has no active learned-position parameters. One seed, exposed synthetic families, no test tuning, execution or promotion."}
    (destination / "probe-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def replay(dataset, headers, bodies, output, source_run):
    import torch
    from scripts.attention_probe import load_model
    from scripts.train_broader_instruction_experiment import read_split
    from scripts.train_instruction_experiment import generate
    from scripts.prompt_robustness import evaluation_variants, paired_outcomes
    from scripts.evaluate_staged_generation import header_fits, score_output

    torch.set_num_threads(2)
    root = Path(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    loaded = []
    for folder, task in ((Path(headers), "headers"), (Path(bodies), "bodies")):
        path = folder / "probe-model.pt"
        report = json.loads((folder / "probe-report.json").read_text(encoding="utf-8"))
        if report["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest() or report["dataset_manifest"] != manifest:
            raise ValueError("source checkpoint hash or frozen manifest mismatch")
        if (report["steps"], report["seed"], report["config"]["task"]) != (10000, 42, task):
            raise ValueError("source task/budget/seed mismatch")
        model, config = load_model(path)
        if config != report["config"]:
            raise ValueError("checkpoint configuration differs from source report")
        loaded.append((model, config, report))
    header_model, header_config, header_report = loaded[0]
    body_model, body_config, body_report = loaded[1]
    if header_config["arm"] != body_config["arm"]:
        raise ValueError("header and body attention arms differ")
    report = {"arm": header_config["arm"], "source_run_id": source_run,
              "header_source": header_report, "body_source": body_report,
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__}, "splits": {},
              "limits": "Two-model FP32 staged inference; no native compatibility claim, execution or promotion. Six correlated wording probes on exposed synthetic tasks, not independent competence tasks. Raw generated headers are accepted or rejected without reference correction."}
    keys = ("nonempty", "syntax_any_function", "requested_function", "reference_ast", "exact_response")
    with torch.no_grad():
        for split in ("validation", "test"):
            original = read_split(root, split, manifest)
            for variant in evaluation_variants(original[0], True):
                cases = []
                for record in original:
                    record = evaluation_variants(record, True)[variant]
                    header = generate(header_model, record["prompt"], 256)
                    accepted = header_fits(record["prompt"], header, 256)
                    staged = generate(body_model, record["prompt"], 256, assistant_prefix=header) if accepted else ""
                    single = generate(body_model, record["prompt"], 256)
                    cases.append({"id": record["id"], "prompt": record["prompt"], "reference": record["answer"],
                                  "generated_header": header, "header_accepted": accepted,
                                  "header_exact_reference": header == record["answer"].splitlines(keepends=True)[0],
                                  "staged": score_output(record, staged), "single_model": score_output(record, single)})
                summary = {"total": len(cases), "headers_accepted": sum(c["header_accepted"] for c in cases),
                           "headers_exact_reference": sum(c["header_exact_reference"] for c in cases),
                           **{arm: {key: sum(c[arm][key] for c in cases) for key in keys} for arm in ("staged", "single_model")}}
                group = split + "/" + variant
                reference_cases = cases if variant == "original" else report["splits"][split + "/original"]["cases"]
                summary["paired_to_original"] = {arm: paired_outcomes(reference_cases, cases, arm) for arm in ("staged", "single_model")}
                print(json.dumps({"split": group, **summary}), flush=True)
                report["splits"][group] = {**summary, "cases": cases}
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for folder, name in ((Path(headers), "header-source"), (Path(bodies), "body-source")):
        shutil.copytree(folder, destination / name, dirs_exist_ok=True)
    (destination / "attention-replay.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "replay"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=("legacy1", "absolute4", "rotary4"))
    parser.add_argument("--task", choices=("headers", "bodies"))
    parser.add_argument("--headers")
    parser.add_argument("--bodies")
    parser.add_argument("--source-run", type=int)
    args = parser.parse_args()
    if args.mode == "train":
        if not args.arm or not args.task:
            parser.error("train requires --arm and --task")
        train(args.dataset, args.output, args.arm, args.task)
    else:
        if not args.headers or not args.bodies or not args.source_run:
            parser.error("replay requires --headers, --bodies and --source-run")
        replay(args.dataset, args.headers, args.bodies, args.output, args.source_run)
