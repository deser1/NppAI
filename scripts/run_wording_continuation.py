"""Fixed four/eight-form continuation from the same verified pretrained pair."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_staged_generation import check_source
from scripts.prompt_robustness import check_augmentation_source, augment_training_records

SOURCE_RUN = 38582420228
TASKS = {"headers": ("header-source", "NppAI-identifier-headers.nppai", "header-report.json", "source_manifest"),
         "bodies": ("body-source", "NppAI-broader-instruction.nppai", "broader-report.json", "dataset_manifest")}


def verify_source(source, dataset, prefix=""):
    manifest = json.loads((Path(dataset) / "manifest.json").read_bytes())
    result = {}
    for task, (folder, model_name, report_name, manifest_key) in TASKS.items():
        model = Path(source) / (prefix + folder) / model_name
        report = json.loads((Path(source) / (prefix + folder) / report_name).read_bytes())
        check_source(report, model.read_bytes(), manifest, manifest_key, 10000)
        check_augmentation_source(report, "four_forms")
        result[task] = {"model": model, "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    return result


def continuation_metadata(source_hash):
    return {"source_run_id": SOURCE_RUN, "source_checkpoint_sha256": source_hash,
            "source_steps": 10000, "additional_steps": 10000, "total_optimizer_updates": 20000,
            "optimizer_reset": True}


def check_continuation(report, source_hash, policy):
    check_augmentation_source(report, policy)
    if report.get("continuation") != continuation_metadata(source_hash) or report.get("steps") != 10000:
        raise ValueError("continuation source/budget differs")
    initialization = report.get("initialization", {})
    if initialization.get("checkpoint_sha256") != source_hash or initialization.get("method") != "all dense FP32 weights; fresh AdamW optimizer":
        raise ValueError("continuation initialization differs")
    if report.get("reload_outputs_identical") is not True:
        raise ValueError("continuation checkpoint reload was not verified")


def train(source, dataset, output, task, policy):
    origins = verify_source(source, dataset)
    destination = Path(output)
    initial_model = origins[task]["model"]
    if task == "headers":
        from scripts.train_identifier_headers import run
        report = run(dataset, output, steps=10000, augment_prompts=True,
                     augmentation_policy=policy, initial_model=initial_model)
    else:
        from scripts.train_broader_instruction_experiment import run, read_split
        from scripts.check_dense_runtime_parity import load_weights, torch_model
        from scripts.train_instruction_experiment import generate
        report = run(dataset, output, steps=10000, identifier_weight=1.0, training_names=512,
                     initial_model=initial_model, augment_prompts=True, augmentation_policy=policy)
        manifest = json.loads((Path(dataset) / "manifest.json").read_bytes())
        seen = augment_training_records(read_split(Path(dataset), "train", manifest), policy)[:24]
        cases = report["training_probe_first24"]["cases"]
        reloaded = torch_model(load_weights(destination / TASKS[task][1]))
        if [c["id"] for c in cases] != [r["id"] for r in seen] or any(
            generate(reloaded, record["prompt"], 256) != case["output"]
            for record, case in zip(seen, cases)):
            raise RuntimeError("body export/reload changed seen probe outputs")
        report["reload_outputs_identical"] = True
        report["reload_seen_probe_count"] = 24
    report["continuation"] = continuation_metadata(origins[task]["sha256"])
    check_continuation(report, origins[task]["sha256"], policy)
    (destination / TASKS[task][2]).write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for name in ("header-source", "body-source"):
        shutil.copytree(Path(source) / name, destination / ("initial-" + name), dirs_exist_ok=True)
    return report


def replay(source, dataset, headers, bodies, output, policy, source_run):
    origins = verify_source(source, dataset)
    for task, folder in (("headers", Path(headers)), ("bodies", Path(bodies))):
        report = json.loads((folder / TASKS[task][2]).read_bytes())
        check_continuation(report, origins[task]["sha256"], policy)
        copied = verify_source(folder, dataset, prefix="initial-")
        if any(copied[k]["sha256"] != origins[k]["sha256"] for k in origins):
            raise ValueError("preserved initialization sources differ")
    from scripts.evaluate_staged_generation import run
    from scripts.audit_wording_report import audit
    report = run(headers, bodies, dataset, output, source_run, True, 10000, True, policy)
    result = audit(report, dataset)
    result["source_report_sha256"] = hashlib.sha256((Path(output) / "staged-report.json").read_bytes()).hexdigest()
    (Path(output) / "wording-audit.json").write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    for name in ("header-source", "body-source"):
        shutil.copytree(Path(source) / name, Path(output) / ("initial-" + name), dirs_exist_ok=True)
    print(json.dumps({"policy": policy, "wording_smoke_passed": result["wording_smoke_passed"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "replay"))
    for name in ("source", "dataset", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--policy", choices=("four_forms", "eight_forms"), required=True)
    parser.add_argument("--task", choices=("headers", "bodies"))
    parser.add_argument("--headers")
    parser.add_argument("--bodies")
    parser.add_argument("--source-run", type=int)
    args = parser.parse_args()
    if args.mode == "train":
        if not args.task:
            parser.error("train requires --task")
        train(args.source, args.dataset, args.output, args.task, args.policy)
    else:
        if not args.headers or not args.bodies or not args.source_run:
            parser.error("replay requires --headers, --bodies and --source-run")
        replay(args.source, args.dataset, args.headers, args.bodies, args.output, args.policy, args.source_run)
