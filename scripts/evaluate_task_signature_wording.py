"""Diagnose body generation using explicit task signatures, not learned headers."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.audit_wording_report import audit
from scripts.evaluate_native_staged import native_generate, summarize
from scripts.evaluate_signature_guidance import signature_prefix
from scripts.evaluate_staged_generation import check_source, header_fits, score_output
from scripts.prompt_robustness import check_augmentation_source, paired_outcomes


def task_header(task):
    # Match the learned-header replay's LF-only declaration prefix, without indentation.
    header = signature_prefix(task).removesuffix("    ")
    if not header_fits(task, header, 256):
        raise ValueError("explicit task header does not fit the frozen context")
    return header


def verify_inputs(source, dataset):
    source, root = Path(source), Path(dataset)
    comparison = json.loads((source / "comparison.json").read_bytes())
    if comparison["source_run_id"] != 38582420228 or comparison["source_policy"] != "four_forms":
        raise ValueError("unexpected frozen source run/policy")
    if comparison.get("infrastructure_errors") != 0:
        raise ValueError("cached comparison has infrastructure errors")
    baselines = {}
    for runtime, relative in (("fp32", "fp32/staged-report.json"), ("native", "native-report.json")):
        raw = (source / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != comparison[runtime + "_report_sha256"]:
            raise ValueError("cached replay hash mismatch: " + runtime)
        baselines[runtime] = json.loads(raw)
        audit(baselines[runtime], root)
    model = source / "fp32/body-source/NppAI-broader-instruction.nppai"
    body_report = json.loads((source / "fp32/body-source/broader-report.json").read_bytes())
    manifest = json.loads((root / "manifest.json").read_bytes())
    check_source(body_report, model.read_bytes(), manifest, "dataset_manifest", 10000)
    check_augmentation_source(body_report, "four_forms")
    if any(report["source_body_report"] != body_report for report in baselines.values()):
        raise ValueError("cached baselines use a different body model/report")
    return baselines, model, manifest


def run(source, dataset, engine, output):
    baselines, model_path, manifest = verify_inputs(source, dataset)
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.identifier_curriculum import validate_initial_weights
    from scripts.train_instruction_experiment import generate
    import torch
    torch.set_num_threads(2)
    weights = load_weights(model_path)
    validate_initial_weights(weights)
    model = torch_model(weights)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    treatments = {runtime: {"dataset_manifest": manifest, "splits": {},
                            "generation_mode": "task_signature_forced", "token_budget": 160,
                            "limits": "Header names/parameters are supplied by the task parser, not learned or autonomously generated. No reference correction, generated-code execution or promotion."}
                  for runtime in baselines}
    infrastructure_errors = 0
    for group, control in baselines["fp32"]["splits"].items():
        native_by_id = {c["id"]: c for c in baselines["native"]["splits"][group]["cases"]}
        cases = {runtime: [] for runtime in treatments}
        for original in control["cases"]:
            task, header = original["prompt"], task_header(original["prompt"])
            fp32_output = generate(model, task, 256, assistant_prefix=header)
            native_output, diagnostic = native_generate(engine, model_path, task, header)
            infrastructure_errors += diagnostic["infrastructure_error"]
            # Only now use frozen references for scoring.
            record = {"answer": original["reference"],
                      "function": ast.parse(original["reference"]).body[0].name}
            for runtime, generated in (("fp32", fp32_output), ("native", native_output)):
                baseline = original if runtime == "fp32" else native_by_id[original["id"]]
                cases[runtime].append({"id": original["id"], "prompt": task,
                    "reference": original["reference"], "forced_task_header": header,
                    "header_accepted": True,
                    "header_exact_reference": header == original["reference"].splitlines(keepends=True)[0],
                    "staged": score_output(record, generated),
                    "single_model": copy.deepcopy(baseline["single_model"]),
                    "cached_learned_header": baseline["generated_header"],
                    "cached_learned_staged": copy.deepcopy(baseline["staged"]),
                    **({"native_diagnostics": diagnostic} if runtime == "native" else {})})
        for runtime in treatments:
            treatments[runtime]["splits"][group] = summarize(cases[runtime])
            print(json.dumps({"assisted_runtime": runtime, "group": group,
                **{k: v for k, v in treatments[runtime]["splits"][group].items() if k != "cases"}}), flush=True)
    result = {"source_inference_run_id": 38591181871, "source_training_run_id": 38582420228,
              "body_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "engine_sha256": hashlib.sha256(Path(engine).read_bytes()).hexdigest(),
              "infrastructure_errors": infrastructure_errors,
              "limits": "Assisted task-signature diagnostic, not autonomous identifier copying. Cached learned-header and free-body controls use the same frozen body model and tasks. Treatment removes a learned-header inference call; resource budgets differ. Exposed correlated tasks, one seed, runtime differences, no execution or promotion."}
    for runtime, report in treatments.items():
        raw = (json.dumps(report, indent=2) + "\n").encode("utf-8")
        (destination / (runtime + "-assisted-report.json")).write_bytes(raw)
        result[runtime + "_assisted_report_sha256"] = hashlib.sha256(raw).hexdigest()
        result[runtime + "_assisted_audit"] = audit(report, dataset)
        result[runtime + "_paired_to_learned_header"] = {
            group: paired_outcomes(baselines[runtime]["splits"][group]["cases"], row["cases"], "staged")
            for group, row in report["splits"].items()}
    (destination / "signature-diagnostic.json").write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"fp32_assisted_wording_smoke_passed": result["fp32_assisted_audit"]["wording_smoke_passed"],
                      "native_assisted_wording_smoke_passed": result["native_assisted_audit"]["wording_smoke_passed"],
                      "infrastructure_errors": infrastructure_errors}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "dataset", "engine", "output"):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    result = run(args.source, args.dataset, args.engine, args.output)
    sys.exit(1 if result["infrastructure_errors"] else 0)
