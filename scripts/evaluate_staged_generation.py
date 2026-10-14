"""Compose learned headers with a frozen body model; never execute source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys


def valid_header(text):
    if not text.startswith("def ") or not text.endswith(":\n") or text.count("\n") != 1 or "\r" in text:
        return False
    try:
        tree = ast.parse(text + "    pass\n")
    except (SyntaxError, ValueError):
        return False
    return len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)


def header_fits(task, header, context):
    prefix = ("[USER]: " + task + "\n[AI]:\n").encode("utf-8")
    return valid_header(header) and len(prefix) + len(header.encode("utf-8")) < context


def score_output(record, output):
    syntax = requested = reference_match = False
    try:
        tree = ast.parse(output)
        syntax = any(isinstance(n, ast.FunctionDef) for n in tree.body)
        requested = any(isinstance(n, ast.FunctionDef) and n.name == record["function"] for n in tree.body)
        reference_match = ast.dump(tree) == ast.dump(ast.parse(record["answer"]))
    except (SyntaxError, ValueError):
        pass
    return {"output": output, "nonempty": bool(output.strip()),
            "syntax_any_function": syntax, "requested_function": requested,
            "reference_ast": reference_match, "exact_response": output == record["answer"]}


def check_source(report, model_bytes, manifest, manifest_key, expected_steps=6000):
    if report.get("model_sha256") != hashlib.sha256(model_bytes).hexdigest():
        raise ValueError("source model hash mismatch")
    if report.get(manifest_key) != manifest:
        raise ValueError("source dataset manifest mismatch")
    if (report.get("steps"), report.get("seed"), report.get("context")) != (expected_steps, 42, 256):
        raise ValueError("source configuration differs from the preselected step budget, seed42 or context256")


def run(headers, bodies, dataset, output, source_run, perturb_prompts=False, source_steps=6000, novel_variants=False):
    if novel_variants and not perturb_prompts:
        raise ValueError("novel variants require prompt-variant evaluation")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.identifier_curriculum import validate_initial_weights
    from scripts.train_broader_instruction_experiment import read_split
    from scripts.train_instruction_experiment import generate
    from scripts.prompt_robustness import paired_outcomes, evaluation_variants

    header_root, body_root, root = Path(headers), Path(bodies), Path(dataset)
    header_path = header_root / "NppAI-identifier-headers.nppai"
    body_path = body_root / "NppAI-broader-instruction.nppai"
    header_report = json.loads((header_root / "header-report.json").read_text(encoding="utf-8"))
    body_report = json.loads((body_root / "broader-report.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    check_source(header_report, header_path.read_bytes(), manifest, "source_manifest", source_steps)
    check_source(body_report, body_path.read_bytes(), manifest, "dataset_manifest", source_steps)
    torch.set_num_threads(2)
    hw, bw = load_weights(header_path), load_weights(body_path)
    validate_initial_weights(hw)
    validate_initial_weights(bw)
    header_model, body_model = torch_model(hw), torch_model(bw)
    report = {"source_run_id": source_run, "source_header_report": header_report,
              "source_body_report": body_report, "dataset_manifest": manifest,
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "threads": 2},
              "policy": "Use the header model's raw single-line output verbatim as the body model's assistant prefix. Reject malformed or context-overflowing headers; no reference-name correction or fallback. Greedy FP32, no optimization.",
              "limits": "Two frozen models, extra inference and memory relative to a single-model control. The body model receives a machine-generated prefix. Final whole-pipeline metrics are separate from header-only accuracy. Previously exposed synthetic tasks, one seed, no execution, INT8 or production promotion.",
              "splits": {}}
    groups = {}
    for split in ("validation", "test"):
        records = read_split(root, split, manifest)
        if perturb_prompts:
            for variant in evaluation_variants(records[0], novel_variants):
                groups[split + "/" + variant] = [evaluation_variants(record, novel_variants)[variant] for record in records]
        else:
            groups[split] = records
    if perturb_prompts:
        report["prompt_robustness"] = {"variants": list(evaluation_variants(next(iter(groups.values()))[0], novel_variants)),
            "novel_variants": novel_variants,
            "derived_groups_sha256": hashlib.sha256(json.dumps(groups, sort_keys=True).encode("utf-8")).hexdigest(),
            "limits": "Same held-out tasks repeated across wording variants, not independent new programming tasks. Added text changes token positions and formatting; results do not isolate one cause. No training, checkpoint selection or name/reference change."}
    keys = ("nonempty", "syntax_any_function", "requested_function", "reference_ast", "exact_response")
    with torch.no_grad():
        for split, records in groups.items():
            cases = []
            for record in records:
                header = generate(header_model, record["prompt"], 256)
                accepted = header_fits(record["prompt"], header, 256)
                # Acceptance does not inspect the reference function name or answer.
                staged_output = generate(body_model, record["prompt"], 256, assistant_prefix=header) if accepted else ""
                baseline_output = generate(body_model, record["prompt"], 256)
                cases.append({"id": record["id"], "prompt": record["prompt"],
                              "reference": record["answer"], "generated_header": header,
                              "header_accepted": accepted,
                              "header_exact_reference": header == record["answer"].splitlines(keepends=True)[0],
                              "staged": score_output(record, staged_output),
                              "single_model": score_output(record, baseline_output)})
            summary = {"total": len(cases), "headers_accepted": sum(c["header_accepted"] for c in cases),
                       "headers_exact_reference": sum(c["header_exact_reference"] for c in cases),
                       **{arm: {key: sum(c[arm][key] for c in cases) for key in keys}
                          for arm in ("staged", "single_model")}}
            print(json.dumps({"split": split, **summary}), flush=True)
            report["splits"][split] = {**summary, "cases": cases}
    if perturb_prompts:
        for split in ("validation", "test"):
            original = report["splits"][split + "/original"]["cases"]
            for variant in report["prompt_robustness"]["variants"]:
                group = report["splits"][split + "/" + variant]
                group["paired_to_original"] = {arm: paired_outcomes(original, group["cases"], arm)
                    for arm in ("staged", "single_model")}
                print(json.dumps({"paired_split": split + "/" + variant,
                                  "paired_to_original": group["paired_to_original"]}), flush=True)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    # Keep reproducible source checkpoints/reports with the new measurement.
    for source, folder in ((header_root, "header-source"), (body_root, "body-source")):
        target = destination / folder
        target.mkdir(exist_ok=True)
        for file in source.iterdir():
            if file.is_file() and file.suffix in (".json", ".nppai"):
                shutil.copy2(file, target / file.name)
    (destination / "staged-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", required=True)
    parser.add_argument("--bodies", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-run", type=int, required=True)
    parser.add_argument("--prompt-variants", action="store_true", help="evaluate preselected prefixes/suffix on unchanged tasks")
    parser.add_argument("--source-steps", type=int, default=6000, help="preselected source checkpoint step count")
    parser.add_argument("--novel-variants", action="store_true", help="also evaluate two wordings excluded from augmentation")
    args = parser.parse_args()
    run(args.headers, args.bodies, args.dataset, args.output, args.source_run, args.prompt_variants,
        args.source_steps, args.novel_variants)
