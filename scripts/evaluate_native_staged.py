"""Frozen two-model FP32/native INT8 replay; generated code is never executed."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.audit_wording_report import audit, METRICS
from scripts.evaluate_staged_generation import run as fp32_replay, header_fits, score_output
from scripts.prompt_robustness import paired_outcomes


def native_generate(engine, model, task, prefix="", timeout=120):
    with tempfile.TemporaryDirectory() as folder:
        target, prefix_file = Path(folder) / "output.txt", Path(folder) / "prefix.txt"
        prefix_file.write_bytes(prefix.encode("utf-8"))
        command = [str(Path(engine).resolve()), str(Path(model).resolve()),
                   "[USER]: " + task + "\n[AI]:\n", str(target), str(prefix_file), "160"]
        try:
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout)
            diagnostics = {"returncode": result.returncode, "output_file_exists": target.is_file(),
                           "stdout_tail": result.stdout[-1000:], "stderr_tail": result.stderr[-1000:],
                           "infrastructure_error": result.returncode not in (0, 5) or
                           (result.returncode == 0 and not target.is_file())}
        except subprocess.TimeoutExpired:
            diagnostics = {"returncode": None, "output_file_exists": target.is_file(),
                           "infrastructure_error": True, "timeout": timeout}
        # Exit 5 is the runner's explicit empty/whitespace generation outcome.
        text = target.read_bytes().decode("utf-8", errors="replace") if target.is_file() else ""
        return text, diagnostics


def summarize(cases):
    return {"total": len(cases), "headers_accepted": sum(c["header_accepted"] for c in cases),
            "headers_exact_reference": sum(c["header_exact_reference"] for c in cases),
            **{arm: {m: sum(c[arm][m] for c in cases) for m in METRICS}
               for arm in ("staged", "single_model")}, "cases": cases}


def run(headers, bodies, dataset, engine, output):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    # Reuse strict frozen source checks and policy validation before native reads.
    fp32 = fp32_replay(headers, bodies, dataset, destination / "fp32", 38582420228,
                       True, 10000, True, "four_forms")
    native = {key: copy.deepcopy(value) for key, value in fp32.items() if key != "splits"}
    native.update(splits={}, engine_sha256=hashlib.sha256(Path(engine).read_bytes()).hexdigest(),
                  token_budget=160,
                  limits="Native INT8 versus PyTorch FP32, same frozen pair/prompt/prefix policy and 160-token budget. Runtime RMSNorm epsilon, stopping markers and numerics also differ; not a quantization-only ablation. Exposed synthetic tasks, one seed, no execution or promotion.")
    header_path = Path(headers) / "NppAI-identifier-headers.nppai"
    body_path = Path(bodies) / "NppAI-broader-instruction.nppai"
    infrastructure_errors = 0
    for group, fp_group in fp32["splits"].items():
        cases = []
        for source in fp_group["cases"]:
            task = source["prompt"]
            header, header_diagnostic = native_generate(engine, header_path, task)
            accepted = not header_diagnostic["infrastructure_error"] and header_fits(task, header, 256)
            staged, staged_diagnostic = (native_generate(engine, body_path, task, header)
                                         if accepted else ("", None))
            single, single_diagnostic = native_generate(engine, body_path, task)
            # Frozen references are used only after native generation.
            function = ast.parse(source["reference"]).body[0].name
            record = {"answer": source["reference"], "function": function}
            diagnostics = {"header": header_diagnostic, "staged": staged_diagnostic,
                           "single_model": single_diagnostic}
            infrastructure_errors += sum(d["infrastructure_error"] for d in diagnostics.values() if d)
            cases.append({"id": source["id"], "prompt": task, "reference": source["reference"],
                          "generated_header": header, "header_accepted": accepted,
                          "header_exact_reference": header == source["reference"].splitlines(keepends=True)[0],
                          "staged": score_output(record, staged), "single_model": score_output(record, single),
                          "native_diagnostics": diagnostics})
        native["splits"][group] = summarize(cases)
        print(json.dumps({"native_group": group, **{k: v for k, v in native["splits"][group].items() if k != "cases"}}), flush=True)
    native["infrastructure_errors"] = infrastructure_errors
    (destination / "native-report.json").write_text(json.dumps(native, indent=2) + "\n", encoding="utf-8")
    fp_audit, native_audit = audit(fp32, dataset), audit(native, dataset)
    paired = {group: {arm: paired_outcomes(fp32["splits"][group]["cases"], cases["cases"], arm)
                      for arm in ("staged", "single_model")} for group, cases in native["splits"].items()}
    comparison = {"source_run_id": 38582420228, "source_policy": "four_forms",
                  "fp32_audit": fp_audit, "native_audit": native_audit,
                  "paired_native_to_fp32": paired, "infrastructure_errors": infrastructure_errors,
                  "fp32_report_sha256": hashlib.sha256((destination / "fp32/staged-report.json").read_bytes()).hexdigest(),
                  "native_report_sha256": hashlib.sha256((destination / "native-report.json").read_bytes()).hexdigest(),
                  "limits": native["limits"]}
    (destination / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fp32_wording_smoke_passed": fp_audit["wording_smoke_passed"],
                      "native_wording_smoke_passed": native_audit["wording_smoke_passed"],
                      "infrastructure_errors": infrastructure_errors}), flush=True)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("headers", "bodies", "dataset", "engine", "output"):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    report = run(args.headers, args.bodies, args.dataset, args.engine, args.output)
    sys.exit(1 if report["infrastructure_errors"] else 0)
