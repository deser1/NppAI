"""Paired local NppAI stage experiment. Record source; never execute it."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_backends import DenseLocalBackend, GenerationRequest, GenerationResponse
from scripts.evaluate_staged_generation import score_output
from staged_generation import run_staged


def evaluate(backend, records, *, tokens_per_stage=128):
    if type(tokens_per_stage) is not int or not 1 <= tokens_per_stage <= 170:
        raise ValueError("tokens_per_stage must be in [1, 170]")
    if not isinstance(records, list) or any(not isinstance(r, dict) or
            any(not isinstance(r.get(k), str) or not r[k] for k in
                ("id", "prompt", "function", "answer")) for r in records):
        raise ValueError("Expected cases with nonempty id, prompt, function and answer strings")
    ids = [r["id"] for r in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError("Dataset must be nonempty with unique case IDs")
    cases = []
    model_ids = set()
    for record in records:
        # Reference answers are used only for scoring, never for inference.
        staged = run_staged(backend, record["prompt"], tokens_per_stage=tokens_per_stage)
        if staged.model_id:
            model_ids.add(staged.model_id)
        arms = {"staged": {"trace": staged.to_dict(),
            "score": score_output(record, staged.code), "error":
            next((s.error for s in staged.stages if s.error), "")}}
        # Two controls: same final-code ceiling, and same aggregate output ceiling.
        # Neither comparison equalizes input tokens, latency or actual computation.
        for name, budget in (("direct_code_budget", tokens_per_stage),
                             ("direct_total_budget", 3 * tokens_per_stage)):
            payload = GenerationRequest(prompt=record["prompt"], max_new_tokens=budget)
            start = time.monotonic()
            arm = {"request": payload.model_dump(), "response": None,
                   "score": score_output(record, ""), "error": ""}
            try:
                response = GenerationResponse.model_validate(backend.generate(payload))
                arm["response"] = response.model_dump()
                model_ids.add(response.model_id)
                if response.generated_tokens > budget:
                    raise ValueError("Backend exceeded direct token budget")
                arm["score"] = score_output(record, response.output)
            except (ValueError, RuntimeError, OSError) as exc:
                arm["error"] = str(exc)
            arm["elapsed_seconds"] = time.monotonic() - start
            arms[name] = arm
        cases.append({"id": record["id"], "prompt": record["prompt"],
                      "reference": record["answer"], "arms": arms})
    if len(model_ids) > 1:
        raise ValueError("Paired comparison requires one unchanged checkpoint")
    names = ("staged", "direct_code_budget", "direct_total_budget")
    metrics = ("nonempty", "syntax_any_function", "requested_function", "reference_ast", "exact_response")
    return {"schema_version": 1, "model_id": next(iter(model_ids), None),
        "tokens_per_stage": tokens_per_stage, "execution_status": "not_run",
        "limits": "No training or functional execution. Exposed tasks are regression evidence only. Output-token ceilings do not equalize compute. Stage completion is not correctness.",
        "total": len(cases), "summary": {name: {
            **{metric: sum(c["arms"][name]["score"][metric] for c in cases) for metric in metrics},
            "errors": sum(bool(c["arms"][name]["error"]) for c in cases)} for name in names},
        "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-per-stage", type=int, default=128)
    args = parser.parse_args()
    try:
        raw = args.dataset.read_bytes()
        report = evaluate(DenseLocalBackend(args.model), json.loads(raw),
                          tokens_per_stage=args.tokens_per_stage)
        report["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
        report["dataset_path"] = str(args.dataset)
        report["runtime"] = {"python": sys.version, "platform": platform.platform()}
        root = Path(__file__).resolve().parent.parent
        sources = ("staged_generation.py", "scripts/evaluate_task_plan.py",
                   "model_backends.py", "scripts/check_dense_runtime_parity.py",
                   "scripts/evaluate_staged_generation.py")
        report["source_sha256"] = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report["summary"]))
        # Do not hide inference/context failures behind a successful process exit.
        return 2 if any(a["errors"] for a in report["summary"].values()) else 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
