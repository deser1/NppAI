"""Audit raw staged replay outputs; never execute generated code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_staged_generation import score_output
from scripts.prompt_robustness import evaluation_variants

NOVEL = ("novel_polite_python", "novel_implementation")
METRICS = ("nonempty", "syntax_any_function", "requested_function", "reference_ast", "exact_response")


def audit(report, dataset):
    root = Path(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if "header_source" in report:
        sources = (report["header_source"]["dataset_manifest"],
                   report["body_source"]["dataset_manifest"])
    else:
        sources = (report["dataset_manifest"],)
    if any(source != manifest for source in sources):
        raise ValueError("report dataset manifest differs from frozen source")
    measured = {}
    for split in ("validation", "test"):
        raw = (root / (split + ".json")).read_bytes()
        spec = manifest["splits"][split]
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("frozen dataset hash mismatch: " + split)
        records = json.loads(raw)
        expected = {record["id"]: record for record in records}
        if len(expected) != len(records) or len(records) != spec["count"] or not records:
            raise ValueError("invalid frozen split IDs/count")
        for variant in evaluation_variants(records[0], True):
            key = split + "/" + variant
            group = report["splits"].get(key)
            if not isinstance(group, dict):
                raise ValueError("missing wording group: " + key)
            cases = group["cases"]
            ids = [case["id"] for case in cases]
            if len(ids) != len(set(ids)) or set(ids) != set(expected):
                raise ValueError("wording case IDs differ: " + key)
            if type(group["total"]) is not int or group["total"] != len(records):
                raise ValueError("wording total differs: " + key)
            counts = {arm: {metric: 0 for metric in METRICS}
                      for arm in ("staged", "single_model")}
            for case in cases:
                record = evaluation_variants(expected[case["id"]], True)[variant]
                if case["prompt"] != record["prompt"] or case["reference"] != record["answer"]:
                    raise ValueError("raw prompt/reference differs: " + key)
                for arm in counts:
                    output = case[arm]["output"]
                    if not isinstance(output, str):
                        raise ValueError("raw output must be text")
                    actual = score_output(record, output)
                    for metric in METRICS:
                        saved = case[arm][metric]
                        if type(saved) is not bool or saved != actual[metric]:
                            raise ValueError("raw output metric differs: " + key + "/" + metric)
                        counts[arm][metric] += actual[metric]
            for arm in counts:
                for metric, count in counts[arm].items():
                    saved = group[arm][metric]
                    if type(saved) is not int or saved != count:
                        raise ValueError("aggregate metric differs: " + key + "/" + metric)
            measured[key] = {"total": len(records), **counts}
    passed = all(measured[split + "/" + variant]["staged"]["exact_response"] >= 1
                 for split in ("validation", "test") for variant in NOVEL)
    return {"wording_smoke_passed": passed,
            "policy": "At least one staged exact reference response in each novel validation/test group.",
            "limits": "Minimal nonzero smoke check, not production readiness or general programming competence. Correlated exposed synthetic tasks; source checkpoint verification remains the replay's responsibility. No generated code executed or checkpoint promoted.",
            "groups": measured}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dataset", default="evaluation/instruction-v3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-wording-success", action="store_true")
    args = parser.parse_args(argv)
    raw = Path(args.report).read_bytes()
    try:
        result = audit(json.loads(raw), args.dataset)
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(1, "Invalid replay evidence: " + str(error) + "\n")
    result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wording_smoke_passed": result["wording_smoke_passed"],
                      "source_report_sha256": result["source_report_sha256"]}))
    return 2 if args.require_wording_success and not result["wording_smoke_passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
