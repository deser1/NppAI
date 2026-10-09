"""Run capture -> import -> evaluate with a local NppAI checkpoint.

Candidate code is executed by the evaluator. Use trusted code and an isolated runner.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.capture_model_code_outputs import capture
from scripts.import_generation_outputs import import_outputs
from scripts.evaluate_code_quality import evaluate


def run_pipeline(engine, model, manifest, output_dir, timeout=120):
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output directory must be empty")
    # Capture everything before writing results; a failed generation cannot
    # leave a misleading partial evaluation report.
    outputs, metadata = capture(engine, model, manifest, timeout)
    destination.mkdir(parents=True, exist_ok=True)
    recorded = destination / "recorded-outputs.json"
    recorded.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    (destination / "generation-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    submissions = destination / "submissions"
    import_outputs(recorded, manifest, submissions)
    report = evaluate(manifest, submissions)
    (destination / "code-quality-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="evaluation/code-quality/manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        parser.error("timeout must be 1..3600 seconds")
    report = run_pipeline(args.engine, args.model, args.manifest, args.output_dir, args.timeout)
    print(json.dumps(report, indent=2))
    return 0 if report["tests_passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
