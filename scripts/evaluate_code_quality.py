"""Offline code-quality evaluator for user-supplied Python candidate files.

This tool EXECUTES candidate code. Run only on trusted code in an isolated runner.
It does not invoke an NppAI model or claim model accuracy.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def evaluate(manifest_path, submissions_dir, timeout=15):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest must be a nonempty JSON array")
    root = Path(manifest_path).resolve().parent
    submissions = Path(submissions_dir).resolve()
    results = []
    seen = set()
    for case in manifest:
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("duplicate or invalid case id")
        seen.add(case_id)
        if not case_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("case id must be alphanumeric, dash or underscore")
        fixture = (root / case["test_file"]).resolve()
        if not fixture.is_relative_to(root) or not fixture.is_file():
            raise ValueError("test fixture must exist under manifest directory")
        candidate = submissions / (case_id + ".py")
        item = {"id": case_id, "submitted": candidate.is_file(),
                "syntax_ok": False, "tests_passed": False, "status": "missing"}
        if candidate.is_file():
            with tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                shutil.copyfile(candidate, work / "solution.py")
                shutil.copyfile(fixture, work / "test_solution.py")
                try:
                    syntax = subprocess.run(
                        [sys.executable, "-I", "-m", "py_compile", "solution.py"],
                        cwd=work, capture_output=True, text=True, timeout=timeout,
                    )
                    item["syntax_ok"] = syntax.returncode == 0
                    if item["syntax_ok"]:
                        tests = subprocess.run(
                            [sys.executable, "-I", "-m", "unittest", "discover",
                             "-s", ".", "-p", "test_solution.py", "-v"],
                            cwd=work, capture_output=True, text=True, timeout=timeout,
                        )
                        item["tests_passed"] = tests.returncode == 0
                        item["status"] = "passed" if item["tests_passed"] else "test_failed"
                    else:
                        item["status"] = "syntax_failed"
                except subprocess.TimeoutExpired:
                    item["status"] = "timeout"
        results.append(item)
    total = len(results)
    return {"total": total, "submitted": sum(x["submitted"] for x in results),
            "syntax_passed": sum(x["syntax_ok"] for x in results),
            "tests_passed": sum(x["tests_passed"] for x in results),
            "pass_rate": sum(x["tests_passed"] for x in results) / total,
            "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submissions", required=True)
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()
    if args.timeout < 1 or args.timeout > 300:
        parser.error("timeout must be between 1 and 300 seconds")
    report = evaluate(args.manifest, args.submissions, args.timeout)
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["tests_passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
