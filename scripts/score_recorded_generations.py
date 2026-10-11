"""Score recorded model generations without executing untrusted code."""
import argparse
import ast
import json
from pathlib import Path

from import_generation_outputs import extract_python


def score(records, manifest):
    if not isinstance(records, list) or not isinstance(manifest, list):
        raise ValueError("records and manifest must be arrays")
    expected = [item["id"] for item in manifest]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("manifest must have unique nonempty cases")
    if len(records) != len(expected):
        raise ValueError("record count does not match manifest")
    outputs = {}
    for item in records:
        key = item["id"]
        if key not in expected or key in outputs:
            raise ValueError("unknown or duplicate output id")
        outputs[key] = item["output"]
    cases = []
    for key in expected:
        source = ""
        error = None
        valid = False
        try:
            source = extract_python(outputs[key])
            ast.parse(source)
            valid = bool(source.strip())
        except (SyntaxError, ValueError, TypeError) as exc:
            error = str(exc)[:200]
        cases.append({"id": key, "syntax_ok": valid, "chars": len(source), "error": error})
    passed = sum(case["syntax_ok"] for case in cases)
    return {"total": len(cases), "syntax_passed": passed,
            "syntax_pass_rate": passed / len(cases),
            "note": "Static AST syntax check only; no code execution or functional correctness claims",
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    report = score(records, manifest)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
