"""Import recorded model outputs into evaluator submissions; no inference performed."""
import argparse
import json
import re
from pathlib import Path


def extract_python(output):
    if not isinstance(output, str):
        raise ValueError("output must be a string")
    fence = chr(96) * 3
    pattern = re.escape(fence) + r"(?:python|py)\s*\n(.*?)" + re.escape(fence)
    blocks = re.findall(pattern, output, flags=re.I | re.S)
    if len(blocks) > 1:
        raise ValueError("ambiguous output: multiple Python blocks")
    if len(blocks) == 1:
        return blocks[0].rstrip() + "\n"
    if fence in output:
        raise ValueError("unsupported or unclosed fenced code")
    return output.rstrip() + "\n"


def import_outputs(input_path, manifest_path, destination):
    records = json.loads(Path(input_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not isinstance(manifest, list):
        raise ValueError("records and manifest must be arrays")
    ids = {case["id"] for case in manifest}
    if len(ids) != len(manifest):
        raise ValueError("duplicate manifest id")
    outputs = {}
    for record in records:
        case_id = record["id"]
        if case_id not in ids or case_id in outputs:
            raise ValueError("unknown or duplicate output id")
        if not isinstance(case_id, str) or not case_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("invalid case id")
        outputs[case_id] = extract_python(record["output"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("destination must be empty to avoid stale submissions")
    for case_id, source in outputs.items():
        (destination / (case_id + ".py")).write_text(source, encoding="utf-8")
    return len(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submissions", required=True)
    args = parser.parse_args()
    count = import_outputs(args.input, args.manifest, args.submissions)
    print(f"Imported {count} recorded outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
