"""Capture actual TestEngine outputs for an evaluation manifest.

Requires a locally built TestEngine and an existing compatible .nppai model.
Model-generated code is NOT executed here; use the evaluator separately.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def capture(engine, model, manifest, timeout=120):
    engine, model = Path(engine).resolve(), Path(model).resolve()
    if not engine.is_file() or not model.is_file():
        raise ValueError("TestEngine executable and model file are required")
    cases = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must be a nonempty JSON array")
    outputs = []
    seen = set()
    for case in cases:
        case_id, task = case["id"], case["prompt"]
        if case_id in seen or not isinstance(task, str) or not task:
            raise ValueError("duplicate id or invalid prompt")
        seen.add(case_id)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "generated.txt"
            prompt = "[USER]: " + task + "\n[AI]:\n"
            result = subprocess.run(
                [str(engine), str(model), prompt, str(target)],
                cwd=directory, capture_output=True, text=True, timeout=timeout,
            )
            # Model failures are scored as failed generations, not CI crashes.
            if result.returncode != 0 or not target.is_file():
                outputs.append({"id": case_id, "output": ""})
                continue
            outputs.append({"id": case_id, "output": target.read_text(encoding="utf-8")})
    metadata = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "engine": str(engine), "cases": len(outputs),
                "note": "Captured with TestEngine defaults; sampling may be stochastic"}
    return outputs, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        parser.error("timeout must be 1..3600 seconds")
    outputs, metadata = capture(args.engine, args.model, args.manifest, args.timeout)
    Path(args.output).write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    Path(args.metadata).write_text(json.dumps(metadata, indent=2) + "\n",
                                   encoding="utf-8")
    print("Captured", len(outputs), "generation outputs")


if __name__ == "__main__":
    main()
