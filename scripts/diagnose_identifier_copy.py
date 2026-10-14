"""Measure identifier bytes under true prefixes; never execute generated code."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def identifier_span(record, context):
    """Return next-token target offsets, not input-token offsets (UTF-8 bytes)."""
    prefix = ("[USER]: " + record["prompt"] + "\n[AI]:\n").encode("utf-8")
    name = record["function"].encode("utf-8")
    answer = record["answer"].encode("utf-8")
    if not name or not answer.startswith(b"def " + name + b"("):
        raise ValueError("reference must start with its declared function")
    sequence = prefix + answer + b"\0"
    if b"\0" in prefix + answer or len(sequence) > context:
        raise ValueError("NUL or context overflow in reference")
    start = len(prefix) + len(b"def ") - 1
    return sequence, start, start + len(name)


def header_name(text):
    # Header-only metric deliberately accepts an otherwise invalid body.
    match = re.match(r"\Adef ([^\s(]+)\(", text)
    return match.group(1) if match else None


def summarize_case(record, predictions, free_output, def_output):
    expected = list(record["function"].encode("utf-8"))
    if len(predictions) != len(expected):
        raise ValueError("prediction count must equal identifier byte count")
    matches = [a == b for a, b in zip(predictions, expected)]
    return {"id": record["id"], "requested_name": record["function"],
            "teacher_forced_expected_bytes": expected,
            "teacher_forced_argmax_bytes": predictions,
            "teacher_forced_correct_bytes": sum(matches),
            "teacher_forced_total_bytes": len(expected),
            "teacher_forced_all_bytes_correct": all(matches),
            "teacher_forced_first_wrong_byte": next((i for i, ok in enumerate(matches) if not ok), None),
            "free_output": free_output, "def_only_output": def_output,
            "free_header_name_correct": header_name(free_output) == record["function"],
            "def_only_header_name_correct": header_name(def_output) == record["function"]}


def summarize(cases):
    if not cases:
        raise ValueError("empty diagnostic split")
    keys = ("teacher_forced_correct_bytes", "teacher_forced_total_bytes",
            "teacher_forced_all_bytes_correct", "free_header_name_correct",
            "def_only_header_name_correct")
    return {"total": len(cases), **{key: sum(c[key] for c in cases) for key in keys}, "cases": cases}


def run(model_path, dataset, output):
    # Keep offset/metric regression tests independent of the ML runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.train_broader_instruction_experiment import read_split
    from scripts.train_instruction_experiment import generate

    torch.set_num_threads(2)
    weights = load_weights(model_path)
    model = torch_model(weights)
    context = weights["context"]
    root = Path(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    splits = {s: read_split(root, s, manifest) for s in ("train", "validation", "test")}
    # Fixed first 16 records, not selected using model outputs or test results.
    splits["train_first16"] = splits.pop("train")[:16]
    report = {"model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
              "dataset_manifest": manifest, "context": context,
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "threads": 2},
              "limits": "FP32 synthetic diagnostic, no optimization or code execution. Teacher forcing supplies every preceding reference byte and does not measure autonomous copying. Header metrics do not require a valid body. def-only supplies no identifier. Training sample is first 16 records and is not held out. No causal architecture claim or native INT8 measurement.",
              "splits": {}}
    with torch.no_grad():
        for split, records in splits.items():
            cases = []
            for record in records:
                sequence, start, end = identifier_span(record, context)
                logits = model(torch.tensor([list(sequence[:-1])], dtype=torch.long))[0, start:end]
                predictions = logits.argmax(dim=-1).tolist()
                item = summarize_case(record, predictions,
                    generate(model, record["prompt"], context),
                    generate(model, record["prompt"], context, assistant_prefix="def "))
                targets = torch.tensor(item["teacher_forced_expected_bytes"], dtype=torch.long)
                item["teacher_forced_target_probabilities"] = logits.softmax(dim=-1).gather(1, targets[:, None]).flatten().tolist()
                cases.append(item)
            report["splits"][split] = summarize(cases)
            print(json.dumps({"split": split, **{k: v for k, v in report["splits"][split].items() if k != "cases"}}), flush=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.model, args.dataset, args.output)
