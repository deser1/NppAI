"""Create a disclosed synthetic dataset with frozen task-family splits.

All templates are authored in this project; no external code is imported.
"""
import argparse
import hashlib
import json
from pathlib import Path

FAMILIES = {
    "add": ("a, b", "return a + b", "the sum of a and b"),
    "subtract": ("a, b", "return a - b", "a minus b"),
    "multiply": ("a, b", "return a * b", "the product of a and b"),
    "square": ("x", "return x * x", "x squared"),
    "negate": ("x", "return -x", "the negation of x"),
    "absolute": ("x", "return abs(x)", "the absolute value of x"),
    "length": ("values", "return len(values)", "the number of values"),
    "total": ("values", "return sum(values)", "the sum of values"),
    "largest": ("values", "return max(values)", "the maximum of nonempty values"),
    "smallest": ("values", "return min(values)", "the minimum of nonempty values"),
    "reverse": ("text", "return text[::-1]", "text reversed"),
    "lower": ("text", "return text.lower()", "text in lowercase"),
    "upper": ("text", "return text.upper()", "text in uppercase"),
    "strip": ("text", "return text.strip()", "text without leading or trailing whitespace"),
    "even": ("x", "return x % 2 == 0", "whether x is even"),
    "positive": ("x", "return x > 0", "whether x is positive"),
}
UNSEEN = {
    "sort": ("values", "return sorted(values)", "a sorted copy of values"),
    "unique": ("values", "return list(dict.fromkeys(values))", "unique values preserving their order"),
    "join": ("values", "return ''.join(values)", "all strings in values joined together"),
    "clamp": ("x, low, high", "return min(high, max(low, x))", "x clamped between low and high"),
}


def record(family, spec, index, style, split):
    args, body, description = spec
    name = f"{split}_{family}_{index}"
    signature = name + "(" + args + ")"
    prompts = [f"Implement {signature} returning {description}.",
               f"Write Python function {signature}. It must return {description}.",
               f"Define {signature} in Python to return {description}."]
    return {"id": f"{split}-{family}-{index}-{style}", "family": family,
            "function": name, "prompt": prompts[style],
            "answer": f"def {signature}:\n    {body}\n"}


def build():
    train, validation, test = [], [], []
    for family, spec in FAMILIES.items():
        for index in range(12):
            for style in range(3):
                train.append(record(family, spec, index, style, "train"))
        validation.append(record(family, spec, 0, 1, "validation"))
        test.append(record(family, spec, 0, 2, "test"))
    for family, spec in UNSEEN.items():
        for index in range(3):
            test.append(record(family, spec, index, index, "test"))
    # Exact prompts, function identifiers and responses may not cross splits.
    for field in ("prompt", "function", "answer"):
        sets = [{item[field] for item in records} for records in (train, validation, test)]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert set(x["family"] for x in train).isdisjoint(UNSEEN)
    return {"train": train, "validation": validation, "test": test}


def save(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "source": "project-authored synthetic templates",
                "limitations": "Template variations are correlated; sample count is not independent task diversity.",
                "unseen_test_families": sorted(UNSEEN), "splits": {}}
    for split, records in build().items():
        raw = json.dumps(records, indent=2, ensure_ascii=False)+"\n"
        (root/f"{split}.json").write_text(raw, encoding="utf-8")
        manifest["splits"][split] = {"count": len(records),
            "sha256": hashlib.sha256(raw.encode()).hexdigest()}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    save(parser.parse_args().output)
