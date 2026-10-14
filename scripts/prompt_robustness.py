"""Preselected wording changes and paired outcomes; no model or data tuning."""


def prompt_variants(record):
    task = record["prompt"]
    return {"original": dict(record),
            "please_prefix": {**record, "prompt": "Please " + task},
            "python_prefix": {**record, "prompt": "Write Python code. " + task},
            "code_only_suffix": {**record, "prompt": task + " Return only Python code."}}


def evaluation_variants(record, include_novel=False):
    variants = prompt_variants(record)
    if include_novel:
        variants["novel_polite_python"] = {**record, "prompt": "Please write Python code. " + record["prompt"]}
        variants["novel_implementation"] = {**record, "prompt": "Provide a Python implementation: " + record["prompt"]}
    return variants


def augment_training_records(records):
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("augmentation requires nonempty records with unique source IDs")
    return [{**item, "id": str(record["id"]) + "::" + variant,
             "source_id": record["id"], "prompt_variant": variant}
            for record in records for variant, item in prompt_variants(record).items()]


def paired_outcomes(original, changed, arm):
    def outcomes(cases):
        result = {case["id"]: bool(case[arm]["exact_response"]) for case in cases}
        if len(result) != len(cases):
            raise ValueError("duplicate case IDs in paired comparison")
        return result
    before, after = outcomes(original), outcomes(changed)
    if before.keys() != after.keys():
        raise ValueError("paired case IDs differ")
    return {"retained_exact": sum(before[k] and after[k] for k in before),
            "lost_exact": sum(before[k] and not after[k] for k in before),
            "gained_exact": sum(not before[k] and after[k] for k in before),
            "neither_exact": sum(not before[k] and not after[k] for k in before)}
