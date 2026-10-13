"""Preselected wording changes and paired outcomes; no model or data tuning."""


def prompt_variants(record):
    task = record["prompt"]
    return {"original": dict(record),
            "please_prefix": {**record, "prompt": "Please " + task},
            "python_prefix": {**record, "prompt": "Write Python code. " + task},
            "code_only_suffix": {**record, "prompt": task + " Return only Python code."}}


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
