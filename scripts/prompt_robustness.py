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


def training_variants(record, policy="four_forms"):
    if policy not in ("four_forms", "eight_forms"):
        raise ValueError("unknown augmentation policy")
    variants = prompt_variants(record)
    if policy == "eight_forms":
        task = record["prompt"]
        variants.update({
            "python_task_prefix": {**record, "prompt": "Python task: " + task},
            "implement_function_prefix": {**record, "prompt": "Implement this function in Python: " + task},
            "please_produce_prefix": {**record, "prompt": "Please produce code: " + task},
            "implementation_suffix": {**record, "prompt": task + " Provide an implementation."},
        })
    return variants


def validate_training_policy(enabled, policy):
    training_variants({"prompt": ""}, policy)
    if not enabled and policy != "four_forms":
        raise ValueError("nondefault augmentation policy requires --augment-prompts")


def check_augmentation_source(report, policy):
    metadata = report.get("prompt_augmentation", {})
    expected = list(training_variants({"prompt": ""}, policy))
    if metadata.get("policy", "four_forms") != policy or metadata.get("variants") != expected:
        raise ValueError("source augmentation policy/forms differ")
    if metadata.get("source_records") != 1536 or metadata.get("derived_records") != 1536 * len(expected):
        raise ValueError("source augmentation counts differ")


def augment_training_records(records, policy="four_forms"):
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("augmentation requires nonempty records with unique source IDs")
    return [{**item, "id": str(record["id"]) + "::" + variant,
             "source_id": record["id"], "prompt_variant": variant}
            for record in records for variant, item in training_variants(record, policy).items()]


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
