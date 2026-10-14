"""Deterministic training-only selection for a bounded memorization probe."""
import hashlib
import json


def select_training_records(records, name_limit=None):
    if name_limit is None:
        return list(records)
    if isinstance(name_limit, bool) or not isinstance(name_limit, int) or name_limit <= 0:
        raise ValueError("training name limit must be a positive integer")
    names = list(dict.fromkeys(record["function"] for record in records))
    if name_limit > len(names):
        raise ValueError("training name limit exceeds available names")
    selected = set(names[:name_limit])
    return [record for record in records if record["function"] in selected]


def selection_metadata(records, name_limit):
    return {"requested_name_limit": name_limit, "records": len(records),
            "names": len(set(record["function"] for record in records)),
            "record_ids": [record["id"] for record in records],
            "selection_sha256": hashlib.sha256(json.dumps(records, sort_keys=True,
                ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "method": "first N distinct training names in frozen file order; retain all their records in original order"}
