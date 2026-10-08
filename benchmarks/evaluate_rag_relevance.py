"""Evaluate structured ranked source-path results against reviewed relevance labels.

Input JSONL rows:
{"query_id":"q1","relevant_paths":["src/a.cpp"],"ranked_paths":["src/b.cpp","src/a.cpp"]}
Each ranked path is one retrieved *source*, in rank order. Duplicate paths are
collapsed at their first occurrence to avoid inflating recall or rank metrics.
"""
import argparse
import json
import math
from pathlib import Path


def evaluate(rows, ks=(1, 3, 5)):
    if not rows:
        raise ValueError("at least one query is required")
    totals = {k: {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0} for k in ks}
    seen_ids = set()
    for row in rows:
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id or query_id in seen_ids:
            raise ValueError("query_id must be a unique nonempty string")
        seen_ids.add(query_id)
        relevant = row["relevant_paths"]
        ranked = row["ranked_paths"]
        if not isinstance(relevant, list) or not relevant or not all(isinstance(p, str) and p for p in relevant):
            raise ValueError("relevant_paths must be a nonempty list of paths")
        if not isinstance(ranked, list) or not all(isinstance(p, str) and p for p in ranked):
            raise ValueError("ranked_paths must be a list of paths")
        relevant_set = set(relevant)
        ranked_unique = list(dict.fromkeys(ranked))
        for k in ks:
            top = ranked_unique[:k]
            hits = [i + 1 for i, path in enumerate(top) if path in relevant_set]
            totals[k]["recall"] += len(set(top) & relevant_set) / len(relevant_set)
            totals[k]["mrr"] += 1.0 / hits[0] if hits else 0.0
            dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_set))))
            totals[k]["ndcg"] += dcg / ideal
    return {
        "queries": len(rows),
        "metrics": {
            str(k): {name: value / len(rows) for name, value in totals[k].items()}
            for k in ks
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="JSONL with reviewed relevant_paths and ranked_paths")
    args = parser.parse_args()
    with args.jsonl.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    print(json.dumps(evaluate(rows), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
