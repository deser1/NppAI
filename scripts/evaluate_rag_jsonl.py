#!/usr/bin/env python3
"""Evaluate ranked RAG source predictions against JSONL relevance judgments."""
import argparse
import json
import math
from pathlib import Path


def load_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("query_id"), str):
                raise ValueError(f"{path}:{line_number}: query_id must be a string")
            records.append(record)
    return records


def evaluate(judgments, predictions, k):
    if k <= 0:
        raise ValueError("k must be positive")
    by_id = {}
    for prediction in predictions:
        query_id = prediction["query_id"]
        sources = prediction.get("ranked_sources")
        if query_id in by_id or not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise ValueError(f"invalid or duplicate prediction: {query_id}")
        by_id[query_id] = sources
    recall_total = mrr_total = ndcg_total = 0.0
    count = 0
    seen_ids = set()
    for judgment in judgments:
        query_id = judgment["query_id"]
        if query_id in seen_ids:
            raise ValueError(f"duplicate judgment: {query_id}")
        seen_ids.add(query_id)
        relevant = judgment.get("relevant_sources")
        if not isinstance(relevant, list) or not all(isinstance(s, str) for s in relevant):
            raise ValueError(f"invalid relevance labels: {query_id}")
        if not relevant:
            continue  # No-hit queries require separate false-positive metrics.
        if query_id not in by_id:
            raise ValueError(f"missing prediction for: {query_id}")
        ranked = list(dict.fromkeys(by_id[query_id]))[:k]
        expected = set(relevant)
        hits = [index for index, source in enumerate(ranked, 1) if source in expected]
        recall_total += len(hits) / len(expected)
        mrr_total += 1.0 / hits[0] if hits else 0.0
        dcg = sum(1.0 / math.log2(index + 1) for index in hits)
        ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(k, len(expected)) + 1))
        ndcg_total += dcg / ideal
        count += 1
    if count == 0:
        raise ValueError("no labeled queries with relevant sources")
    return {"queries": count, "k": k, "recall_at_k": recall_total / count,
            "mrr_at_k": mrr_total / count, "ndcg_at_k": ndcg_total / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(evaluate(load_jsonl(args.judgments), load_jsonl(args.predictions), args.k), indent=2))


if __name__ == "__main__":
    main()
