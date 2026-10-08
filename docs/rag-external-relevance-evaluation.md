# External RAG retrieval relevance evaluation

The existing [external timing results](rag-external-measured-results.md) measure latency, **not** whether returned code answers a developer's question. This protocol adds a separately auditable relevance evaluation before any claims about external retrieval quality.

## Dataset and ground truth

Use the pinned `fmtlib/fmt` and `pallets/click` revisions in [the fixture guide](rag-external-repository-benchmark.md). For each repository, create a checked-in JSONL query dataset with at least 10 independently reviewed developer questions. Each row should contain `query_id`, `query`, `relevant_paths` (nonempty array of repository-relative source paths), `rationale`, `fixture_repo`, and `fixture_sha`. Verify every relevant path exists at the pinned revision. Include code-navigation questions about public entry points, parsing, formatting, errors and tests; avoid relying on filenames in the query itself. Require a second reviewer or record the reason for each relevance label. Include distractors and queries with similar vocabulary.

## Evaluation procedure

1. Verify the checkout SHA, count indexable files and report which extensions were actually indexed. Fail rather than reporting quality if the relevant source files were skipped.
2. Index the repository exactly once for each fixture, and execute each query with top-k = 1, 3 and 5. Store **structured ranked source paths**, not parsed positions from formatted context text.
3. Compute `Recall@k` (at least one relevant path in the top k), `MRR@k` (reciprocal rank of the first relevant path, or zero), and `nDCG@k` for binary relevance labels. Report macro-averages and per-query results, including misses.
4. Record NppAI SHA, fixture SHA, indexable file count, query dataset SHA, ranking weights, and runtime environment with the results.
5. Compare against a simple lexical baseline on the **same** indexed corpus. Do not tune weights on the evaluation set; separate any development queries from held-out queries.

## Implementation gate

The existing `BenchmarkRAGRanking` uses eight synthetic documents plus distractors and 17 handcrafted queries; it is useful for regression testing, but is not an external-repository quality score. The existing external workflow measures index/retrieval speed with a fixed NppAI-specific query. Neither establishes relevance on fmt or Click.

Implement a structured top-k result API or test-only adapter that preserves the exact ranked source path for each result before automating these quality metrics. Add regression tests for tie handling, duplicate chunks per source, missing paths, and zero-hit queries. Publish no external Recall/MRR/nDCG values until ground-truth labels and ranked outputs are available.
