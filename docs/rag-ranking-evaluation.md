# RAG ranking weight evaluation

CI: [#37751463571](https://github.com/deser1/NppAI/actions/runs/37751463571) (PR #99), 8/8 jobs successful.

The benchmark uses 17 synthetic coding queries, hard-negative documents sharing vocabulary, and 4 cosine/lexical weight combinations. Each query has one expected source. Metrics are Recall@3 and mean reciprocal rank at 3 (MRR@3).

| Cosine / lexical weights | Recall@3 | MRR@3 |
| --- | ---: | ---: |
| 35 / 65 | 1.0000 | 0.941176 |
| 50 / 50 | 1.0000 | 0.941176 |
| **65 / 35 (production)** | **1.0000** | **0.941176** |
| 80 / 20 | 1.0000 | 0.941176 |

The four variants tied. Preserve the existing 65/35 production default; these measurements do not demonstrate an improvement from changing it.

## Limitations and next steps

- Synthetic, English-language, short documents with a small number of manually chosen queries; not representative of real repositories.
- A single CI run, no statistical confidence intervals; no independent relevance judgments.
- One relevant source per query and top-3-only metrics hide some ranking differences.
- The evaluation isolates **quality**, not latency per variant; see [retrieval scaling](rag-retrieval-benchmark.md) for latency and [index scaling](rag-index-benchmark.md) for indexing time and peak RSS.
- Extend to real repositories, multilingual queries, multiple relevant passages, NDCG@k, and compare latency before considering tuning.
