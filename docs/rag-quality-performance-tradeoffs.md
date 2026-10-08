# RAG quality / performance tradeoffs

This document consolidates **measured CI baselines**, not production service-level objectives. All measurements below were collected on Windows x64 Release GitHub Actions runners with deterministic synthetic inputs; measurements are from individual runs and are not statistically stable.

## Indexing throughput and memory

Source: [CI #37742533920](https://github.com/deser1/NppAI/actions/runs/37742533920), [methodology](rag-index-benchmark.md).

| Source files | Indexing time | Time/file | Peak RSS growth |
| ---: | ---: | ---: | ---: |
| 250 | 143.332 ms | 0.573328 ms | 2,064,384 bytes |
| 1,000 | 1,881.15 ms | 1.88115 ms | 7,593,984 bytes |
| 2,000 | 6,704.31 ms | 3.35215 ms | 15,282,176 bytes |

**Tradeoff:** larger repositories increase indexed coverage but increase both indexing time and process memory high-water mark. Time per file rises substantially in these fixtures; `updateSource` reconstructs the document key set after updates and is a profiling target. RSS growth is the change in process peak working set, not exact heap allocation by the index.

## Retrieval latency versus index size

Source: [CI #37745155204](https://github.com/deser1/NppAI/actions/runs/37745155204), [methodology](rag-retrieval-benchmark.md).

| Indexed documents | Median query latency | p95 query latency |
| ---: | ---: | ---: |
| 250 | 0.8125 ms | 0.8173 ms |
| 1,000 | 3.2756 ms | 3.3227 ms |
| 2,000 | 6.6754 ms | 6.9611 ms |

**Tradeoff:** larger indices may improve recall by holding more knowledge but require longer searches. In this synthetic fixed-query fixture, latency scales roughly linearly. These numbers exclude any downstream LLM inference and network overhead; p95 comes from only 11 timed samples after 3 warmups.

## Ranking quality versus weighting

Source: [CI #37751463571](https://github.com/deser1/NppAI/actions/runs/37751463571), [methodology](rag-ranking-evaluation.md).

| Cosine / lexical | Recall@3 | MRR@3 |
| --- | ---: | ---: |
| 35 / 65 | 1.0 | 0.941176 |
| 50 / 50 | 1.0 | 0.941176 |
| **65 / 35 (current)** | **1.0** | **0.941176** |
| 80 / 20 | 1.0 | 0.941176 |

**Tradeoff:** cosine and lexical matching can favor different document types, but all tested weights tied on 17 synthetic coding queries with overlapping-vocabulary distractors. **Do not change the production 65/35 default based on this experiment.** The evaluation did not measure per-weight latency, so no measured latency-quality frontier exists yet.

## Decision and follow-up experiments

1. Keep 65/35 weights and current topK behavior until a larger independently labeled relevance dataset demonstrates a repeatable improvement.
2. Profile and optimize incremental indexing (`updateSource`) before increasing default repository size; benchmark before and after.
3. Repeat retrieval latency across representative repositories, multiple queries, language mixes, and index sizes; report distributions across CI runs.
4. Measure relevance with Recall@k, MRR@k and NDCG@k on real coding tasks and include ambiguous queries, multilingual prompts, and multiple relevant passages.
5. Evaluate candidate weight changes on **both** relevance and latency, with a held-out set, before modifying defaults.

## Scope and limitations

Synthetic fixture sizes, source contents, and query distribution differ from production workloads. The indexing and retrieval benchmarks use different fixture construction and therefore cannot be combined into an end-to-end per-query latency claim. Shared CI runner variation, sample counts, and the process-wide RSS metric limit extrapolation. No accuracy, memory, or latency guarantees are asserted.
