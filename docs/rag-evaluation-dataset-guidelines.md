# RAG evaluation dataset guidelines

## Goal

Assess retrieval on real repository content before changing the production 65/35 cosine-to-lexical weights. Keep training/tuning queries separate from final evaluation queries.

## Dataset format

Store query judgments as JSON Lines with fields: `query_id`, `query`, `language`, `repository_revision`, `relevant_sources` (array of repository-relative paths), and `split` (`development` or `test`). Every source must exist at the pinned revision. Document the annotator and rationale separately without storing secrets.

## Coverage

- Include Polish and English queries, identifier-heavy questions, and natural-language descriptions.
- Cover authentication, HTTP, persistence, build configuration, and tests from at least two real repositories.
- Include ambiguous queries, multiple relevant sources, hard negatives, and zero-relevance cases.
- Avoid overlapping near-duplicate queries across development and test splits.

## Evaluation

Compare the fixed baseline (0.65 cosine / 0.35 lexical) against candidates 0.50/0.50, 0.35/0.65 and 0.80/0.20 using identical corpora and top-K settings.

Report Recall@1/3/5, MRR@3, NDCG@5 and p50/p95 retrieval latency. Report metrics by language and repository, and distinguish no-hit queries from missing judgments. Use the test split once after choosing weights on the development split. Do not claim a production improvement unless held-out metrics improve without material regressions.

## Safety and reproducibility

Pin repository revisions, exclude credentials and private content, fix query order and evaluation parameters, and retain the baseline result in CI artifacts. Never make CI fail solely because of machine-dependent latency variance; use latency as a reported metric.
