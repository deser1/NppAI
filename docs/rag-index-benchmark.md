# RAG indexing scale benchmark

Source: [CI run #37742533920](https://github.com/deser1/NppAI/actions/runs/37742533920), benchmark job on windows-2022 x64 (Release), PR #94.

| Synthetic C++ source files | Indexing wall time (ms) | Time per file (ms) | Peak process RSS (bytes) | Increase in process peak RSS (bytes) |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 143.332 | 0.573328 | 4,894,720 | 2,064,384 |
| 1,000 | 1,881.15 | 1.88115 | 10,223,616 | 7,593,984 |
| 2,000 | 6,704.31 | 3.35215 | 17,924,096 | 15,282,176 |

## Interpretation

- The time per file increases with repository size in this fixture, suggesting indexing does not scale linearly. This is consistent with the existing `updateSource` implementation rebuilding the document-key set on each update, but needs profiling to attribute precisely.
- The memory metric is the **process peak working set**, not isolated RAG heap consumption; the reported growth is the difference between two high-water marks.
- This is **one CI run**, not a median or confidence interval; shared-runner noise may affect timings.
- The generated repositories contain deterministic, small C++ source files. These are **synthetic** and do not represent the mix of languages, file sizes, and directory structures in a real large repository.
- No performance regression gate is asserted from this baseline.

## Reproduce

Configure with `-DNPPAI_BUILD_BENCHMARKS=ON`, build `BenchmarkRAGIndex` in Release mode, and run `BenchmarkRAGIndex.exe 250`, `1000`, and `2000` on Windows. The executable prints one JSON record per run. CI archives those records under the `tensor-benchmark-x64` artifact.

## Next steps

Profile `updateSource` and its key-set reconstruction, then measure retrieval latency across increasing index sizes. Repeat measurements with representative real repositories and multiple runs before setting performance thresholds.


## After incremental key invalidation (PR #102)

Source: [CI run #37776896318](https://github.com/deser1/NppAI/actions/runs/37776896318), benchmark job on Windows x64 Release, after replacing the full document-key rebuild with per-source key removal.

| Files | Before (ms) | After (ms) | Before / after | After peak RSS growth (bytes) |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 143.332 | 19.4122 | 7.38x | 1,683,456 |
| 1,000 | 1,881.15 | 82.8891 | 22.69x | 6,189,056 |
| 2,000 | 6,704.31 | 167.111 | 40.12x | 12,271,616 |

The new time per file is approximately 0.078–0.084 ms across the three sizes. These are separate, single-run synthetic CI measurements, not a controlled paired benchmark; do not extrapolate the speedup as a guaranteed production result. The prior non-linear scaling observation above refers to the pre-optimization implementation.


## Real NppAI source tree (PR #104)

Source: [CI run #37827786426](https://github.com/deser1/NppAI/actions/runs/37827786426), Windows x64 Release, indexing the repository's real `src/` tree.

| Metric | Value |
| --- | ---: |
| Indexed source files | 56 |
| Indexing runs | 5 |
| Median indexing wall time | 14.3135 ms |
| Median time per indexed file | approximately 0.256 ms |
| Reported process peak RSS growth | approximately 1.90 MB |

Five run times (ms): 14.9645, 14.4342, 14.3135, 14.1249, 14.1236. The per-file figure cannot be compared directly with the synthetic fixture because file sizes, content, and languages differ. The peak-RSS metric is a process high-water mark, not a direct heap allocation measurement. These values are an initial CI baseline, not a performance gate.

## Real-repository retrieval follow-up

`BenchmarkRAGRetrieval --repo src` indexes real source files and measures warmup plus repeated ranked retrieval queries. The CI job stores `rag-real-retrieval.json` with median and p95 latency. Future validation should repeat this on larger, independently selected C++ and Python repositories, with pinned revisions and documented file counts; no external code is fetched by CI at this stage.
