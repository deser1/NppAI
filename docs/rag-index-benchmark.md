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
