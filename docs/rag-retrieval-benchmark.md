# RAG retrieval latency baseline

Source: [successful CI #37745155204](https://github.com/deser1/NppAI/actions/runs/37745155204), Windows 2022 x64 Release benchmark job.

| Synthetic indexed documents | Median (ms) | p95 (ms) |
| ---: | ---: | ---: |
| 250 | 0.8125 | 0.8173 |
| 1,000 | 3.2756 | 3.3227 |
| 2,000 | 6.6754 | 6.9611 |

The benchmark uses deterministic, short synthetic C++ documents, a single fixed query (`calculateChecksum module input`), `retrieveContextRanked` with topK=3, 3 warmups, and 11 timed samples per index size. The displayed p95 uses the second-largest observation (index 9 of 11 after sorting). Results reflect a single CI run; they are not representative of real coding workloads or a statistically stable performance regression baseline.

Observed latency is approximately linear with document count across these fixtures. The benchmark includes query embedding, lexical scoring, sorting, and result formatting, not repository indexing or model generation. For realistic performance claims, repeat runs and test different queries, chunk lengths, language mixes, and real repositories.

Reproduce: `cmake -S . -B build-bench -A x64 -DNPPAI_BUILD_BENCHMARKS=ON`, `cmake --build build-bench --config Release --target BenchmarkRAGRetrieval`, then run `build-bench/Release/BenchmarkRAGRetrieval.exe 250` (or 1000/2000). CI archives JSON results in `tensor-benchmark-x64`.
