# Measured external C++ and Python RAG benchmark results

Source: [main GitHub Actions #37847014008](https://github.com/deser1/NppAI/actions/runs/37847014008), successful pinned external repository workflow. The measurements below were extracted from the successful PR workflow [#37846651203](https://github.com/deser1/NppAI/actions/runs/37846651203), artifact names `rag-external-cpp-fmt` and `rag-external-python-click`. Both workflow runs succeeded, but **numeric values below specifically come from PR run #37846651203**, not a claimed analysis of main's artifact. Windows 2022, x64 Release, five independent processes per metric and fixture.

| Metric | fmtlib/fmt (C++) | pallets/click (Python) |
| --- | ---: | ---: |
| Pinned revision | `35c58f084e7cd79b51997439cff68346f4c724c0` | `2247b35ea1c47c727d7a06e51fa280e12a863ff6` |
| Indexed files (every run) | 103 | 134 |
| Median indexing wall time | 85.8924 ms | 48.4084 ms |
| Median per-run ranked retrieval median | 153.989 ms | 64.7453 ms |
| Median process peak RSS growth (indexing) | 13,807,616 bytes | 6,090,752 bytes |

## All five indexing measurements (milliseconds)

| Run | fmt | Click |
| ---: | ---: | ---: |
| 1 | 87.2223 | 49.7024 |
| 2 | 85.6575 | 48.4057 |
| 3 | 85.5518 | 48.4084 |
| 4 | 85.8924 | 48.0036 |
| 5 | 102.821 | 48.4768 |

## All five retrieval medians (milliseconds)

| Run | fmt | Click |
| ---: | ---: | ---: |
| 1 | 154.121 | 64.7792 |
| 2 | 153.517 | 66.9264 |
| 3 | 153.734 | 64.7453 |
| 4 | 153.995 | 64.6186 |
| 5 | 153.989 | 64.528 |

## Interpretation and limitations

- The retrieval values above are the **median of five per-process medians**. The per-process p95 is **not** a pooled p95 of all timed queries.
- The two repositories differ in size, file contents, source languages, and structure. This is **not** a general C++ versus Python speed comparison.
- The fixed real-repository retrieval query refers to NppAI symbols; results reflect timing, **not verified semantic relevance** on fmt or Click.
- The indexed count may include files other than the repository's primary language; this is not evidence that all Python syntax is supported.
- Peak RSS growth is the difference between two process high-water marks, not isolated index heap allocation.
- Five runs on shared Windows CI are a reproducibility baseline, not a robust cross-machine performance regression threshold.
