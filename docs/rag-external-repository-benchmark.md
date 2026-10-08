# Reproducible external C++ / Python RAG benchmark protocol

This is a **procedure**, not a claim that NppAI has already been benchmarked on an external project. External repositories are **not cloned in CI** by this guide.

## Prepare two representative source trees

Choose one C++ repository and one Python repository with licenses permitting local inspection. Record for each: upstream URL, immutable commit SHA, license, checkout date, total tracked files, source-file count, and checkout size. Use a clean, trusted local checkout at a pinned revision. Do not run untrusted project build scripts, install its dependencies, or execute its code. NppAI's indexer reads source files; review exclusions for generated files, secrets, vendor directories, and very large inputs before proceeding.

Use the **same NppAI executable build** and machine for both repositories. On Windows, build the `BenchmarkRAGIndex` and `BenchmarkRAGRetrieval` targets in Release mode with `-DNPPAI_BUILD_BENCHMARKS=ON`.

## Measurements (PowerShell)

Set `$indexExe`, `$retrievalExe`, and `$sourcePath` to the local absolute paths. The repository must be a directory. Run five fresh-process repetitions for each binary, writing JSON Lines; a fresh process avoids mixing successive index states.

```powershell
$indexExe = "C:\\NppAI\\build-bench\\Release\\BenchmarkRAGIndex.exe"
$retrievalExe = "C:\\NppAI\\build-bench\\Release\\BenchmarkRAGRetrieval.exe"
$sourcePath = "C:\\bench-fixtures\\pinned-project"
if (!(Test-Path -LiteralPath $sourcePath -PathType Container)) { throw "Missing checkout" }
foreach ($kind in @("index", "retrieval")) {
  $exe = if ($kind -eq "index") { $indexExe } else { $retrievalExe }
  $records = @()
  1..5 | ForEach-Object {
    $raw = & $exe --repo $sourcePath
    if ($LASTEXITCODE -ne 0) { throw "$kind failed on repetition $_" }
    $parsed = $raw | ConvertFrom-Json
    if ($kind -eq "index" -and $parsed.indexed_files -le 0) { throw "No files indexed" }
    if ($kind -eq "retrieval" -and $parsed.documents -le 0) { throw "No files indexed" }
    $records += $parsed
  }
  $records | ForEach-Object { $_ | ConvertTo-Json -Compress } |
    Set-Content -Encoding utf8 "$kind-results.jsonl"
}
```

Store raw JSONL outputs, exact commit SHAs, NppAI build SHA, operating system, CPU, memory, compiler, and execution date. Repeat on each repository, saving outputs separately so one run does not overwrite another.

## Interpretation and limitations

- `BenchmarkRAGIndex` measures indexing wall time, indexed file count, and **process peak RSS high-water mark**, not isolated index memory.
- `BenchmarkRAGRetrieval` measures one fixed ranked-retrieval query (3 warmups, 11 timed queries per process). Its real-repository query string is tuned to NppAI identifiers, **not necessarily relevant to external C++ or Python projects**. Nonempty output is not proof of relevant retrieval.
- Report median of five per-process medians and the full spread, **not** the p95 from one process as a global p95.
- The indexer may not support every Python/C++ file type; compare reported indexed counts with the source tree inventory. Do not equate tracked file count with indexed document count.
- Keep real-source results separate from synthetic fixtures. Do not set a regression threshold or claim cross-language retrieval quality without a relevance-labeled query set.

## Pinned candidates (selected 2026-10-08)

| Language | Repository | Immutable Git commit | License |
| --- | --- | --- | --- |
| C++ | [fmtlib/fmt](https://github.com/fmtlib/fmt) | `35c58f084e7cd79b51997439cff68346f4c724c0` | MIT |
| Python | [pallets/click](https://github.com/pallets/click) | `2247b35ea1c47c727d7a06e51fa280e12a863ff6` | BSD-3-Clause |

These are **candidate source trees**, not completed measurements. Fetch the repository yourself in a trusted environment, verify the commit with `git rev-parse HEAD`, and supply its checkout path to the commands above. Do not run dependencies or build scripts from either repository.

```powershell
git clone https://github.com/fmtlib/fmt.git C:\bench-fixtures\fmt
git -C C:\bench-fixtures\fmt checkout --detach 35c58f084e7cd79b51997439cff68346f4c724c0
git clone https://github.com/pallets/click.git C:\bench-fixtures\click
git -C C:\bench-fixtures\click checkout --detach 2247b35ea1c47c727d7a06e51fa280e12a863ff6
```

Run each fixture separately; update `$sourcePath` and move the generated JSONL outputs between runs. Before comparing results, verify that the indexer actually supports the source extensions used by each fixture. In particular, a nonzero index count does not establish Python-language support or semantic retrieval relevance.
