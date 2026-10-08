## Unreleased

- Documented RAG quality/performance tradeoffs using validated CI indexing latency, process peak RSS, retrieval median/p95, and ranking quality; production ranking weights remain unchanged.

- Expanded RAG ranking-weight evaluation from 8 to 17 coding queries with overlapping-vocabulary hard negatives and documentation distractors; production weights remain unchanged.

- Added an experimental RAG ranking-weight evaluation harness with recall@3 and MRR@3 across eight synthetic coding queries; production 65/35 weights unchanged pending CI evidence.

- Added CI RAG retrieval-latency benchmark for synthetic indices of 250/1000/2000 documents, reporting median and p95 of 11 timed queries after warmup; validated in CI #37745155204; median latency 0.8125/3.2756/6.6754 ms at 250/1000/2000 documents; see docs/rag-retrieval-benchmark.md.

- Added RAG indexing scale benchmark for 250/1000/2000 synthetic source files, recording wall-clock indexing latency and process peak RSS in CI artifacts; validated on windows-2022 x64: 143.332/1881.15/6704.31 ms for 250/1000/2000 files respectively; see docs/rag-index-benchmark.md.

- Added deterministic RAG retrieval-quality fixtures covering representative JSON, authentication, SQL, DirectX, and Python coding queries with expected top-k ordering.
- Added a deterministic CI benchmark comparing 8-expert dense combination with top-2 MoE routing; initial windows-2022 x64 baseline measured 20.962 us/op dense versus 21.1726 us/op MoE (0.990x dense/MoE ratio), documenting that the current routing path is not yet a speedup.

# Changelog

All notable changes to NppAI are documented in this file.

## [0.4.0] - 2026-10-05

### Added
- Python model-format v3 metadata validation and header export tooling for MoE development, with exact layout/hash tests and fail-closed rejection of MoE files until router/expert tensor serialization is specified.
- C++ MoE inference correctness coverage and a validated expert-output combination boundary, including weighted top-k mixing, capacity fallback results, deterministic repeatability, and fail-fast shape/weight checks.
- MoE expert-capacity routing with deterministic fallback to the next ranked available expert, post-fallback weight normalization, overload rejection, and atomic load accounting.
- Deterministic MoE top-k expert router with stable expert-index tie-breaking, selected-expert softmax weights, finite-logit validation, and dedicated CTest regression coverage.
- Experimental model-format v3 architecture metadata contract for dense/MoE models, including expert-count and experts-per-token invariants while preserving v1/v2 compatibility boundaries.
- Backend abuse/error-path integration coverage verifying authentication/rate-limit ordering, non-persistence of rejected samples, validation isolation, dataset-write failure handling, and early oversized-request rejection.
- Correction-data retention and privacy policy documenting persisted fields, sensitive-data boundaries, operator retention duties, access controls, training/derived-artifact implications, logging, and incident handling.
- Production TLS termination and backend deployment requirements, including trusted-proxy forwarding, secret handling, layered request/rate limits, and a deployment checklist.
- Explicit backend authentication/authorization policy for correction-data writes, including API-key handling, authorization scope, rotation expectations, and deployment boundaries.
- Repository-level coding-agent quality evaluation metrics covering end-to-end task success, patch accuracy, build/test pass rates, workspace safety, and repair effort, with deterministic regression coverage.
- Live incremental assistant/code output in the Notepad++ panel, synchronized with generation backtracking while keeping private reasoning out of the UI.
- Concise generation status updates in the panel through `GenerationProgressEvent`.
- Structured coding-agent task outcome records with success/failure/rollback status, repair diagnostics, unified patch diff, and deterministic JSONL serialization for later correction-data pipelines.
- Deterministic end-to-end coding-agent integration coverage for successful patch validation and failed-build rollback/recovery.
- Guarded patch rollback/recovery that restores original content only when the workspace still matches the applied proposal, refusing to overwrite later user changes.
- Bounded coding-agent repair controller with a hard validation-attempt limit and ordered diagnostic history for failed repair attempts.
- Coding-agent build/test feedback loop that preserves compiler or test diagnostics as repair feedback and emits observable validation lifecycle events.
- Explicit patch review workflow with pending/accepted/rejected decisions and guarded apply that refuses stale or out-of-workspace changes.
- Reviewable structured patch model with before/after content, typed changed lines, and unified-diff preview without writing to disk.
- Workspace-bounded coding-agent file reader with canonical path validation and configurable read-size limits.
- Structured coding-agent progress events for repository file reads/changes, build and test lifecycle results, and task completion.
- Context-aware RAG retrieval that can softly prioritize the active source file and programming language while retaining relevant cross-file results.
- Repository-aware generation context for using relevant project files during AI-assisted code generation.
- Automated release workflow for version tags.
- Ready-to-install Notepad++ plugin ZIP packages for x64, x86, and ARM64.
- Installation guidance for release packages and compatible .nppai models.

### Changed
- Updated backend rate-limit tests to authenticate correction-data requests, preserving coverage after the API was changed to fail closed when no API key is configured.
- Correction-data submission now fails closed when `NPPAI_API_KEY` is not configured instead of silently accepting unauthenticated writes.
- The AI panel now displays user-visible assistant output instead of the internal thought stream; `GenerationStreamRouter` maintains a corrected assistant buffer for UI streaming.
- Extended `GenerationStreamRouter` with typed observable generation-progress events while retaining `NppAIEngine::generate()` callbacks as the low-level streaming mechanism for the planned coding agent.
- Expanded the project roadmap with a prioritized repository-aware coding-agent loop, live token/progress streaming, reviewable patches, build/test repair feedback, rollback, and agent integration testing.
- Ranked RAG retrieval now emits normal multiline context and diversifies near-duplicate chunks from the same source while preserving source/language preference bonuses.
- Fixed the Windows/MSVC build for ranked RAG retrieval by declaring the new `retrieveContextRanked` public API in `RAGManager`, and made its ranking regression test independent of the compiler code page.
- Repository indexing exclusion tests now use isolated fixtures and direct indexed-file counts, making generated/vendor/virtual-environment checks deterministic across Release builds.
- Release artifacts now use the Notepad++ plugin-directory layout and include README and model-format documentation.
- Tagged releases are built directly from the tagged source revision for easier verification and evaluation.

### Notes
- The v0.4.0 release publishes NppAI-x64.zip, NppAI-x86.zip, and NppAI-arm64.zip.
- Model files are intentionally not bundled with plugin release packages.

## [0.3.0] - 2026-10-03

### Added
- Versioned .nppai model format v2 with an explicit magic value, format version, payload length, and SHA-256 payload integrity verification.
- End-to-end generation coverage for v2 model loading, SHA-256 validation, BPE tokenization, inference, and generated output.
- Deterministic scalar-versus-SIMD parity tests for FP32 matmul, INT8 matmul, and RMSNorm.
- Tokenizer coverage for UTF-8 round trips, source-code whitespace, long inputs, and chained BPE merges.
- CI tensor benchmark regression gate with per-size FP32 and INT8 performance ceilings and failure diagnostics.

### Changed
- Added runtime AVX2/FMA capability detection on supported x64 Windows systems and scalar fallback when SIMD requirements are unavailable.
- Removed the global x64 AVX2 compiler requirement so the plugin can run on baseline x64 CPUs through scalar kernels.
- Strengthened model loading to require the exact expected tensor payload size and reject both truncated and trailing data.
- New training exports use model format v2 while the loader remains compatible with legacy v1 model files.
- Benchmark artifacts are preserved even when the performance regression gate fails.

### Security
- Model format v2 validates SHA-256 payload integrity before committing model dimensions or allocating tensor state.
- Exact payload-length checks reject malformed model files before tensor allocation.
- SHA-256 provides corruption/integrity detection only; model files are not cryptographically signed or authenticated.

### Notes
- Legacy v1 model files remain supported for compatibility but do not contain the v2 SHA-256 integrity metadata.
- Non-MSVC x86/x64 builds currently use the scalar path rather than runtime AVX2/FMA dispatch.
- The CI performance gate uses deliberately conservative ceilings to tolerate normal GitHub-hosted runner variance.

## [0.2.0] - 2026-10-03

### Added
- Native CTest coverage for tensor math, model loading, tokenizer fallback, and the generation pipeline.
- Generation-pipeline fixture that verifies BPE merging through observable public generation behavior.
- Backend tests and CI benchmark coverage.
- Contributor, architecture, model-format, testing, and security documentation.

### Changed
- Improved INT8 inference kernel performance on supported x64 AVX2/FMA systems.
- Strengthened model-header and minimum-payload validation.
- Hardened backend request-size handling with incremental request streaming.
- Moved model hashing off the async event loop and cached model versions by file metadata.
- Improved recruiter-facing project documentation and development guidance.

### Security
- Added backend request-size enforcement and documented remaining public-deployment hardening work.
- Documented the current x64 AVX2/FMA and OS YMM-state requirements.

### Notes
- The optimized x64 build currently requires AVX2, FMA, and OS YMM-state support; runtime SIMD dispatch/fallback is not yet implemented.
- Exact model payload-length validation and rejection of trailing model data remain future hardening work.
