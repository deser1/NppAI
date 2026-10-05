# Changelog

All notable changes to NppAI are documented in this file.

## [0.4.0] - 2026-10-05

### Added
- Repository-aware generation context for using relevant project files during AI-assisted code generation.
- Automated release workflow for version tags.
- Ready-to-install Notepad++ plugin ZIP packages for x64, x86, and ARM64.
- Installation guidance for release packages and compatible .nppai models.

### Changed
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
