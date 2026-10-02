# Changelog

All notable changes to NppAI are documented in this file.

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
