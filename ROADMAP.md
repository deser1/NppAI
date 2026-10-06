# NppAI Roadmap

This roadmap tracks the engineering milestones for NppAI as an experimental native AI coding assistant for Notepad++.

## Completed

- [x] Custom C++ inference engine and `.nppai` model format
- [x] Runtime AVX2/FMA detection with scalar CPU fallback
- [x] Deterministic tensor, tokenizer, model-loader, and generation tests
- [x] Reproducible CPU/INT8 and CPU/DirectX 11 benchmarks
- [x] INT8 quantization correctness and accuracy coverage
- [x] Model integrity checks and hashes
- [x] Formal release process with architecture-specific plugin packages
- [x] GUI-free Notepad++ plugin DLL contract tests
- [x] Host-level Notepad++ integration tests
- [x] Repository-aware RAG indexing with source/language metadata
- [x] Incremental RAG source updates and persisted metadata
- [x] UTF-8-safe RAG chunking and identifier-aware tokenization
- [x] Context-aware RAG ranking with source/language preferences
- [x] RAG near-duplicate diversification and deterministic regression coverage

## Next

### 1. Public API hardening

- [ ] Define authentication and authorization policy
- [ ] Add rate limiting
- [ ] Enforce request-size limits
- [ ] Document TLS termination/deployment requirements
- [ ] Define correction-data retention and privacy policy
- [ ] Add abuse/error-path integration tests

### 2. Experimental Mixture-of-Experts support

- [ ] Define MoE model-format metadata
- [ ] Implement deterministic top-k expert routing
- [ ] Add expert capacity/fallback behavior
- [ ] Add C++ inference correctness tests
- [ ] Extend Python export/validation tooling for MoE
- [ ] Benchmark MoE against the dense baseline

### 3. RAG quality and scale validation

- [ ] Add retrieval-quality fixtures with expected top-k results
- [ ] Measure indexing latency and memory use on larger repositories
- [ ] Benchmark retrieval latency as the index grows
- [ ] Evaluate ranking weights against representative coding queries
- [ ] Document quality/performance tradeoffs

### 4. Model quality and training

- [ ] Establish reproducible training/evaluation datasets
- [ ] Add generation-quality evaluation metrics
- [ ] Compare FP32 and INT8 end-to-end generation quality
- [ ] Validate correction-data retraining on held-out examples
- [ ] Publish reproducible model-quality reports

### 5. Release readiness

- [ ] Add end-user installation/update validation
- [ ] Document supported Notepad++ and Windows versions
- [ ] Add release smoke tests against packaged ZIP artifacts
- [ ] Complete security/privacy documentation before public backend deployment
- [ ] Prepare a stable-version readiness checklist

## Guiding principles

NppAI prioritizes local inference, measurable correctness, reproducible benchmarks, explicit fallbacks, privacy-aware data handling, and technical claims backed by tests rather than feature labels alone.
