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

- [x] Define authentication and authorization policy
- [x] Add rate limiting
- [x] Enforce request-size limits
- [x] Document TLS termination/deployment requirements
- [x] Define correction-data retention and privacy policy
- [x] Add abuse/error-path integration tests

### 2. Coding agent and live progress

Turn NppAI from a code-answering assistant into a repository-aware local coding agent whose observable work can be followed in the Notepad++ UI.

- [x] Use the existing `NppAIEngine::generate(..., onToken, onRemove)` callbacks as the low-level token/backtracking stream
- [x] Stream generated code and assistant output incrementally into the Notepad++ panel
- [x] Add structured coding-agent task events such as file read, file changed, build/test started, failed, passed, and task completed through `GenerationStreamRouter`
- [x] Show concise observable progress in the UI without exposing private model reasoning
- [x] Add controlled repository file-reading tools with explicit workspace boundaries
- [x] Generate structured patches/diffs instead of blindly overwriting files
- [x] Add a user review/accept/reject step before applying proposed code changes
- [x] Add a compiler/test feedback loop: task -> context -> patch -> build/test -> error feedback -> repair
- [x] Limit repair iterations and preserve diagnostics to prevent uncontrolled agent loops
- [x] Add rollback/recovery when an applied change breaks the workspace
- [x] Add deterministic integration tests for successful and failing coding-agent tasks
- [x] Record task outcomes suitable for later supervised/correction-data training
- [x] Evaluate coding quality on repository-level tasks, not only isolated generation prompts

Streaming architecture decision:

- `NppAIEngine::generate()` remains the low-level inference/token producer.
- `GenerationStreamRouter` is the UI-facing streaming boundary for code, backtracking, and observable progress events.
- Higher-level coding-agent orchestration should emit progress through the router rather than coupling Notepad++ UI code directly to the inference engine.

Target workflow:

```text
User task
   |
   v
Task planner
   |
   v
Repository context / RAG
   |
   v
NppAI model + streaming
   |
   v
Read -> Patch -> Build -> Test
                  |
               failure
                  |
                  v
            Diagnose -> Repair
                  |
                  v
             Build -> Test
                  |
                success
                  |
                  v
          Review diff -> Apply
```

### 3. Experimental Mixture-of-Experts support

- [x] Define MoE model-format metadata
- [x] Implement deterministic top-k expert routing
- [x] Add expert capacity/fallback behavior
- [x] Add C++ inference correctness tests
- [x] Extend Python export/validation tooling for MoE
- [x] Benchmark MoE against the dense baseline (CI baseline: dense 20.962 us/op, MoE top-2 21.1726 us/op; 0.990x dense/MoE ratio on windows-2022 x64)

### 4. RAG quality and scale validation

- [x] Add retrieval-quality fixtures with expected top-k results
- [ ] Measure indexing latency and memory use on larger repositories
- [ ] Benchmark retrieval latency as the index grows
- [ ] Evaluate ranking weights against representative coding queries
- [ ] Document quality/performance tradeoffs

### 5. Model quality and training

- [ ] Establish reproducible training/evaluation datasets
- [ ] Add generation-quality evaluation metrics
- [ ] Compare FP32 and INT8 end-to-end generation quality
- [ ] Validate correction-data retraining on held-out examples
- [ ] Publish reproducible model-quality reports

### 6. Release readiness

- [ ] Add end-user installation/update validation
- [ ] Document supported Notepad++ and Windows versions
- [ ] Add release smoke tests against packaged ZIP artifacts
- [ ] Complete security/privacy documentation before public backend deployment
- [ ] Prepare a stable-version readiness checklist

## Guiding principles

NppAI prioritizes local inference, measurable correctness, reproducible benchmarks, explicit fallbacks, privacy-aware data handling, and technical claims backed by tests rather than feature labels alone.
