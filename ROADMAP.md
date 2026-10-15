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

## Current status and evidence (October 2026)

The completed checkboxes below describe implemented engineering milestones, **not** a production-ready autonomous coding model. CI validates native builds and deterministic test fixtures; real-world generation quality, end-user UX and production backend security require separate evidence. See the open work under model quality and release readiness.

Recent RAG regression work:

- [x] JSONL escaped-string decoding and UTF-8 round-trip regression coverage (PR #123)
- [x] Non-BMP Unicode / UTF-16 surrogate-pair round-trip regression coverage (PR #124)
- [ ] Negative JSONL cases: malformed Unicode escapes, unpaired surrogates, and control characters (proposed regression tests; merge after CI)
- [x] Replace regex-based JSONL field extraction with a bounded structural JSON reader for evaluation fixtures, including malformed-record and duplicate top-level field rejection (PR pending CI)
- [ ] Evaluate a maintained JSON library for full schema validation and broader reuse

## Next

- [ ] Measure body-only loss against full-response loss on matched varied-argument training; see [body-loss-masking](docs/body-loss-masking.md).

- [x] Diagnose frozen argument normalization: assisted alias exact85/77/73 of336 versus0, but original test16→8/7/7 of28. See [canonical-argument-binding](docs/canonical-argument-binding.md); fixed normalization fails retention and is not promoted.

- [x] Measure full-response continuation: fixed assisted function-renamed exact72→150/336, varied71/336; both new-argument0/336 and autonomous full matches zero. See [body-argument-aliases](docs/body-argument-aliases.md); original validation retention and variable binding remain open, no promotion.

- [x] Measure prompt-derived signature assistance: function-renamed exact63→72/336, but new argument AST/exact0/336 despite all correct prefixes. See [argument-signature-assistance](docs/argument-signature-assistance.md); assistance is separate from learned copying, no promotion.
- [ ] Improve or explicitly diagnose body variable binding/operation choice for argument-alias development tasks; header assistance alone does not resolve complete-reference failures.

- [x] Measure fixed/varied argument aliases with unchanged replay: both0/336 new-argument exact signatures, original hits retained20/21 versus15/21. See [header-argument-aliases](docs/header-argument-aliases.md); reject promotion of the varied policy.
- [ ] Resolve autonomous argument-name/arity copying on development diagnostics; keep assisted prompt-derived signatures explicitly separate from learned-model quality.

- [x] Measure frozen replay complete replies: original exact15/28 versus13/28 with all13 retained; renamed AST60/336, exact59/336; sort case shows values→x argument drift and nonmatching bodies in both stages. See [header-replay-pipeline](docs/header-replay-pipeline.md); no promotion.

- [x] Measure fixed50% original-example replay: original test retains19/21 baseline hits with2 lost/2 gained; renamed exact159/336 versus198/336 control. See [header-original-replay](docs/header-original-replay.md); no checkpoint promotion.
- [x] Diagnose the lost baseline header and separately measure full-output retention: argument drift in sort, all13 prior complete-reference hits retained; this measurement does not resolve copying/body quality.

- [x] Measure matched header-length continuation: renamed test81/336 versus206/336 exact signatures, both original test21/28 to0/28; evidence documented in [header-name-length](docs/header-name-length.md), no checkpoint promotion.
- [x] Evaluate a preselected original-example replay mix and report original/remapped development outcomes together; full retention remains unresolved (see header replay evidence).

- [x] Measure both frozen continuation pairs on fresh-probe-v1 and retain raw/audited paired results: both 0/144 AST/exact, signatures 2/144 versus 3/144; this is completed measurement, not a quality milestone. See [fresh-probe-v1](docs/fresh-probe-v1.md).
- [ ] Improve new-task signature copying and body/reference correctness on training-only development data; retain exposed fresh-probe-v1 for regression and reserve a later independent holdout.

### 1. Public API hardening (implementation checklist; deployment review still required)

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
- [x] Measure indexing latency and memory use on larger repositories (synthetic CI fixtures: 250/1000/2000 files; see docs/rag-index-benchmark.md)
- [x] Benchmark retrieval latency as the index grows (250/1000/2000 synthetic documents; see docs/rag-retrieval-benchmark.md)
- [x] Evaluate ranking weights against synthetic coding queries (17 fixtures; four configurations tied; see docs/rag-ranking-evaluation.md)
- [x] Document quality/performance tradeoffs (see docs/rag-quality-performance-tradeoffs.md)

### 5. Model quality and training

Experiments #145–155 are integrated into `main` through
[PR #145](https://github.com/deser1/NppAI/pull/145), merge commit
`45f6a81372a2258228a9b251db0f8b810afabd18` (14 October 2026).
All 13 PR workflows passed before integration; the separate
[post-merge CI_build](https://github.com/deser1/NppAI/actions/runs/38576031974)
also passed. This integrates experimental tooling and reports, without promoting
a checkpoint or changing the production architecture.

Experimental evidence from PR #155 (measured 13 October 2026):

- [x] Record the fixed three-arm attention/position comparison and artifact provenance in [the experiment report](docs/attention-architecture-experiment.md)
- Original staged exact reference matching: single head 13/28, four absolute heads 14/28, four rotary heads 0/28; all arms score 0/28 on both novel wording probes
- [ ] Demonstrate robustness to unseen instruction wording before checkpoint promotion
- Continued four/eight-form training (PR #163) still scores 0/28 on both novel forms. A new [evaluation-only probe](docs/fresh-probe-v1.md) freezes 36 new name/argument task instances across 12 families and four correlated wording views per instance; model scores on it are pending.
- [Task-signature diagnosis](docs/task-signature-wording-diagnostic.md) (PR #162) improves assisted original exact matching to 15/28, but novel exact scores stay 0/28; polite/Python reference AST matching reaches only 2/28. Failures persist with an explicit correct header, so header learning alone does not explain this blocker.
- [ ] Validate native FP32/INT8 compatibility for any proposed architecture change; current multi-head/RoPE checkpoints are PyTorch-only

The subsequent [four-versus-eight-form wording experiment](docs/diverse-wording-experiment.md)
(PR #159, measured 14 October 2026) scores 13/28 versus 0/28 original test exact
responses and 0/28 on both novel forms in both arms. Both actual-output wording
smoke audits failed; eight-form training is not promoted. Successful experiment
and post-merge CI do not resolve the wording robustness milestone above.

These measurements use exposed synthetic tasks and one seed. They do not complete
the broader quality and held-out evaluation milestones below.

- [ ] Establish reproducible training/evaluation datasets
- [x] Add an offline Python candidate syntax/unit-test evaluation harness (infrastructure only; see docs/code-quality-evaluation.md)
- [ ] Add generation-quality evaluation metrics on actual model outputs and held-out tasks
- [ ] Compare FP32 and INT8 end-to-end generation quality
- Narrow measurement completed in [PR #161's frozen staged replay](docs/native-staged-evaluation.md): original test exact matching 13/28 for both runtimes, novel forms 0/28 for both, with validation/output differences. Broader repository-task evaluation and native multi-head/RoPE compatibility remain open.
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
