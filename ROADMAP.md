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

- [x] Add executable standalone local/online/hybrid generation backends and an authenticated inference service using the same dense v2 runtime. See [model-backends](docs/model-backends.md).
- [x] Integrate automatic local/online selection with plugin generation, context checks, empty-local fallback and visible decision reasons. See [automatic-plugin-mode](docs/automatic-plugin-mode.md).
- [x] Connect plugin generation to a Python syntax/test profile, at most three candidates, diagnostic repairs, permitted online escalation, diff review and guarded editor Apply/Reject. See [validated-generation](docs/validated-generation.md).
- [ ] Add language-specific multi-file build/test profiles and evaluate real complete tasks; current fixture tests establish engineering behavior, not autonomous model competence.

- [x] Diagnose frozen body argument bytes: body-only conditional initials38/456 and suffixes141/1512; all models0/456 complete alias occurrences despite gold prefixes. See [body-byte-diagnostic](docs/body-byte-diagnostic.md); arbitrary-name copying remains open, no model promotion.

- [x] Measure matched body-only loss: assisted renamed test84/336 versus65/336 control, new-argument0/336 in both and body-only autonomous valid-function syntax0/1100. See [body-loss-masking](docs/body-loss-masking.md); binding and original-test retention remain open, no promotion.

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

- [x] Add real-model candidate replay evaluator with Docker syntax/function validation and first-pass/repair metrics; see [evaluation protocol](docs/real-model-e2e.md).
- [ ] Collect held-out real model outputs and report complete-task success, repair rates and local/online comparisons; do not infer model quality from fixture tests.

- [x] Provide CLI collection of actual first-attempt local/online model outputs for replay evaluation.
- [ ] Execute held-out model runs with configured checkpoints/endpoints and publish measured first-pass rates; add bounded repair-attempt collection later.

- [x] Add bounded (up to 3) real-model candidate repair collection with Docker syntax/functional validation and feedback.
- [ ] Run held-out local/online experiments, measure repair success, and integrate validated repair into the Notepad++ UI with explicit apply approval.

- [x] Add separate local/online replay metrics and reject test-suite changes between retries.
- [ ] Collect held-out actual model outputs with identical task sets and publish observed quality, latency and failure breakdowns.

- [x] Add a pinned-checkpoint, six-task local first-pass functional smoke workflow with raw outputs, latency and failure categories. See [protocol](docs/local-model-smoke.md).
- [x] Record actual frozen local smoke: 0/6 passed, 2 syntax failures and 4 missing requested callables; raw report retained, no promotion.
- [ ] Improve requested function-name/signature copying and complete bodies on training/development data, then evaluate a genuinely independent broader task set and configured online comparison.

- [x] Implement a matched 1024-update fixed/per-draw function-alias continuation with full-response targets and auditable sampling. See [protocol](docs/resampled-name-training.md).
- [x] Measure both name-training arms: 0/88 requested names/signatures and complete matches before/after; retain raw reports and explicitly vacuous retention, no promotion.
- [ ] Diagnose prompt-to-output name-byte prediction on training/development prefixes before allocating a larger copying experiment; retain independent evaluation for later.

- [x] Implement frozen name-byte inference on actual training draws, exposed development names and a matched namespace control; distinguish teacher forcing from scaffolded and complete generation. See [diagnostic](docs/name-byte-diagnostic.md).
- [x] Measure frozen name-byte diagnosis: 5580 causal checks passed; all three models remain 0/132 complete development names under both disclosed assistance modes. Familiar namespace helps bytes but does not resolve suffix copying.
- [ ] Design a training-only identifier-copy objective with source/target byte alignment and matched-namespace unseen suffix evaluation before scaling continuation; retain separate full-code and functional checks.

- [x] Implement a source-position aligned attention loss for function-name bytes with an exact prior-recipe control, unchanged inference architecture and full-response/assisted evaluation. See [experiment](docs/aligned-name-copy.md).
- [x] Measure source-aligned attention objective with exact same-host control repeat and historical reply replay: both arms remain 0/132 development names/signatures and complete matches; retain losses, raw reports and original-training regressions, no promotion.
- [ ] Evaluate an explicit source-byte copy readout or other copying mechanism on training-only diagnostics before scaling; require complete-code and independent functional evidence.

- [x] Specify gated source-byte identifier-copy readout experiment, causal safeguards, paired baseline and promotion gates; see [protocol](docs/explicit-identifier-copy-protocol.md). Implementation and measured limitations are recorded below.
- [x] Run gated source-byte copy experiment and independent Docker functional probes; reject promotion because complete-code gates fail.

- [x] Implement prompt-only gated copy distribution, versioned Python bundle and matched same-host control/repeat experiment with separate Docker functional probes.
- [x] Measure gated readout: conditional `nc_` bytes166/308→218/308, but full development names/signatures0/132 and separately authored functional probes0/6 in both arms; retain raw evidence and reject promotion. See [readout](docs/gated-copy-readout.md).
- [ ] Measure a dedicated training-only identifier-copy curriculum and gate behavior before scaling full-code training; retain independent evaluation and original-task regression gates.

- [x] Preserve raw control reports before historical guards, capture CPU/library provenance and every-step parameter/loss traces; require exact historical checkpoint identity before paired copy training.
- [x] Measure eight controls across two profiles and two runners per profile: all same-host repeats exact, both profiles diverge across recorded AMD/Intel environments at update 1; reject the portable candidate and retain evidence.
- [ ] Localize the numerical operator/backend divergence or freeze a supported hardware environment before claiming cross-runner repeatability; keep historical identity and promotion gates strict.

- [x] Localize an observed first-step boundary: FP32 attention softmax differs for identical inputs on recorded AMD EPYC AVX512/AVX2 hosts; traced/untraced steps and same-host repeats match exactly. Retain evidence; Intel absent in this allocation.
- [x] Capture Intel in a second operator allocation: portable-profile AMD–Intel traces first differ in AdamW FP32 sqrt at event 523 for identical inputs (max 1 ULP), with matching prior forward/backward/clip events.
- [ ] Test isolated attention-softmax and AdamW-sqrt arithmetic alternatives on frozen inputs and complete updates with cross-vendor coverage before changing the training recipe or claiming repeatability.

- [x] Measure four arithmetic variants on frozen inputs and complete steps: `both64` aligns isolated outputs and forward logits/loss across the observed historical AMD/Intel allocation, but backward/updated weights still diverge. Portable agreement covered AMD only; reject promotion.
- [x] Trace the observed remaining both64 mismatch on unchanged inputs: historical AMD-only loss-log-softmax output and saved tensor differ; full-training/cross-vendor generalization remains open.

- [x] Add guarded both64 operator and saved-autograd-tensor tracing with independent CPU jobs and full-step instrumentation equivalence.
- [x] Measure and retain the earliest observed both64 divergence and actual CPU coverage; full-training numerical repeatability remains open.

- [x] Measure both64 first-step traces: historical AMD-only pairs first differ at loss log-softmax (event 141), saved index 80; portable both64 matches all 799 events, 84 saves and complete updates across observed AMD/Intel pairs. Retain evidence and coverage limits.
- [ ] Predeclare fixed loss-log-softmax and backward-matmul arithmetic arms using frozen observed inputs; test complete-step and then full-training repeatability before any recipe promotion.

- [x] Observe historical-profile Intel/AMD coverage in a second both64 allocation: first differences at loss log-softmax or backward FP32 matmul for identical recorded inputs; portable first-step Intel/AMD agreement repeats. Retain comparison and verification limits.

- [x] Predeclare guarded portable native/both64 full-training repeatability with fixed 1024 updates, same-host repetitions, every-step traces and actual CPU coverage.
- [x] Measure and retain full 1024-update Intel/AMD evidence: portable both64 matches all traces/checkpoints across recorded Xeon 8573C and EPYC 7763/9V74; native first differs in parameters at update 1. Quality/promotion remain separate.

- [x] Measure portable native/both64 1024-update repeats on four AMD hosts; verify all trace/checkpoint bytes and retain reconstructed evidence plus original aggregation import failure/repair.
- [x] Obtain and audit full-schedule Intel/AMD evidence in regression allocation 38959870764 (three cross-vendor pairs); limit the result to the recorded seed/schedule/runtime/hosts.

- [x] Add actual CPU/OS qualification, existing-runner label inputs and a separate passed/failed/inconclusive full-training gate; no training on mismatched hardware or vacuous single-host equality.
- [x] Measure the bounded vendor-qualified allocation: three wrong-vendor slots skip training; one eligible AMD host yields inconclusive and strict exit 2. Preserve this result separately from the successful unrestricted Intel/AMD regression evidence.
- [ ] Supply existing vendor-specific Ubuntu 24.04 x64 runner labels for predictable hardware certification.
- [x] Evaluate frozen portable native/both64 exports with identical FP32 decoding and separate Docker functional checks: both remain 0/132 development names/signatures/exact replies and 0/6 functional passes. Retain audited raw evidence and reject promotion; see [quality readout](docs/portable-quality.md).
- [ ] Diagnose a training-only identifier-copy curriculum and complete-code regressions, then freeze a candidate before genuinely fresh broader functional evaluation; numerical repeatability alone is insufficient.
- [x] Freeze and implement the paired 256/768 identifier-copy curriculum with training-only prompt-position/gate supervision, identical initialization, same-host repeats and separate complete-code/Docker regression measurements; see [protocol](docs/copy-curriculum.md).
- [x] Measure the fixed curriculum with exact same-host repeats: 5/132 exposed development full hits versus 0/132 control, but 0/6 functional passes and original-training exact regression 4/16→1/16. Retain complete replies, gate diagnostics and all training traces; reject promotion.
- [ ] Diagnose training-only namespace-invariant sequential copying and original-task retention before scaling; freeze a candidate before fresh broader functional evaluation.
- [x] Freeze a 2×2 512-update namespace/teacher-retention pilot with unchanged original targets during warmup, train-only optimization, 64 unseen-alias train-family probes, exact same-host repeats and separate full-code/Docker regression checks; see [protocol](docs/namespace-retention.md).
- [x] Measure all four namespace/retention arms and eight exact same-host repeats: 0/64 new alias prefixes/full replies and 0/6 functional passes throughout, with all four initial exact hits lost. Retain raw replies, complete traces and teacher invariance; higher total training hits do not establish retention. Reject promotion.
- [ ] Diagnose training-family sequential pointer transitions/name termination and drift on the original exact hits before a new monotonic-copy or protected-base training protocol; keep fresh broader functional evaluation reserved.
- [x] Freeze and implement the five-model copy-transition diagnostic with 68 training-reference cases, separate autonomous/gold-prefix/scaffolded modes, exact cached replay, loaded parameter identity and first-error vocabulary/readout comparisons; see [protocol](docs/copy-transition-diagnostic.md).
- [x] Measure all 340 frozen replays: conditional pointer/byte gains do not complete 0/64 scaffolded names; original-hit first errors occur in 15 name positions and one body position, with four correct-vocabulary overrides and twelve vocabulary drifts. Retain raw causal/trajectory readouts and exit-gate failures; no promotion.
- [x] Predeclare a protected/frozen-vocabulary versus prompt-only sequential-pointer/name-exit protocol, then require autonomous full-code and original-hit retention before fresh functional evaluation.

- [x] Predeclare and implement the protected/frozen vocabulary versus prompt-only sequential pointer and learned name-exit comparison, including causal cursor safeguards and complete-reply/retention gates; see [protocol](docs/protected-sequential.md).
- [x] Measure both protected arms with repeated traces, immutable bases, raw autonomous code and original exact-hit retention before fresh broader evaluation; results and remaining failures below.

- [x] Measure both frozen-base arms with exact repeats: sequential copying retains 4/4 original exact hits, yields 27/132 exposed development exact replies (natural-name views only) and 11/64 new TRAIN-family alias replies; renamed development remains 0/88 and Docker regression 0/6. Retain audited raw replies, cursor events, traces and exported-base identity; no promotion.
- [ ] Diagnose frozen source-start/exit decisions across failed alias prefixes and six exposed functional outputs before another controlled mechanism change; keep fresh broader evaluation reserved.

- [x] Define frozen alias source-start/exit and Docker first-failure diagnostic with causal traces, assisted upper bounds and evidence gates; see [protocol](docs/alias-copy-failure-diagnostic.md). Design only.

- [x] Implement and test read-only retained alias trace extraction: byte offsets, overlapping start/exit observations, historical Docker diagnostics; 86/88 alias replies select a wrong source start, two diverge before names.
- [ ] Run pinned-checkpoint oracle-start/oracle-exit comparisons and fresh restricted Docker diagnostics; retained evidence extraction does not complete this gate.
