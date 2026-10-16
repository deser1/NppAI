## Unreleased
- Add fixed both64 forward/backward and saved-autograd-tensor tracing, exact instrumentation/repeat guards and CPU coverage comparison. Measured historical AMD-only first difference at loss log-softmax and its saved tensor, while portable both64 matches a complete first step across recorded AMD/Intel pairs. Retain independently verified evidence; no full-training or baseline/model promotion claim.
- Record four-arm arithmetic results: FP64-intermediate softmax/sqrt aligns isolated operators and forward logits/loss on observed historical AMD/Intel hosts, but gradients and updated weights still differ. Preserve all reports, verified output hashes and CPU coverage; no recipe promotion.
- Add a fixed four-arm arithmetic experiment: native, FP64-intermediate attention softmax, FP64-intermediate AdamW sqrt and both, with verified frozen inputs, complete-step/native-reference guards and CPU coverage reporting. No default recipe change.
- Localize the portable-profile AMD–Intel difference to AdamW FP32 sqrt at event 523 with identical input: 276/16384 outputs differ by at most 1 ULP; retain matching prior-event fingerprints and verified selected members. No arithmetic replacement or portability claim.
- Localize first observed historical-profile difference to FP32 attention softmax with identical inputs across recorded AMD EPYC AVX512/AVX2 hosts: max absolute delta about 6e-8 (4 ULP). Preserve traced/untraced equality checks, ordered fingerprints and selected tensor verification; no Intel coverage or portability claim in this allocation.
- Add bounded CPU first-step operator snapshots with traced/untraced equivalence checks and independently provisioned runner comparisons; keep historical model identity and promotion gates unchanged.
- Diagnose eight CPU controls: exact same-host repeats but first-update divergence across AMD/Intel runners in both historical and candidate single-thread profiles. Reject cross-runner portability; retain traces, environment comparison and artifact file digests.
- Retain raw outputs before failed historical guards, record per-step CPU training traces and environment provenance, and diagnose historical versus separate single-thread CPU profiles on two runners. Require exact historical checkpoint identity before paired copy training.
- Record gated-readout results: conditional `nc_` test bytes166/308→218/308, while both arms retain0/132 complete development names/signatures and0/6 separate Docker functional passes; preserve raw evidence, no promotion.
- Add an experimental prompt-only gated source-byte readout, safe versioned Python bundle, causal/UTF-8/duplicate-byte tests and paired training plus separate restricted Docker functional probes. No native/default-model change.
- Require byte-identical same-host control repeats for aligned-copy comparisons; record cross-runner historical weight identity separately while retaining exact cached-output checks.
- Preserve recorded model-evidence hashes on Windows with LF checkout rules for docs/evidence.
- Record source-aligned copy results: exact same-host control repeat and historical reply replay succeeded, but both arms remain 0/132 development names/signatures despite lower logged auxiliary loss. Retain raw evidence and original-training losses; no checkpoint promotion.
- Add a matched training-only function-name source-position attention objective with exact control reproduction, UTF-8/causal index validation and separate complete-response versus assisted-copy evidence.
- Record name-byte diagnostic results: 5580 causal checks passed; all three frozen models remain 0/132 complete development names under gold-prefix or def-scaffold assistance. Preserve raw evidence chunks; namespace gains do not establish copying or justify promotion.
- Add frozen function-name byte diagnostics with causal-prefix checks, training/development namespace controls and independently audited raw evidence; no model changes.
- Record matched name-training results: fixed and per-draw aliases both remain 0/88 requested names/signatures and full reference matches; preserve raw reports and reject promotion based on syntax-only changes.
- Add matched fixed versus per-draw function-alias training, raw-output/retention scoring and an independently reconstructed evidence audit; production checkpoints remain unchanged.
- Record actual frozen local smoke results: 0/6 passed, 2 syntax failures, 4 missing requested callables; retain unmodified raw evidence and runner latency (52.6 ms median), without model promotion.
- Add real local-model first-pass functional smoke measurement with pinned checkpoint provenance, six authored probes, Docker isolation, latency and explicit failure categories.
- Report real-model replay metrics by initial backend and reject changed test suites between attempts.

- Add bounded model repair candidate collection with Docker validation, feedback-based retry, and tests for success and infrastructure failures.

- Add CLI for collecting real first-attempt local/online model outputs, collector tests in CI, and reproducible replay instructions.

- Add real-model candidate replay evaluator, unit tests wired into CI, and Docker-backed evaluation protocol; real-model quality results are not yet collected.


- Stage plugin proposals for Python syntax/tests in a restricted local Docker profile; connect existing feedback, bounded repair, structured diff, review and outcome components. Limit work to three candidates, permit configured online escalation after two local failures, and require validated unchanged-buffer review before one-undo Apply. Add authored test-suite configuration, container/process/editor-boundary tests and bundled validator setup documentation. Other language profiles and real-model task success remain open.

- Make plugin generation select local or configured online inference automatically using model availability and byte-context fit, with empty-local fallback before streaming, visible decision reasons, native WinHTTP transport, cancellation/error handling and signature-prefix parity. Add policy/protocol and real Windows HTTP tests.

- Add a shared local/online/hybrid Python model backend contract, dense v2 CPU generation client and authenticated inference endpoint with explicit remote-transfer opt-in, admission limits and transport/runtime tests. Plugin UI integration and autonomous tool orchestration remain open.

- Record frozen body-byte results: new-argument body-only conditional initials38/456 and suffixes141/1512, but complete argument occurrences0/456 in all three models; all4350 causal-prefix checks and raw audits pass. Difficulties include initial and later bytes; teacher forcing is not autonomous improvement.

- Add a frozen body-byte teacher-forcing diagnostic for source/full-response/body-only models with argument initial/suffix categories, UTF-8 target spans, per-occurrence truncated-prefix causal checks, pinned generation caches and raw prediction audits; no training or promotion.

- Record matched body-only loss: assisted renamed test exact84/336 versus65/336 control, new-argument AST/exact remains0/336 in both; body-only loses2 prior original test hits and autonomous valid-function syntax falls to0/1100. Actual schedules/body target exposure match; reject promotion.

- Add matched full-response versus body-only loss continuation on identical varied-argument data with50% original replay, exact draw-order/target-exposure audits, real gradient-mask tests and separate assisted/autonomous evidence; no canonical repair or model promotion.

- Record frozen binding results: assisted argument-alias exact0→85/77/73 of336 under fixed x/values/text policies, but original test16→8/7/7 of28 and renamed165→85/77/73 of336. All1100 cached outputs reproduce; reject default normalization/promotion because retention fails.

- Add frozen prompt-only argument normalization diagnostics with three fixed naming policies, AST/token-based restoration, capture/scope rejection, cached-output reproduction and raw paired audits; assisted evaluation only, no training or promotion.

- Record full-response continuation: fixed arguments improve assisted function-renamed exact72→150/336 and retain15/15 prior original test hits; varied reaches71/336 and loses3/15. New-argument AST/exact remains0/336 in both, autonomous full matches remain zero; no promotion.

- Add a matched full-response body continuation with fixed/varied argument aliases, unchanged50% original replay, frozen20k body initialization, separate assisted/autonomous raw metrics, paired retention audits and complete export/reload checks; no fresh-probe use or model promotion.

- Record assisted signature results:1100/1100 prompt-derived headers, function-renamed exact63→72/336, but new argument full AST/exact remains0/336 despite correct prefixes. Raw bodies still use unrelated names/operations; assistance is not learned copying or model promotion.

- Add an inference-only prompt-derived signature diagnostic against pinned fixed/varied learned headers with the same frozen body, complete token-renamed references, paired raw-output audits and explicit assisted labeling; no training or model promotion.

- Record negative argument-alias results: both policies0/336 exact new-argument signatures; varied training retains15/21 prior test hits versus20/21 control and worsens the sort argument/arity case. Audits pass; no checkpoint promotion.

- Add a matched argument-copying header experiment: fixed versus varied training argument aliases with unchanged50% original replay, disjoint development aliases, source/byte-budget verification, raw metrics and paired retention audits; no fresh-probe use or model promotion.

- Record frozen replay pipeline results: half-original exact original replies15/28 versus13/28 pretrained (all13 retained), renamed AST60/336 and exact59/336; zero cached-header deviations. Confirm sort argument regression had a nonmatching body already before replay; no model promotion.

- Diagnose the remaining replay header regression as argument drift (`test-sort-2-2`: values→x), and add frozen pretrained/no-replay/half-original complete-body comparison with pinned sources, cache-deviation counts, raw-output audits and no training or promotion.

- Record original-example replay results: retains19/21 original test baseline hits (2 lost,2 gained; total21/28), retains14/14 validation hits, and reaches159/336 renamed exact signatures versus198/336 without replay. Full-retention diagnostics remain false; no checkpoint promotion.

- Add a preselected header original-example replay experiment: matched6000-update control versus two original/two renamed samples per batch, training-only pools, actual draw counts, paired retention audits and export/reload checks; no fresh-probe access or model promotion.

- Record repeat-training variation for the header-length experiment: renamed test80/336 versus198/336 on rerun, with unchanged zero original-name exact matches; distinguish deterministic data and within-run reload checks from bit-identical retraining.

- Record matched header-length results: renamed test exact signatures81/336(short) versus206/336(mixed), but both original test groups fall21/28 to0/28; evidence audits pass and neither checkpoint is promoted.

- Add a matched header-only name-length continuation experiment: fixed four-form starting checkpoint, short-name control versus balanced9/12/15 lengths, development-only remapping, full source/reload verification and paired raw-output audits; no fresh-probe access or model promotion.

- Record first fresh-probe results: both frozen continuation pairs have 0/144 reference AST/exact matches; staged syntax 40/144 versus 67/144 and requested signatures 2/144 versus 3/144. Evidence audits pass, quality flags fail; no checkpoint promotion.

- Add inference-only fresh-probe evaluation for the two preselected continuation pairs, source/hash verification, signature metrics, raw-output audit and paired wording/family/name-length summaries; no training or checkpoint promotion.

- Freeze an evaluation-only fresh Python probe with 12 task families, 36 new name/argument instances and 144 correlated wording views; verify isolation, context bounds, canonical bytes and trusted reference edge cases on Linux/Windows.
- Record continuation results: original exact matching 13/28 versus 14/28, mixed known-form changes and both novel forms 0/28; no checkpoint promotion.

- Add fixed four-versus-eight-form continuation from the same verified pretrained pair, full header checkpoint initialization, source/total-budget provenance and body checkpoint reload checks; no checkpoint promotion.

- Record task-signature diagnostic: assisted original exact matching 15/28 versus cached learned-header 13/28; novel exact matching stays 0/28, with 2/28 polite/Python reference AST matches. Both assisted smoke checks fail; no checkpoint promotion.

- Add a frozen inference-only task-signature wording diagnostic, separating assisted body generation in FP32/native INT8 from cached learned-header and free-body controls; no checkpoint promotion.

- Record frozen staged native INT8 replay: original test exact matching remains 13/28 like FP32; both novel forms remain 0/28, and two validation forms each lose one exact response. Zero native infrastructure errors; no checkpoint promotion.

- Add inference-only frozen staged FP32/native INT8 comparison with a dedicated evaluation runner, matched token limits, raw-output audits and explicit native infrastructure diagnostics; existing TestEngine remains unchanged.

- Preserve frozen instruction dataset LF bytes on Windows checkouts and verify Git autocrlf behavior against dataset manifest hashes.

- Record measured four-versus-eight-form wording results: original test exact matching 13/28 versus 0/28, both novel wording probes 0/28 in both arms, and both raw-output smoke audits failed despite successful CI. No checkpoint promotion.

- Fix header-training CLI import initialization before augmentation-policy validation; add isolated direct-script regression coverage.

- Add a fixed four-versus-eight-form training wording experiment with frozen targets, strict source-policy replay checks and actual-output wording audits; defaults retain four-form augmentation.

- Audit actual six-form replay outputs against frozen cases and recomputed metrics, preserve explicit failed wording-smoke results, and distinguish infrastructure CI from model-quality checks.

- Synchronize README and roadmap with the completed integration of experiments #145–155 into main and successful post-merge CI; retain the unseen-wording and native FP32/INT8 blockers.

- Record PR #155 architecture results and artifact provenance: original staged reference matching 13/28 (single head), 14/28 (four absolute heads), 0/28 (four rotary heads); both novel wording probes fail in all arms. No checkpoint promotion.

- Add isolated PyTorch-only legacy/split-head/RoPE architecture probes with strict checkpoint configuration, causal/rotation tests and matched-budget staged wording evaluation.

- Add training-only whitespace position offsets with an independent seed, unchanged source-batch sampling, ordered-view hashes and matched-budget staged wording replay.

- Add paired fixed-budget training-only prompt augmentation and staged replay on four trained forms plus two novel wording probes, preserving frozen name splits and defaults.

- Add frozen two-stage prompt-robustness replay with paired original/prefix/suffix outcomes and unchanged requested names and reference answers.

- Add inference-only learned-header/body composition with verified frozen source checkpoints, uncorrected generated prefixes and separate whole-function versus header metrics.

- Add a paired original-prompt header-only versus full-code learning probe with unchanged frozen name splits, separate partial-output metrics and checkpoint reload evidence.

- Add fixed copying-to-code curriculum schedules, compatible full-checkpoint initialization and an equal-step direct-code control without promoting a model.

- Add a bounded fixed-format identifier echo sanity experiment with frozen name splits, separate autonomous/teacher-forced metrics and exported-checkpoint reload verification.

- Add a fixed-budget eight-versus-512-name memorization probe with training-only deterministic selection, provenance and separate seen-data reconstruction metrics.

- Diagnose identifier copying with per-byte teacher-forced predictions, autonomous and def-only headers on reloaded paired experiment checkpoints; keep these diagnostic metrics separate from syntax and code correctness.

- Add a frozen random-identifier dataset and paired response-name loss experiment; preserve raw free-generation evidence without replacing the failed 300-step checkpoint.

- Integrate explicit Python signature mode into the Notepad++ panel and AIManager, including input validation, visible help/status, synchronized prefix streaming and complete correction tracking.

- Exclude EOS from detokenized generated source and streaming callbacks; add deterministic generation regression coverage.
- Preserve LF bytes in Windows signature-prefix capture instead of translating model context to CRLF.

- Add explicit signature-guided evaluation and a bounded optional assistant-prefix file in TestEngine; report forced headers separately from generated bodies and compare PyTorch/native results.

- Add frozen instruction dataset v2 with a shared identifier format across splits; compare v1/v2 and report function-name copying failures separately from syntax and reference AST matching.

- Freeze synthetic instruction dataset v1 (576 train, 16 validation, 28 test), verify split isolation, and add a 3000-step experiment with held-out loss and raw generation reports.

- Add bounded response-only instruction training with task-level holdout and explicit baseline comparison; publish measured memorization and held-out failures.

- Compare dense checkpoint logits across PyTorch FP32, NumPy FP32/INT8 references and native INT8 inference with checkpoint hash evidence.

- Optimized incremental RAG source replacement by removing only invalidated document keys instead of rebuilding the complete key set; added regression coverage for source replacement and deletion. Indexing performance awaits CI benchmark validation.

- Documented RAG quality/performance tradeoffs using validated CI indexing latency, process peak RSS, retrieval median/p95, and ranking quality; production ranking weights remain unchanged.

- Expanded RAG ranking-weight evaluation from 8 to 17 coding queries with overlapping-vocabulary hard negatives and documentation distractors; production weights remain unchanged.

- Added an experimental RAG ranking-weight evaluation harness with recall@3 and MRR@3 across eight synthetic coding queries; production 65/35 weights unchanged pending CI evidence.

- Added CI RAG retrieval-latency benchmark for synthetic indices of 250/1000/2000 documents, reporting median and p95 of 11 timed queries after warmup; validated in CI #37745155204; median latency 0.8125/3.2756/6.6754 ms at 250/1000/2000 documents; see docs/rag-retrieval-benchmark.md.

- Added RAG indexing scale benchmark for 250/1000/2000 synthetic source files, recording wall-clock indexing latency and process peak RSS in CI artifacts; validated on windows-2022 x64: 143.332/1881.15/6704.31 ms for 250/1000/2000 files respectively; see docs/rag-index-benchmark.md.

- Added deterministic RAG retrieval-quality fixtures covering representative JSON, authentication, SQL, DirectX, and Python coding queries with expected top-k ordering.
- Added a deterministic CI benchmark comparing 8-expert dense combination with top-2 MoE routing; initial windows-2022 x64 baseline measured 20.962 us/op dense versus 21.1726 us/op MoE (0.990x dense/MoE ratio), documenting that the current routing path is not yet a speedup.

# Changelog

## Unreleased

- Add smoke checks for published release ZIPs and the production x64 DLL exports.
- Validate newly packaged release ZIPs before uploading them to GitHub Releases.
- Print checkpoint provenance and generation diagnostics in native evaluation logs.
- Fail native evaluation on failed or whitespace-only completions after preserving diagnostics.
- Document the measured empty-completion baseline and reproducible demo procedure.

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
