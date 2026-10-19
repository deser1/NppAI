# Release readiness: current plugin preview

The [v0.5.0-rc.1 candidate](releases/v0.5.0-rc.1.md) adds a separately packaged
inference service and unchanged research checkpoint. Its main-branch publication
requires package/portable-editor success, report-to-asset hash agreement and
successful CI_build on the exact commit. PR checks upload previews only.
This experimental candidate preserves the model-quality and manual-test limitations.

The latest published v0.4.0 release predates automatic routing, validated review,
the current Unicode panel and checked own-model setup. Current branch builds
must be verified before preparing a new release. No model-quality gate is waived.

## Automated candidate checks

The release workflow now runs for relevant PRs and manual dispatch without
publishing. It builds x64/x86/ARM64 ZIPs, includes the validator, current Windows
setup and factorial test cases, and checks package structure. Production x64 DLL
exports/name/Unicode are loaded on Windows. Only tag-triggered runs can attach
files to a GitHub Release; branch/PR checks publish Actions artifacts only.

A subsequent isolated Windows job downloads official portable Notepad++
**8.9.8.1 x64**, verifies its published SHA-256, and extracts it into a fresh
temporary directory. It installs the actual candidate ZIP, starts an authenticated
NppAI service with the pinned own dense-v2 research model, checks real readiness,
opens the plugin through Notepad++'s real menu, verifies the panel, round-trips
Polish prompt text and checks Apply is disabled without validation configuration.
The report retains editor/package/model hashes and actual UI labels. This is
an automated host installation test, not a human recording or proof of useful
model answers. No existing user installation is modified.

Docker execution of model outputs is tested separately on Linux in the functional
model experiment and sandbox-integration CI. The Windows smoke does not certify
Docker Desktop setup or GUI Apply after successful validation. ARM64/x86 packages
are built and structurally checked; only x64 is launched in the real editor.

## Remaining release gates

- Confirm the candidate package and portable-editor jobs on the final revision.
- Manually verify installation/update, Polish UI, cancellation and review/apply
  on the user's supported Windows/Notepad++/Docker Desktop combination.
- Preserve the selected model outside expiring Actions artifacts, with its hash,
  format, context and explicit research-quality limitations. It is not bundled.
- Run an independently frozen real-task functional benchmark before promoting
  weights or claiming an autonomous coding assistant.
- Prepare a candidate release description and architecture-matched assets;
  creating a tag/release is separate from a successful preview build.

Pinned model artifacts have finite retention. If the exact model cannot be
downloaded, the smoke must fail rather than silently choose another checkpoint.
