# Release readiness: published experimental packages

The current plugin prerelease is
[v0.5.0-rc.2](https://github.com/deser1/NppAI/releases/tag/v0.5.0-rc.2).
The separate experimental
[model-factorial-v1](https://github.com/deser1/NppAI/releases/tag/model-factorial-v1)
contains the repaired checkpoint, matching service, model card, raw evidence,
recording and checksums. Neither is a stable/general-purpose model promotion.

## Verified automated checks

- Release/Debug builds for Windows x64, x86 and ARM64, native tests, backend
  tests, benchmark and restricted Linux Docker integration.
- Matching plugin ZIP contents, validator beside the DLL, standalone service
  and installation/source/model/checksum identity checks.
- Actual official portable Notepad++ **8.9.8.1 x64** on a `windows-2022` runner:
  fresh install, authenticated own-model readiness, Polish prompt round trip,
  actual generation, readable review and Apply disabled without validation.
- Published plugin rc.2 source `5ddc609` passed
  [exact-source CI](https://github.com/deser1/NppAI/actions/runs/39322405472).
  Model source `af3e532` passed
  [exact-source CI](https://github.com/deser1/NppAI/actions/runs/40414350060)
  and [training/Docker/editor/publication](https://github.com/deser1/NppAI/actions/runs/40414350103).
- Real generated factorial code passed authored tests in Linux Docker; the
  Windows recording retains the same actual response, with Apply disabled.

Only x64 has been launched in the real editor. Building x86/ARM64 packages does
not certify their editor execution. The hosted `windows-2022` test does not
certify every Windows desktop version or a user's Docker Desktop installation.

## Remaining gates

The [stable checklist](stable-readiness.md) separates verified package checks
from missing upgrade/rollback, interactive Docker Desktop + GUI Apply/Undo,
version compatibility, broader model quality and deployment-specific review.
Run `scripts/windows_validated_smoke.ps1` on the required interactive environment
and retain actual evidence. This remains unverified until that run succeeds.

The [released-model evaluation](released-model-generalization.md) freezes the
already published weights and reports real broader task results, separately
from the older factorial training/development probes. Measurement is not a
waiver of model-quality requirements. The suite becomes exposed after readout.

The historical staged pilot references an expired initialization artifact.
Its exact old experiment cannot be reproduced until those source bytes are
restored. Current release evaluation downloads permanent pinned release weights
and fails closed if their identity differs.
