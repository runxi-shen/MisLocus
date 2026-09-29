# SubCell v2: review-stack provenance corrections

This is an additive follow-up to the [checkpoint/frozen review](subcell-v2-readiness-followup.md),
not a replacement of historical producer records. The original reference remains
`d4a2293`; corrected review branches deliberately differ from that reference.

## Corrections

Independent AI review of PRs #2–#6 found three P2 issues. Re-review found three
additional consistency windows; tracing callers also found the crop completion
ordering problem in training's final hook. The corrected stack:

- Publishes crop and training completion markers only after required provenance
  recording succeeds. A ledger entry can describe an incomplete attempt; it is
  not a substitute for a completion marker and verified artifacts.
- Archives and fingerprints the same buffered source bytes; binds crop/preflight
  completion to the captured identity and rejects detected source drift.
- Pins crop/preflight source identity before inventory/cohort validation. Fresh
  training, diagnostic and campaign identities are verified against their snapshots;
  both export modes bind their invocation to the captured source.
- Resolves preflight output paths outside both input trees, including symlinks.
- Retains the early shard checksum and additionally hashes the compressed stream
  actually consumed by extraction, including bytes after the tar end marker.
- Parses and hashes one extraction-receipt buffer, retains its identity through
  preflight, rejects replacement before completion, and loads that receipt consistently.

Regression checks cover ledger failures, source edits during archival/extraction/
validation, direct and aliased protected outputs, shard replacement, trailing bytes,
and extraction-receipt replacement. These change failure handling and provenance,
not scientific settings, image transforms, sampling, losses or checkpoint selection.

## Historical provenance impact

A read-only audit rehashed **235 unique receipts/snapshots** from the unchanged
115-entry original and 121-entry frozen inventories. All **18 source snapshots**
verify internally and match their associated recorded source identities. Both
export ledgers retain the completed outputs; all six training ledger entries and
source bindings remain present. The canonical preflight is outside both inputs.

The original crop producer was `08e_prepare_subcell.py`, whose archived implementation
published its receipt last and did not call the later shared ledger recorder. Its
missing shared sidecar/ledger entry is expected; no retroactive entry was fabricated.
Training remains attributed to `03b1961`, adapted exports to `e306037`, and frozen
exports to `d0c61f0`. Old logs, receipts, source archives, models and exports were
not rewritten or regenerated. New verification is evidence about old artifacts,
not evidence that the new code produced them. Future runs get new source identities;
old checkpoints must not be made resumable by editing their identity fields.

This audit did not rescan bulk crop/model/embedding payloads. Earlier full-payload
and readback audits retain their original scope. Consistent archived records are
not proof of filesystem immutability or absence of every historical transient race.
These fixes alone do not demonstrate that old scientific artifacts need regeneration.

## Review and merge status

Corrections are propagated through the existing dependency stack with forward-only
parent synchronization, not force-pushes. PR #1 remains an unmerged historical reference.
Scoped AI review and synthetic CPU checks are not human approval or GitHub CI.
Full incremental AI reviews are complete. Authorized CI publication/checks and
human approval remain merge gates; no PR is merged or approved by this note.

Detailed red/green, correction-review, per-tip validation and historical audit records
are retained under the external job `pr3-pr5-provenance-fixes-20260924-v1`.
Historical rank-RNG correlation, unknown backend flags, seed42-only reporting and
unresolved matched downstream-processing/evaluation gates remain unchanged.

## 2026-09-25: remaining-PR follow-up

One independent reviewer per PR completed #7–#19 in batches of five/five/three.
Seven grouped issues were corrected in their introducing slices, then propagated:

- #9 requires fresh probe outputs, including collective refusal on distributed ranks.
- #10/#11 load historical extraction configuration from the saved run, not migrated
  recipes. The shared fix covers both families; #15 retires that legacy path/test.
- #14 checks frozen digests against the selected family's pinned recipe, protects
  checkpoint/receipt directories and the recorded original run without assuming
  a relocated checkpoint layout, and binds source capture in this slice, not #15.
- #16 requires this controller process to own a live service in the user manager.
  Jobs bind to its lifetime and slice; an ExecStartPre guard verifies cgroup limits
  before the job body starts. Missing or incorrect live limits stop active jobs.

Regression and review records are retained in external job
`pr9-16-sequential-fixes-20260925-v1`. These are implementation corrections, not
new production fits/exports, changes to scientific settings, or retroactive
provenance repair. Review each remote PR against its updated immediate parent;
retain parent branches and ancestry until dependent PRs have been retargeted.
