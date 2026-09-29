# SubCell v2: skeptical-review follow-up

This supplements, rather than replaces, the historical
[merge review](subcell-v2-merge-review.md). The review base remains
`87c0ec71f676a775e9fad743f3d018d0bf2036b9`. A fresh skeptical review of `eacf0db`
found a checkpoint-lifecycle defect that the earlier review and 26 tests missed.
The unchanged internal contract has SHA256
`1ae7ad94fb06b3db2ad0b46aed51bcebf5335561b17f3343beba429a157c67fe`.

## Standards

- **Resume documentation:** corrected by the lifecycle fix below. Recovery state
  is now saved after every completed validation, not only AP improvements. Native
  best-checkpoint selection and exact-tie behavior are unchanged.
- **Reviewability:** the original 49-file / 6,874-line change remains large. The
  focused follow-up commit `d0c61f0` changes only the checkpoint callback and its
  regression test. Review dependency pins/eval-only masking, staging, training,
  extraction, supervision and documentation as separate stages. No speculative
  refactor or history rewrite was used to hide the original change size.
- **CI:** a locked synthetic-CPU workflow was prepared on local branch
  `chore/subcell-cpu-ci` (`a94b1de`), but GitHub rejected its publication because
  the OAuth credential lacks `workflow` scope. It is not part of PR #1 and no
  passing GitHub checks are claimed. Publishing it requires authorized credentials.

## Spec

### Fixed: stale recovery checkpoint on non-improving AP

Pinned Lightning 2.6.1's `save_last=True` did not update `last.ckpt` when the
monitored AP failed to improve. A constant-AP run retained pass10 / patience0
through early stopping at pass60. An interruption before `completed.json` left
the resume guard accepting this stale checkpoint.

`AlleleCheckpoint.on_validation_end` now lets native selection run first, then
saves current recovery state if native checkpointing did not already save that
step. This preserves best/tie selection while recording current optimizer,
scheduler, per-rank RNG and EarlyStopping state. Normal sanity/non-fit/duplicate
save guards still apply. Recovery is from the latest published checkpoint, not
from unsaved progress preceding an interrupted checkpoint write.

The new real-CPU-Lightning regression first failed with `10 != 20`, then passed:

- Plateau AP leaves the best checkpoint and selection receipt at pass10 unchanged.
- Interrupted pass20 has current optimizer/scheduler state and patience1.
- Its resumed weights match uninterrupted training exactly.
- Early-stopped pass60 saves patience5/stopped_epoch59; interruption before the
  completion marker cannot reopen the run.

The full clean suite passes **27 tests**. Fresh native CPU MAE and ViT probes,
including stochastic augmentation, reproduce LR/data trajectories and resumed
parameters exactly. The lock check and patch-whitespace check pass. These CPU
checks are not a new production training experiment or a distributed GPU replay.

All six original runs were fresh/uninterrupted and retain pass100/update9700
selected/last checkpoints. Their training source remains `03b1961`, not the
hardened callback commit. This defect requires neither retraining nor replacement
of the already verified seed42 exports.

### Matched frozen controls

**Complete and verified on 2026-09-24.** Both four-channel HPA models passed a
fresh real-cell loading/portable-parity check. Both 288-cell pilots and repeated
pilots passed finite-FP32, metadata and hash verification; repeated features matched
exactly. Full extraction then produced **3,332,309 cells ×1,536 finite FP32 features
per model**, all six batches/T1–T4, from clean source `d0c61f0`:

- `subcell_frozen_rybg_v2_mae`
- `subcell_frozen_rybg_v2_vit`

The full verifier checks hashes and every feature, and compares every ordered
metadata column to the corresponding completed adapted export. Image preprocessing,
cohort, feature schema, precision and locked library versions match. The inference
implementation is unchanged from adapted-export source `e306037`; model weights
are the intended difference. Both workers had memory-high reclaim events but no
recorded OOM/hard-limit events. No adapted export or training run was repeated.

The [additive inventory](../evidence/subcell-v2-frozen-completion.json) binds actual
source/operational archives, HPA weight hashes, spec, pilot/repeat/production status,
readback results, output hashes and logs: 121 receipt/snapshot checksums, twelve
full frozen Parquet checksums and two separately rooted pretrained-weight checksums.
A separate post-export crop audit rehashed all17,060 files /436,847,345,580 bytes
successfully against the original receipt, without rewriting producer bindings.
This closes the **raw frozen-export gap**, not downstream scientific comparisons
or artifact distribution.

### Retained scientific and provenance limits

- Same-run rank RNG correlation is unchanged. Distinct cells and separately drawn
  views do not imply independent rank streams. Image-dependent transforms also
  prevent inferring every historical stochastic operation from endpoint RNG states.
  Performance impact remains unmeasured. Changing this requires an approved,
  versioned seeding experiment and fresh fits, not a provenance edit.
- Missing historical backend flags remain unknown. Expanded runtime recording in
  new invocations does not reconstruct the six original training environments.
- User-approved seed42 reporting preceded T4 extraction/outcomes. Seeds43/44 fits
  remain archived, not downstream replications; no multi-seed robustness claim.
- Matched raw exports do not certify downstream normalization, zero-MAD handling,
  feature selection, XGBoost/calibration, copairs populations or manuscript claims.
  Identical downstream processing remains necessary for adaptation-only comparisons.

## Evidence and merge status

Fresh review, red/green regression and native-probe logs are retained externally
under `jobs/skeptical-pr1-review-ySzra6` and `jobs/merge-readiness-20260924-v1`.
Old reports and inventories remain unchanged historical snapshots. No training
settings, production checkpoints, crop payloads, HF mirror or adapted exports
were rewritten. No independent parallel reviewer was available.

A fresh read-only audit with the fixed code again checked all600 original sampling
passes, all60 validation records, LR traces, six selections/source bindings and
terminal guards. Their checkpoint hashes and pass100/update9700 outcomes are
unchanged. Historical backend unknowns and rank-RNG limitations were disclosed,
not silently "fixed" by relabeling artifacts or changing scientific settings.

No remaining demonstrated implementation or raw-artifact blocker was found in this
follow-up. The large-change human-review advisory and CI-publication credential
limitation remain explicit; passing GitHub CI is not claimed. PR #1 remains open
and unmerged. This is readiness for human code/artifact review, not automatic merge
approval or certification of downstream/manuscript claims.
