# SubCell v2 completion evidence

This package is a **portable inventory and bounded audit summary**, not the model
weights, crop release or embedding payloads themselves. The new outputs remain in
external storage; they have not been uploaded to the frozen HF dataset by this PR.
No raw data, private host paths or internal planning history is committed here.

## Matched frozen controls: completed follow-up

Both four-channel HPA controls completed on 2026-09-24 from source `d0c61f0`:
`subcell_frozen_rybg_v2_mae` and `subcell_frozen_rybg_v2_vit`. Each has 3,332,309 cells
and 1,536 finite FP32 features over all six batches/T1–T4. Full readback checks every
feature and compares every ordered metadata column against the corresponding
adapted export. The inference implementation and locked library versions match
those used for adapted extraction; weights are the intended difference.

- [`subcell-v2-frozen-completion.json`](subcell-v2-frozen-completion.json): actual
  source, weight/input/output bindings, matching checks, resource events and tests.
- [`subcell-v2-frozen-receipts.sha256`](subcell-v2-frozen-receipts.sha256): 121
  receipts/snapshots, source/operational archives, pilots, verification and regression logs.
- [`subcell-v2-frozen-payloads.sha256`](subcell-v2-frozen-payloads.sha256): the twelve
  full frozen embedding Parquets, relative to `ARTIFACT_ROOT`.
- [`subcell-v2-frozen-weights.sha256`](subcell-v2-frozen-weights.sha256): the two HPA
  encoder/pool inputs, relative to the separate RYBG `WEIGHTS_ROOT`.

```bash
set -euo pipefail
CODE=/absolute/path/to/this/code-checkout
ARTIFACT_ROOT=/absolute/path/to/artifact-store
WEIGHTS_ROOT=/absolute/path/to/rybg-weights
cd "$ARTIFACT_ROOT"
sha256sum --check "$CODE/docs/evidence/subcell-v2-frozen-receipts.sha256"
sha256sum --check "$CODE/docs/evidence/subcell-v2-frozen-payloads.sha256"
# WEIGHTS_ROOT contains mae_contrast_supcon_model/ and vit_supcon_model/.
cd "$WEIGHTS_ROOT"
sha256sum --check "$CODE/docs/evidence/subcell-v2-frozen-weights.sha256"
```

The [checkpoint/frozen follow-up](../reviews/subcell-v2-readiness-followup.md) records
27 passing CPU tests, exact tiny-model resume checks, the corrected plateau recovery
bug, and the CI-publication permission limitation. Repeated frozen GPU pilots matched
exactly for288 cells/model; this is a bounded check, not universal reproducibility.
A separate CPU-bounded post-export crop audit rehashed **17,060 files /
436,847,345,580 bytes** against the unchanged original extraction receipt. The
producer retains its original input-rehash binding; this later audit is an additive
record, not a rewritten receipt or a guarantee of filesystem immutability.
Matched **raw** controls now exist; matched downstream processing, scientific
comparisons, public artifact distribution and manuscript synchronization remain
separate work. The original training/adapted artifacts were not regenerated.

The later [review-stack provenance corrections](../reviews/subcell-v2-provenance-corrections.md)
cover source/receipt consistency and completion ordering, with a separate read-only
historical audit. They do not change the producer attribution of these artifacts.

## Original training/adapted package: unchanged snapshot

The following inventories retain their original 2026-09-23 scope and bytes. Their
then-pending frozen-control statements are historical, not the status of the
completed follow-up above. Original reports have not been rewritten to hide the
later-discovered checkpoint defect.

- [`subcell-v2-completion.json`](subcell-v2-completion.json): six native selected
  checkpoints, both complete seed42 all-split exports, source identities, counts,
  metrics, runtime, resource events and the remaining interpretation limits.
- [`subcell-v2-receipts.sha256`](subcell-v2-receipts.sha256): 115 original receipts,
  source snapshots, operational scripts, configuration and verification records.
- [`subcell-v2-payloads.sha256`](subcell-v2-payloads.sha256): 25 large artifacts:
  12 best/last checkpoint paths, the canonical manifest and12 embedding Parquets.
  At pass100 best and last have the same hashes in these runs; extraction was still
  explicitly bound to the native **selected** checkpoint, not an automatic last-file choice.

All inventory paths are relative to an `ARTIFACT_ROOT` containing the sibling
`subcell-training/` and `mislocus-derived/` trees. Verify retained copies without
modifying them, using standard SHA256 tools:

```bash
CODE=/absolute/path/to/this/code-checkout
ARTIFACT_ROOT=/absolute/path/to/artifact-store
cd "$ARTIFACT_ROOT"
sha256sum --check "$CODE/docs/evidence/subcell-v2-receipts.sha256"
# Optional larger read: includes weights and all12 embedding payloads.
sha256sum --check "$CODE/docs/evidence/subcell-v2-payloads.sha256"
```

Missing files fail these checks; a hash is not a download URL. Raw receipts retain
their original operational paths. This inventory relocates **lookups**, not historical
configuration, preflight path bindings or the source identity of a produced artifact.
Do not rewrite original receipts to make them resemble the current PR head.

## What was established

- Production training: commit `03b1961`, code fingerprint
  `1d7ca8218d49cce12fa1ee609777c841d35e103e696e64a7e8a193d480c4f175`.
  Six fresh HPA starts, 100 passes /9,700 updates each, selected pass100, no terminal
  resume allowed. The campaign audit was rerun read-only for this PR: all600 pass
  receipts, all60 validation receipts and every logged learning rate matched the
  approved schedule. Its new report does not overwrite the earlier audit.
- Extraction: commit `e306037`, using the approved seed42 selection receipts, not
  historical gene-trained models. Two exports ×3,332,309 cells ×1,536 finite FP32
  features, all six batches, exact ordered canonical coverage and split labels.
  Full readback and source/input/output verification completed at22:20 UTC on
  2026-09-23. The checksum package rechecked all listed checkpoint/export hashes.
- A fresh crop audit verified17,060 files /436,847,345,580 bytes against the unchanged
  original extraction receipt. The checksum inventories do **not** rehash those
  407GiB of crop payloads; rerun01 `verify` for another point-in-time content audit.
- Clean extraction source:26 tests passed. Bounded GPU pilots covered288 cells/model;
  repeated pilots had maximum absolute embedding difference0. Earlier CPU native
  Lightning trajectory probes had maximum resumed parameter difference0 for both
  tiny model families. These different scopes are not interchangeable.
- Native checkpoint inspection now covers all six final runs: NumPy/CPU-Torch/CUDA
  RNG states agree across the two ranks within each run and differ across seeds
  42/43/44 within each family. The summary preserves the state digests. No GPU
  augmentation trajectory or performance-penalty ablation was rerun for this audit.
- Memory-high reclaim events occurred during extraction; no hard memory-limit or
  OOM events were recorded. Do not interpret the successful run as an absence of
  memory pressure or proof of optimal throughput.

The external producer control directory retains the exact controller/readback scripts,
script archive, fixed run spec, shared locked ledger, pilot/production statuses and
verification receipts. Their hashes are included here; the operational scripts are
not misrepresented as part of the training commit or as a general-purpose framework.
Original source archives and the approved contract remain authoritative for old runs.

## What this does not establish

- No independent parallel reviewer was available for the original artifact review.
  Later scoped AI reviews are recorded in the correction note above; neither is
  independent scientific certification. The [original report](../reviews/subcell-v2-merge-review.md)
  retains its historical scope and merge risks.
- Same-run rank RNG correlation has **not** been removed; its scientific effect is
  unmeasured. Historical missing determinism flags have not been reconstructed.
- Seed42-only exported/downstream results do not demonstrate multi-seed robustness.
  T3 allele AP is neither mislocalization performance nor unseen-allele generalization.
- The original package did not include matched frozen bulk exports; the additive
  follow-up above now binds those completed controls. Downstream policy/implementation
  review, manuscript synchronization and public artifact distribution remain separate
  work. Neither package certifies XGBoost/copairs results or licenses T4-driven tuning.

Consumers must check `extraction.json` plus the corresponding full verification
receipt and file hashes, then write only into a separate analysis root. A pilot,
partial directory or lone Parquet file is not a complete representation export.
