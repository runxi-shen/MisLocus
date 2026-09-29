# SubCell allele-v2: protocol, completed runs and reproduction

This optional workflow trains from the canonical MisLocus crop release. The default
benchmark quickstart still consumes published, already-processed features. Those
historical features are **not** relabeled as allele-v2 outputs.

## Completed scope

All six fresh MAE/ViT × seed42/43/44 fits completed on 2026-09-23: 100 sampling
passes / 9,700 optimizer updates each. Each native selector chose pass100. The user
then chose **seed42 for both families before T4 extraction/downstream outcomes**.
Both seed42 exports completed over all six batches and T1–T4: **3,332,309 cells per
model**, each with 1,536 finite FP32 features and exact canonical identity coverage.
No model was retrained or its historical source/receipts rewritten during hardening.

| Model | Seed | Selected T3 macro allele AP | Selected pass |
|---|---:|---:|---:|
| MAE | 42 | 0.10971675464386649 | 100 |
| MAE | 43 | 0.11047012615038677 | 100 |
| MAE | 44 | 0.11093676243440515 | 100 |
| ViT | 42 | 0.11544648890729702 | 100 |
| ViT | 43 | 0.11431145377782005 | 100 |
| ViT | 44 | 0.11431371260698849 | 100 |

These are **allele-identification selection metrics**, not mislocalization accuracy,
downstream retrieval mAP, or unseen-allele performance. Seeds43/44 checkpoints are
archived, not exported in the completed seed42 campaign. Matched frozen four-channel
MAE and ViT controls subsequently completed on 2026-09-24: **3,332,309 cells each**,
all six batches/T1–T4, with full finite-FP32 readback and identical ordered metadata.
Their producer source is `d0c61f0`; the earlier adapted exports remain attributed to
`e306037` and training to `03b1961`. No retraining or adapted re-extraction occurred.
No downstream analysis results or manuscript synchronization are claimed complete.

See the original [training/adapted evidence](evidence/subcell-v2-completion.json),
[matched frozen evidence](evidence/subcell-v2-frozen-completion.json),
[verification instructions](evidence/README.md), and current
[two-axis follow-up](reviews/subcell-v2-readiness-followup.md). The earlier
[review](reviews/subcell-v2-merge-review.md) remains an unchanged historical snapshot.
The original approved planning contract is retained unchanged in the external artifact archive under SHA256
`1ae7ad94fb06b3db2ad0b46aed51bcebf5335561b17f3343beba429a157c67fe`.
The operational specification below reconciles that contract with actual execution;
it does not rewrite the original prospective methods as evidence of completed analyses.

## Scientific configuration

- Release: `anonymous-xyz96/MisLocus`, revision
  `cf70fbaedf64874907730f8b8db3a62d79149164`; B7/B8/B13/B14/B15/B16.
- Complete canonical allele IDs throughout sampling, SupCon, probe CE and selection;
  no gene collapse, localization targets, historical supervision arrays or alias resolver.
- T1/T2 train: 1,661,362 cells / 1,562 alleles. T3: 870,145 cells; fixed selection
  subset 49,922, at most32 per allele, seed2026. `MSH2_Leu440Pro` has no T3 positives
  and is omitted from macro AP, not the classifier vocabulary. T4: 800,802 cells.
  T4 supplied no representation gradients, selector statistics or vocabulary fitting.
- Native mmap uint16 crops → FP32, channel order **AGP/Mito/DNA/GFP**, bilinear
  128→955 (`0.598/0.0801`, `align_corners=False`), slice `[253:701,253:701]`,
  joint per-cell/channel/spatial min–max with epsilon1e-6. No ImageNet normalization,
  per-channel normalization or segmentation/background masks. Constant cells are
  recorded, not dropped. Resized tensors are not staged on disk.
- Full ViT-B/16 encoder (width768, 12 layers/heads, MLP3072) and two-head gated pool
  (intermediate512, dropout.2, output1536); no LoRA or frozen-encoder warm start.
  Architecture-defined fixed MAE positional embeddings remain fixed. Encoder/pool
  load strictly from the corresponding four-channel HPA artifact; decoder, projectors
  and detached probe initialize anew. The MAE decoder has 8 layers, width512,
  16 heads and MLP2048; contrastive projectors are 1536→8192→8192→512.
- Each global batch: 16 distinct alleles × 8 distinct cells =128 cells /256 views.
  Draws round-robin across available batch/physical-plate/well strata, without
  replacement within a draw; reshuffle alleles/resample cells each pass. Drop the
  incomplete final group batch. A pass visits allele groups, **not every cell**.
- MAE: reconstruction + cell contrastive + .1 allele SupCon; ViT: allele SupCon.
  Contrastive temperature.1; projectors use SyncBatchNorm in distributed fits.
  MAE reconstruction uses patch-normalized four-channel targets on removed patches.
  Both add unit-weight integer-target CE averaged across the two detached probe
  inputs. Probe: Dropout(.5)→Linear(1536,768)→ReLU→Dropout(.5)→Linear(768,1562).
  Detachment blocks direct probe gradients into encoder/pool; shared norm clipping
  can still couple update magnitudes indirectly.
- Both views receive separately drawn horizontal/vertical flips (.5 each), then
  an equal choice of affine (90°, translation.2, scale.8–1.2) or perspective
  (distortion.25, internal p=.5), jointly across channels, bilinear/zero fill.
  Intensity order: non-GFP removal(.25), upstream inverse-maximum GFP rescaling(.25),
  per-channel brightness/contrast jitter(.5 ranges), equal blur/sharpness choice,
  noise, erasing. Blur uses kernel7/sigma.1–2/outer p=.5; sharpness factor2/outer
  p=.5/inner p=.5; noise p=.5/sigma.01–.05 per channel; erasing area.02–.1,
  aspect.3–3.3/outer p=.5/inner p=.5. MAE applies intensity to view2; ViT to both.
  No post-augmentation clipping/renormalization. See `setup_transforms` and
  `resolved_protocol` for the executable definitions and the RNG caveat below.
- MAE masks .25 in view1 (588/784 patches retained); view2, validation and export
  retain all patches. Only zero-mask **evaluation** bypasses the upstream random
  permutation/RNG draw. Training zero-mask shuffling/RNG remains upstream behavior.
  ViT has no MAE patch masking; validation/export have no stochastic augmentation.
- AdamW β=(.9,.95), epsilon1e-8. Encoder/decoder decay.05 except bias/1D parameters0;
  pool/projector/probe decay.01. One peak LR5e-5, from base1e-4×128/256; global clip1;
  no accumulation or mixed precision. FP32 tensors with `matmul_precision=high`
  do not imply full-IEEE internal matmul arithmetic or cross-hardware bitwise identity.
- D=97 updates/pass, warmup W=485, horizon S=9700. At update u, LR is
  `5e-5*u/W` before W, then `5e-5*(.001+.999*(S-u)/(S-W))`. Native LambdaLR starts
  at zero and advances after updates. The last applied LR is above the floor;
  the post-final scheduled LR is5e-8. The zero-LR first update still updates moments.
- Validate every10 passes. Whole-set softmax one-vs-rest AP, unweighted across
  supported alleles, deduplicates distributed padding. AP statistics/native selection
  are FP64; model/probabilities/features stay FP32. Native highest-AP checkpoint,
  earliest exact tie; early stopping patience5 **checks**, min_delta.001. All actual
  runs reached the100-pass limit rather than early stopping.

## Remaining caveats and pre-analysis gates

1. **Within-run rank RNG correlation is retained and disclosed.** Each distributed
   rank calls `seed_everything` with the same run seed; no rank-specific stream offset
   is applied. Saved NumPy/CPU-Torch/CUDA states agree across ranks within inspected
   completed runs. The biological cells are nevertheless distinct, and view1/view2
   are drawn separately. Different run seeds change sampling, stochastic transforms,
   masking/dropout and new initialization; the regression tests and checkpoint audit
   distinguish these from same-run rank correlation. Its performance impact is
   **unmeasured**. Single-seed reporting does not eliminate it. If independent rank
   streams are required, approve a new versioned seeding protocol, test DDP/resume,
   and rerun affected experiments; do not silently patch or relabel these artifacts.
2. **Single-seed exports do not establish seed robustness.** Do not publish a
   downstream mean/std, confidence interval or ensemble claim across seeds from these
   two exports. Seeds43/44 exist as training artifacts, not downstream replications.
   Do not choose a different reporting seed from T4 outcomes.
3. **Historical backend recording is incomplete.** Training recorded FP32/TF32 and
   other runtime settings, but did not explicitly record all determinism flags.
   Later hardening records cuDNN determinism, deterministic algorithms/warn-only and
   CUBLAS workspace for future runs/export. This does not recover unknown old values
   or establish exact upstream deterministic-cuDNN parity. No backend behavior was
   retrospectively changed. Matching GPU pilots were exactly repeatable for288 cells
   per model; this is not universal bitwise reproducibility.
4. **Matched raw controls are complete; downstream matching remains a gate.** The
   frozen four-channel MAE/ViT exports have the same cohort, image preprocessing,
   precision, feature schema and metadata as the adapted exports. Use identical
   downstream processing before claiming benefit from fine-tuning. Historical RBG
   features are still not an adaptation-only control. Raw-export completion is not
   a completed performance comparison.
5. **Downstream policies are separate.** All-split inference does not authorize T4
   fitting/tuning. Fix evaluator train/test roles, copairs query/context/positive
   rules, normalization/feature-selection populations, zero-MAD and calibration/hit
   policies in the downstream lane. Never normalize published `features.parquet`
   twice. Allele separation alone is not evidence of mislocalization.

No reviewed evidence requires discarding the completed artifacts solely for items
1–3. This is a bounded direct review, not an independent endorsement or a claim that
those limitations have been experimentally removed.

## Storage, provenance and completion

Use external, versioned storage; keep the HF mirror and existing crops/models fixed:

```text
ARTIFACT_ROOT/
├── subcell-training/<revision>/
│   ├── crops/                         # Native bytes; extraction receipt
│   ├── cohort-common-preparer/        # Canonical manifest/vocabulary/T3 list
│   ├── production-20260923-v2/runs/    # Six original training runs
│   └── jobs/<export-id>/              # Spec, source, logs, ledger, verification
└── mislocus-derived/<revision>/
    ├── exports/<export-id>/<representation>/<batch>/embeddings.parquet
    └── analyses/<analysis-id>/         # Separate downstream-owned writable outputs
```

The exported columns include the required control/platemap metadata,
`Metadata_BatchQualifiedCellID`, `Metadata_Batch`, `Metadata_Split` and
`SubCell_0`…`SubCell_1535`. Preflight uses CP **plate annotations only**, via complete
many-to-one left joins; it never intersects crop cells with CP's filtered cohort.

Each producer records its actual source archive/fingerprint, inputs, checkpoint or
frozen-weight binding and runtime. Training includes configuration, sampled indices,
losses, clipping, sampled parameter drift, timings and per-rank memory/RNG state.
Hashes bind bytes, not scientific validity or numerical invariance across hardware.

`extraction.json` is published atomically **last**, after all six batches and ledger
writes succeed. It binds input hashes, the selected checkpoint/receipt or frozen
weight hash, crop-rehash receipt, source, feature order, ordered IDs and per-batch
hashes/counts. File existence alone is not success. Pilots write `pilot.json`, never `extraction.json`.
The completed campaign additionally performed full readback: finite FP32 features,
exact ordered IDs/splits, schema, source and hashes; see `verified-production-*.json`
in its external control directory. Consumers must verify these bindings and write
elsewhere. Partial exports cannot resume/overwrite success; use a fresh destination.

The shared ledger is a locked invocation index, not the authority for completion.
Use explicit `--provenance-log` outside inputs/exports; without it the legacy default
is the code checkout's `data/provenance_log.json`. Routine preflight verifies saved
hashes and crop size/mtime, not a new407-GiB payload scan; run01 `verify` when a fresh
content audit is needed. A separate post-frozen-export audit rehashed all17,060 crop
files /436,847,345,580 bytes successfully. It supplements, rather than rewrites, the
producer's original input-rehash binding. Such a rehash is point-in-time integrity,
not a read-only mount.

## Commands and intentional CLI migration

| Script | Current purpose |
|---|---|
| `01_prepare_cell_crops.py` | Standard-library-only inspect/extract/verify; native crops shared by any model |
| `08a_prepare_subcell.py` | Six-batch canonical SubCell preflight |
| `08b_extract_subcell_embeddings.py` | Frozen four-channel HPA encoder/pool export |
| `08c_train_subcell_finetune.py` | Explicit fresh fit, diagnostic smoke or compatible interrupted-run resume |
| `08d_extract_subcell_finetune_embeddings.py` | Explicit selected production-checkpoint export |
| `08e_run_subcell_campaign.py` | Linux/systemd release gates and paired two-GPU seeded fits |

The old `08a_extract_subcell_embeddings.py` is removed. Legacy channel/model-type,
FP16, discovery and historical YAML/gene-checkpoint interfaces are intentionally
unsupported. Frozen mode formerly exposed through08d now belongs to08b. Keep legacy
artifacts and their original producer source; there is no automatic conversion.
`scripts/08b_extract_vit_embeddings.py` (MorphEm) is unchanged from the review base.

Install the **existing** locked `subcell` environment, not a new dependency recipe:

```bash
set -euo pipefail
pixi install --locked -e subcell
# Set these to absolute paths; no writable symlink into the HF mirror.
RELEASE=/absolute/path/to/pinned-HF-git-LFS-checkout
TRAIN=/absolute/path/to/new-training-storage
CONTROL=/absolute/path/to/new-export-control-directory
EXPORT=/absolute/path/to/new-export-version
WEIGHTS=/absolute/path/to/rybg-weights
test ! -e "$CONTROL"
mkdir -p "$CONTROL"
python scripts/01_prepare_cell_crops.py inspect --release "$RELEASE"
python scripts/01_prepare_cell_crops.py extract --release "$RELEASE" --crops "$TRAIN/crops"
pixi run -e subcell python scripts/08a_prepare_subcell.py preflight \
  --release "$RELEASE" --crops "$TRAIN/crops" --output "$TRAIN/cohort"
python scripts/01_prepare_cell_crops.py verify --crops "$TRAIN/crops" > "$CONTROL/crop-verification.json"
```

Stop on any failed command.01 requires Python3.12+ and git, not torch. Its source must
be a git/LFS release, not the remapped output of `00_download_dataset.py`. Reuse a
verified existing cohort instead of re-extracting into it. Extraction never overwrites.

Copy a seed-specific YAML and set only its operational paths/device/worker fields.
For a production two-GPU fit (`devices: 2`), after readiness approval:

```bash
CUDA_VISIBLE_DEVICES=0,1 pixi run -e subcell torchrun --standalone --nproc_per_node=2 \
  scripts/08c_train_subcell_finetune.py -c /absolute/path/to/run.yaml --fit
```

Omitting `--fit` validates bindings only. `--smoke-updates 24` requires a separate
output and yields no selectable checkpoint. Explicit resume names that run's
`models/last.ckpt`; it requires unchanged source/config/data/runtime/rank count and
a compatible pass-boundary checkpoint. Since `d0c61f0`, every completed validation
persists current optimizer/scheduler/RNG/stopping state even if AP ties or decreases;
`last.ckpt` is no longer left at the last best-AP pass. Native best/tie selection is
unchanged. Completion markers and saved horizon/early-stop states reject terminal
production resume, including interruption after the terminal validation save but
before the completion marker. Unsaved progress cannot be recovered from a process
killed before checkpoint publication. Use the archived original implementation for
compatible historical interrupted recovery, not a new PR head. Never resume these six finished runs.

Example selected MAE-s42 export (repeat with ViT's selected checkpoint/output name):

```bash
pixi run -e subcell python scripts/08d_extract_subcell_finetune_embeddings.py \
  --preflight "$TRAIN/cohort" --family mae \
  --checkpoint /absolute/path/to/mae-s42/models/best_model_ap.ckpt \
  --selection /absolute/path/to/mae-s42/selection.json \
  --output "$EXPORT/subcell_allele_rybg_v2_mae_s42" --split all --device cuda:0 \
  --provenance-log "$CONTROL/provenance_log.json" \
  --crop-verification "$CONTROL/crop-verification.json"
```

The CLI default remains `--split test`; **use explicit `--split all`** when downstream
classifiers/retrieval need non-T4 train/context inputs. A T4-only export cannot form
the existing per-batch LOPO folds. Use `--pilot-cells-per-split 16` with a separate
output for a bounded diagnostic before bulk extraction; it changes no weights.

Frozen08b replaces `--checkpoint/--selection` with `--frozen-weights` and
`--weights-sha256`. MAE: `$WEIGHTS/mae_contrast_supcon_model/encoder.pth`, SHA
`493ff6d17e108e89b805f79a17cf09c2f237458637ed847e3ae7d187295f387d`;
ViT: `$WEIGHTS/vit_supcon_model/encoder.pth`, SHA
`6a2d117bcacaa0697034d06d6922580ca7cd996564f270529ec97e6502559845`.
Use names `subcell_frozen_rybg_v2_{mae,vit}` with the same explicit split and provenance flags.

The adapted run used an independent installation of the locked environment in a
clean detached checkout. The later frozen campaign reused that environment without
mutation, with explicit imports from its own clean detached `d0c61f0` source.
Separate worktrees do not isolate resources: the adapted campaign capped each extractor at64GiB RAM/4 CPU-equivalents/128 tasks, one GPU per
model, with15-second supervision and separate logs. The frozen campaign used the
same worker limits and separately retained its spec, source, controller, pilot/repeat
and full-readback logs. Its verifier also compared every metadata column against
the corresponding adapted export. Memory-high reclaim occurred;
there were no OOMs or hard-limit hits. This is tested containment, not optimal
throughput or distributed extraction. Reusing a verified environment is possible
with explicit source imports and no package mutation. Some hosts require
`LD_LIBRARY_PATH=/run/opengl-driver/lib` and an initialized PATH under systemd.

## Runnable regression checks

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  POLARS_MAX_THREADS=2 PYTHONPATH=src:vendor/subcell_embed \
  pixi run -e subcell python -m unittest discover -s tests
```

`tests/subcell_lightning_probe.py` checks native tiny-model LR/data/RNG resume;
`tests/subcell_gather_probe.py` checks distributed gradients;
`tests/subcell_frozen_parity.py` compares real frozen implementations;
`tests/subcell_release_check.py` performs full-size T3/checkpoint release gates.
Read their CLI help and use fresh external outputs. The clean suite now passes
27 tests, including the non-improving-AP recovery regression. Fresh CPU native MAE
and ViT resume probes also pass with zero parameter differences. These are bounded
checks, not fresh full-size GPU training runs. GitHub CI is not claimed: publication
of the prepared workflow requires a credential with `workflow` scope.
Historical diagnostic snapshots remain scoped to their own source hashes; later
documentation does not relabel them.
