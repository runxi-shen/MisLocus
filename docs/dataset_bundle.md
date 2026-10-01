# Dataset bundle

The published `features.parquet` files are already cleaned and ready for
scoring. **Skip `scripts/06_preprocess_profiles.py`: do not normalize them
again.** Crops and encoder weights are optional. The bundle is published
alongside the parent paper / preprint on Hugging Face Hub at
[`anonymous-xyz96/MisLocus`](https://huggingface.co/datasets/anonymous-xyz96/MisLocus).

## Downloading

Two scripts, run independently:

| Script                                      | Source                                                             | When you need it                                       |
|---------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------|
| `scripts/00_download_dataset.py`            | HF dataset repo `anonymous-xyz96/MisLocus`                         | Always (unless you already have features locally).     |
| `scripts/00b_download_subcell_weights.py`   | CZI public S3 (`czi-subcell-public.s3.amazonaws.com/models/`)      | Only if you'll run `08a` / `08c` (SubCell extract / fine-tune). |

Use the dependency-only setup in the [README](../README.md#download-published-features), from
the repository root. The commands below use that environment directly.

### Dataset bundle — all features at the pinned revision

```bash
# Every representation and batch, plus manifests/metadata; no crop archives.
.pixi/envs/default/bin/python scripts/00_download_dataset.py --no-include-crops

# Optional full bundle, including large crop archives and browseable sample.
.pixi/envs/default/bin/python scripts/00_download_dataset.py
```

Identical feature re-imports are accepted; conflicting destination bytes stop
feature remapping before any feature moves. `--force` re-fetches but does not
override this guard. Because downloaded files are moved, repeated commands may
transfer them again. This is input materialization, not an exact sync: stale
files are not pruned and revision changes are not transactional. Use a fresh
`data/` directory (preserve the previous one) when changing repo or revision.

### Dataset bundle — one representation or batch

Restrict by representation and/or batch for partial materialization. This
still downloads complete feature files, not a small row sample. Crop shards
are excluded by default in subset mode:

```bash
# One real-data scoring input; this does not run downstream comparisons.
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --rep morphem --batch 2024_02_06_Batch_8

# Multiple reps / batches (comma-separated or repeat the flag)
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --rep cytoself,cellprofiler --batch 2025_01_27_Batch_13

# Optional crop archives (large)
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --batch 2025_01_27_Batch_13 --include-crops
```

### Dataset bundle — override the HF repo

Alternate repositories, including `PROT_LOC_BENCHMARK_HF_REPO` overrides, require
an explicit full commit SHA belonging to that repository. Branches and tags
are rejected; `repo@sha` is not supported syntax. Replace the repository and
quoted SHA placeholder before running:

```bash
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --hf-repo myorg/my-dataset --revision "<40-character-commit-sha>" --no-include-crops
```

### SubCell pretrained weights

```bash
# Six encoder.pth checkpoints (~2 GB) → data/interim/subcell_portable/weights/
.pixi/envs/default/bin/python scripts/00b_download_subcell_weights.py
```

Skip this step if you're only consuming the shipped per-rep
`features.parquet` files. You only need the encoder weights if you'll
run `scripts/08a_extract_subcell_embeddings.py` (frozen ViT extraction)
or `scripts/08c_train_subcell_finetune.py` (fine-tuning).

### What the dataset-download script does

After `snapshot_download`, paths are remapped:
- `representations/{rep}/{batch}/features.parquet` → `data/interim/{rep}/{batch}/features.parquet`, except public `morphem` → local `vit` (feature bytes unchanged)
- `manifest/manifest_Batch_X.parquet` → `data/interim/crop_manifest/{full_batch}/manifest.parquet`
- `single_cell_crops/{batch}/shard-NN.tar.gz` → extracted into `data/interim/single_cell_crops/{batch}/`

HF-only repo metadata (`LICENSE`, `README.md`, `MisLocus_croissant.json`,
`.gitattributes`) is removed at the end of the remap; the
`.gitattributes` in particular would otherwise silently activate LFS
smudge for every parquet in the working tree.

Anything not under `data/` (notably the curated reference parquets in
`annotations/`) is in-repo and does not come from either downloader.

## Run CPU scoring on published features

After downloading features, run from the repository root in the dependency-only
`default` environment. Unlike the downloader, the scoring scripts require `src`
on the Python import path:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MISLOCUS_CLASSIFIER_BACKEND=cpu

# One imaging channel; keep controls and variants in the same invocation.
.pixi/envs/default/bin/python scripts/09_classify.py \
    --batch 2024_02_06_Batch_8 --representation morphem \
    --scope all --channels GFP
```

Outputs go to `data/processed/classification/vit/2024_02_06_Batch_8/`:
`predictions.parquet`, `metrics.csv`, `classifier_info.csv`,
`metrics_summary.csv`, and additional feature-importance/summary files.
The summary uses controls from this invocation. Running `--scope control`
followed by an allele-only run does **not** reuse the earlier calibration;
it writes to the same directory. Use a fresh checkout/data directory for a
separate run rather than overwriting results you need to retain.

For a PA command with inexpensive permutation settings:

```bash
.pixi/envs/default/bin/python scripts/09c_classify_PA.py \
    --batch 2024_02_06_Batch_8 --representation morphem \
    --scope AGXT_Ala186Val --test-split t4 \
    --null-size 32 --ctrl-null-size 32 --max-workers 1
```

This writes `mAP_results.parquet` and `mAP_control.parquet` under
`data/processed/classification_PA/vit_t4/2024_02_06_Batch_8/`.
`--test-split t4` restricts queries while retaining the comparison pool.
Control-null computation stays enabled. The producer's `is_hit` uses p95;
it is not the reporting rule that additionally requires within-batch BH.

**These are usage checks, not production-analysis settings.** The commands
were exercised on an 840-row unchanged-feature subset, not a complete download.
A complete batch can require substantial memory and time: `--scope` and
`--channels` do not limit how many feature rows XGBoost loads, and PA retains
its reference pool. The 32-draw PA setting is only for checking execution,
not reliable significance or hit calling. Changing null sizes alone does not
establish a validated production workflow. `--sample` downloads browseable
crops, not a ready-to-run scoring fixture.

### Script map and downstream prerequisites

| Script | Reads → writes / purpose |
|--------|--------------------------|
| `00_download_dataset.py` | Pinned HF bundle → local features, manifests and optional crops. |
| `00b_download_subcell_weights.py` | Optional pretrained SubCell encoder weights. |
| `00c_inspect_sample.py` | Inspect `data/sample/`; no scoring. |
| `01_prepare_cell_crops.py` | Inspect, unpack and verify already-cropped arrays; not segmentation. |
| `06_preprocess_profiles.py` | Raw profiles/embeddings → cleaned features; **skip for published features**. |
| `07a` / `07b` / `07c` / `07d` | Cytoself manifest preparation / training / embedding-format conversion / trained-model inference. |
| `08a` / `08b` | Frozen SubCell / MorphEm embedding extraction. |
| `08c` / `08d` | SubCell fine-tuning / fine-tuned embedding extraction. |
| `09_classify.py` | Cleaned features → XGBoost predictions, metrics and control-calibrated summaries. |
| `09c_classify_PA.py` | Cleaned features → PA scores and optional control-null outputs. |
| `10_benchmark_clinvar.py` | XGBoost or PA scores + annotations → per-representation ClinVar summaries. |
| `10b_benchmark_hpa.py` | Reference-cell features + HPA labels → localization retrieval; independent of variant scores. |
| `11_summarize_across_reps.py` | Existing `10` summaries → cross-representation tables and plots. |

For ClinVar, prepare the intended biological-replicate batches (see
[batch identifiers](#batch-identifiers)) and select representations explicitly;
a one-batch smoke is not a complete paired comparison. `10` defaults to
four-fold XGBoost summaries; `--fold-mode t4-only` selects T4-test metrics.
`--pa` reads PA results instead, and ignores `--fold-mode`. The current PA
reader looks under unsuffixed representation directories, not the `vit_t4`
output above: do not rename those outputs to imply compatible evaluation.
`11` consumes the matching `10` output directory, not features directly.
These downstream comparisons were not validated by the scoring smoke.

### HPA reference-localization analysis

HPA uses gene-level labels from the bundled
`annotations/hpa_gene_localization_table.parquet`, not Lacoste variant labels
or classification scores. Using the same CPU environment and `PYTHONPATH`:

```bash
# Choose a fresh output directory. Small null size is for a usage check only.
.pixi/envs/default/bin/python scripts/10b_benchmark_hpa.py \
    --representations morphem --batches 2024_02_06_Batch_8 \
    --null-size 32 --max-workers 1 \
    --output-dir data/processed/benchmark/hpa-smoke
```

Outputs include `summary/ap_scores_pooled.parquet`, per-channel CSV summaries,
per-representation tables, PNG heatmaps/distributions/PCA plots, and provenance
sidecars. The CLI was checked to completion on the same 840-row subset with
six reference genes. **LaTeX export is not required or produced**; CSV/Parquet
are the numerical results, and the plots are analysis aids, not a manuscript
regeneration workflow. Generated results stay outside source control. As with
PA, these small-null smoke settings do not establish production significance.

Inspect current arguments without running an analysis:

```bash
.pixi/envs/default/bin/python scripts/10_benchmark_clinvar.py --help
.pixi/envs/default/bin/python scripts/10b_benchmark_hpa.py --help
.pixi/envs/default/bin/python scripts/11_summarize_across_reps.py --help
```

### GPU and optional model tools

The CPU commands explicitly set `MISLOCUS_CLASSIFIER_BACKEND=cpu`; omitting
`--gpu` alone does not force CPU because the default backend is automatic.
GPU XGBoost uses the `gpu` environment (CUDA 12 requirement) and `09 --gpu`.
Device selection can fall back to CPU: check the actual backend in the log.
Never substitute CPU calibration for GPU scores, or reuse controls computed
with different scoring settings. GPU installation/execution and parity were
not checked in this usage pass.

Encoder work uses the separate `cytoself`, `subcell` or `vit` environments,
currently declaring CUDA 12.4. Their model-input preparation is not feature
normalization. Use each script's input/label requirements; Cytoself and SubCell
do not share one universal training protocol. No encoder installation,
training or extraction is needed for the published-feature route.

The `Justfile` retains legacy recipes: `pixi run` can install the project
package, classification recipes select GPU, and `all` includes preprocessing.
Use the direct commands above for dependency-only published-feature usage;
`just all` is **not** its quickstart. Raw-feature preprocessing and destructive
`clean` recipes are separate operations, not preparation for downloaded features.

## Layout after download

The seven published names are `cellprofiler`, `cytoself`, `morphem`,
`subcell_finetuned_mae`, `subcell_finetuned_vit`, `subcell_portable_rbg_mae`, and
`subcell_portable_rbg_vit`. `--rep vit` and `--rep morphem` both select HF
`morphem` and admit it to `interim/vit/`; outputs retain the internal `vit` label.
MorphEm uses `ViT_{gfp,dna,agp,mito}_0..383`: GFP=384, Morph=1152, ALL=1536
features. Cleaned feature counts for other representations can vary by batch.
Representation names alone do not establish fine-tuned model/export lineage.

The download script populates `data/` (everything under `data/` is
gitignored — pure dataset payload):

```
data/
├── interim/
│   ├── single_cell_crops/             # only present if --include-crops
│   │   └── {batch}/{allele}/
│   │       ├── dna.npy        # uint16, (N, 128, 128)
│   │       ├── gfp.npy
│   │       ├── agp.npy
│   │       └── mito.npy
│   ├── crop_manifest/
│   │   └── {batch}/manifest.parquet
│   ├── cellprofiler/
│   │   └── {batch}/features.parquet
│   ├── cytoself/
│   │   └── {batch}/features.parquet
│   ├── vit/                          # public HF name: morphem
│   │   └── {batch}/features.parquet
│   ├── subcell_portable_rbg_vit/
│   │   └── {batch}/features.parquet
│   ├── subcell_portable_rbg_mae/
│   ├── subcell_finetuned_vit/
│   └── subcell_finetuned_mae/
└── processed/                         # empty until 09 / 10 / 10b / 11 run
```

In-repo (NOT downloaded — tracked in git, ~9.7 MB total):

```
annotations/
├── full_allele_collection.parquet         # ClinVar / dbNSFP / pLDDT / RSA / G2P / OMIM
└── hpa_gene_localization_table.parquet    # 262 genes × 50 organelles
```

These are read via `prot_loc_benchmark.config.{ALLELE_COLLECTION_PATH,
HPA_GENE_LOCALIZATION_PATH}`.

## Batch identifiers

The bundle is keyed by MisLocus batch IDs:

| Batch | ID                  | Notes      |
|-------|---------------------|------------|
| 7     | 2024_01_23_Batch_7  | A1R1 only  |
| 8     | 2024_02_06_Batch_8  | A1R2 only  |
| 13    | 2025_01_27_Batch_13 |            |
| 14    | 2025_01_28_Batch_14 |            |
| 15    | 2025_03_17_Batch_15 |            |
| 16    | 2025_03_17_Batch_16 |            |

Biological-replicate pairs used for benchmark averaging:
B7+B8, B13+B14, B15+B16 (`config.BIOREP_PAIRS`).

## Manifest schema

`data/interim/crop_manifest/{batch}/manifest.parquet` is the canonical
cell ordering. Each row is one cell; the corresponding crop sits at index
`i` in every `*.npy` channel file under
`data/interim/single_cell_crops/{batch}/{allele}/`.

Required `Metadata_*` columns (also produced by every embeddings extractor —
see `scripts/08b_extract_vit_embeddings.py:52`):

```
Metadata_Plate              str
Metadata_well_position      str
Metadata_Site               int
Metadata_ImageNumber        int
Metadata_ObjectNumber       int
Metadata_gene_allele        str
Metadata_symbol             str
Metadata_node_type          str   # "disease_wt" | "allele" | "TC"
Metadata_Control            str   # "Exp" | "cPC" | "NC" | "PC" | "TC"
Metadata_plate_map_name     str
```

## Bringing new embeddings

This is an optional developer route, not automatic plug-in support. Published
representations are already configured; new names need explicit admission,
preprocessing and channel-selection checks.

- Raw embeddings contain cell identities/metadata and numeric feature columns.
  Check `REP_RAW_FILES` in `src/prot_loc_benchmark/config.py` for the expected
  filename; it is not universally `embeddings.parquet`.
- `06_preprocess_profiles.py` separates CellProfiler feature selection from
  learned-embedding preprocessing. Verify identities, feature order and QC
  assumptions before running it once, in a fresh output location.
- Already-cleaned `features.parquet` skips `06`; a filename alone does not
  prove that normalization or cell/feature QC has been performed correctly.
- New representation names require reviewing `REP_FEATURE_FILES`,
  `REP_RAW_FILES`, `BENCHMARK_CHANNELS`, `get_feature_channels`, and downstream
  CLI selections. Channel-name substrings alone are not a registration API.
- If starting from crops, use the appropriate `07`/`08` model-input and
  metadata requirements. Training/export interfaces and new-representation
  integration need their own checks; they were not exercised by the CPU
  scoring example.

## Reproducibility

- Both bundle and `--sample` downloads default to commit
  `74f63113a76b4a832285da308f3df2932266e456` of `anonymous-xyz96/MisLocus`.
  `--revision <40-character-commit-sha>` overrides it; the downloader logs and
  passes the SHA to `snapshot_download(revision=...)`. No fallback to main.
- Validation inspected 42 Parquet schemas and hash-verified seven Batch 8
  payloads against this revision. Small unchanged real-row subsets passed
  CPU XGBoost, PA and HPA API checks. This is not a fresh-network full-pipeline
  test, production calibration, GPU equivalence, or paper-result reproduction.
- A dataset pin fixes the source selection, not the entire experiment. Retain
  code/environment/annotation versions and backend/settings-matched controls.
  It does not verify historical producer lineage or prevent mixing old local
  files if a nonempty data directory is reused across revisions.
