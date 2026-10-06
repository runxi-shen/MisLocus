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

### Downloading from another Hugging Face dataset (optional)

The commands above use `anonymous-xyz96/MisLocus`. To download a different
dataset with the same directory layout, use `--hf-repo` for its name and
`--revision` for its version. Replace both example values below; the version
must be a full 40-character lowercase commit ID from that dataset's history,
not a branch or tag:

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

HF repository metadata (`LICENSE`, `README.md`, `MisLocus_croissant.json`,
`.gitattributes`) is removed from the download directory after remapping.

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

**The 32-draw setting is for checking execution, not significance or hit calling.**
A complete batch can require substantial memory and time: `--scope` and
`--channels` do not limit how many feature rows XGBoost loads, and PA retains
its reference pool. `--sample` downloads browseable crops, not scoring features.

### PA: T4 queries, full comparison pool

`09c_classify_PA.py --test-split t4` restricts the **queries entering the
allele-level score**, not the profiles available as comparison partners.
By default, single-cell features are median-aggregated per field of view.
For each T4 variant query, cosine-similarity ranking uses:

- **Positives:** the same variant on different physical plates in the same
  batch, including T1–T3. A different T4 plate can also supply positives if
  it contains the same allele; experimental positives need not share a platemap.
- **Negatives:** matched reference-protein profiles on the query's own physical
  plate, with the same gene. Unrelated proteins are not negative partners.

Per-profile AP is computed with this full pool; only T4 variant AP rows enter
allele-level mAP, its random-ranking null and within-batch/channel BH correction.
T4 is held out from encoder training, but comparison partners can come from
encoder-training or validation plates. This is cross-plate reproducibility,
not retrieval against an entirely encoder-unseen pool.

Do not prefilter inputs to T4: most alleles would lose their different-plate
positives. The same applies to LOO controls, which compare a pseudo-variant
well across plates against other same-allele wells on its platemap.
The producer's `is_hit` is **strict p95 exceedance only**; the reporting PA hit
also requires `below_corrected_p_vs_ref` (BH-adjusted p < 0.05). Missing support
or calibration is not evidence of a non-hit. Reporting-cohort filtering must
not redefine the producer's null or BH family.

Before aggregation, PA excludes queries lacking same-plate/same-gene references
and logs their count. Their profiles remain in the full pool as possible
cross-plate positives for other queries. With no eligible queries, the helper
returns an empty result; a run with no scored channels exits without new score
tables. Any pool used for scoring must have finite profiles with positive finite
norms, and retained queries must have positive and negative partners plus finite
AP/normalized AP. Invalid inputs fail rather than produce degenerate scores.

LOO controls use the same checks. Explicit empty-query runs are skipped; unexpected
control errors now abort instead of silently calibrating on a partial set or
continuing as if controls were merely absent. A genuinely empty control set still
produces null thresholds/hit flags, which the reporting task treats as unavailable.
These safeguards can change eligibility, BH results or control calibration on
unsupported inputs; they are not a claim that historical numerical results are
unchanged. CLI provenance is recorded only after a successful run; a failed rerun
must not rebind previous results to its command. Use fresh output locations:
output-directory reuse is unchanged, and old files left after a failed run must
not be consumed as new results.

### Calibrated hits and shared cohorts

`10c_report_hits.py` reads **already-calibrated batch score tables**. It does
not run models, estimate control thresholds, normalize features, recompute BH,
or verify that calibration used the matching backend/settings. Keep XGB and PA
in separate invocations and fresh output directories.

Required input columns (CSV or Parquet):

| Task | Allele | Score | Matched control p95 | Additional requirement |
|---|---|---|---|---|
| `xgb` | `allele_var` | `auroc_mean` | `null_threshold` | AUROC and threshold in [0, 1] |
| `pa` | `Metadata_gene_allele` | `mAP_vs_ref_norm` | `null_threshold_p95` | `corrected_p_value_vs_ref`, already corrected in the producer's family |

Both require `channel`. Each `--input REPRESENTATION BATCH FILE` supplies the
file's identity; existing `representation`/`batch` columns must agree. Use
experimental-variant summaries, not control classifiers, and consistent scoring
protocols across files. Each file's required columns and supplied producer flags
are validated before concatenation; optional flags may be absent from individual
files. Duplicate setting×allele×batch rows are rejected.

**Calibration is an input contract, not inferred from a column name.** Do not
feed the 0.5/False placeholders from `load_single_fold_metrics`, fallback
thresholds from runs without valid controls, or the 32-draw PA smoke outputs
into scientific hit reporting. The script rejects contradictory producer flags
when present, but cannot establish that an otherwise plausible threshold is
real or matched. The full producer-to-report admission path is separate work.

Example using two independently verified, calibrated batch tables:

```bash
PYTHONPATH=src .pixi/envs/default/bin/python scripts/10c_report_hits.py \
    --task xgb --setting vit GFP \
    --input vit 2024_01_23_Batch_7 scores/batch7-calibrated.csv \
    --input vit 2024_02_06_Batch_8 scores/batch8-calibrated.csv \
    --output-dir data/processed/benchmark/hits-xgb
```

Repeat `--setting REPRESENTATION CHANNEL` and `--input` for a cross-setting
comparison. Settings are explicit, not inferred from whichever files happened
to load, and need not number nine. Channels retain their native names.

- XGB hit: **score > p95**. PA reporting hit: **score > p95 AND corrected p < .05**.
  Equality fails either strict test. Negative finite normalized PA scores remain
  valid. Original producer flags are retained, not overwritten by reporting hits.
- Nonfinite/missing scores or thresholds, and invalid PA corrected p-values,
  are unavailable: `reporting_hit=null`, never a negative. Missing rows in a
  requested setting are recorded explicitly over the union of observed
  allele×batch keys; wholly unobserved alleles cannot be counted by this task.
- Shared support requires an eligible row in **every requested setting** for
  an allele×batch. This filtering does not redefine the producer BH family.
- `any_available`: descriptive summary across all shared available batches;
  hit in any batch, even if its partner is missing. This is distinct from the
  paired lenient/stringent comparison below.
- `complete_pair_any` (**lenient**): either batch hits within a complete pair.
  `complete_pair_all` (**stringent**): both batches hit within a complete
  pair. In either setting, **any successful pair makes the allele a hit**; a
  different pair cannot cancel it. For example, pair outcomes `(hit, hit)` and
  `(non-hit, non-hit)` yield a hit under both settings. `(hit, non-hit)` and
  `(non-hit, hit)` yield a lenient hit but not a stringent hit.
- Both paired summaries retain only complete biological pairs from `BIOREP_PAIRS`;
  incomplete extra pairs do not contribute. Means still use all retained batches,
  not only successful pairs. The command emits both settings as `cohort` rows in
  `allele_summary` and `denominators`; select the intended rows downstream.

Outputs: `batch_audit.parquet` (observed/eligible/shared flags and exclusion
reasons), `shared_mask.parquet`, `allele_summary.parquet`, and `denominators`
(CSV and Parquet, including zero denominators and null empty-cohort fractions).
`report.json` is written last with input-byte hashes and task/settings; it is a
report manifest, **not a historical producer or calibration receipt**. Existing
output directories are refused. A failed run may leave an incomplete directory;
do not use it unless `report.json` exists. This reporting task does not yet
replace the older ClinVar/predictor readers.

### Execution and statistical settings

Both PA and HPA expose `--max-workers` (default **16**) and `--blas-threads`
(default **1**). These bound copairs similarity/null workers and BLAS threads;
they do not subsample profiles or change null-draw counts. For small-memory
verification, pass `--max-workers 1 --blas-threads 1` explicitly. Copairs calls
within a process are serial; run concurrent scoring jobs in separate processes.
Each mAP call uses its own temporary null cache, independent of earlier calls.

ClinVar requires **nine selected representation/channel settings per task**, with
separate nine-test BH corrections for coarse and strict labels—not one18-test
family. Select them with repeated `10_benchmark_clinvar.py --setting REP CHANNEL`
arguments (PA channels include `_vs_ref`). If `--representations` is also supplied,
it must agree. The legacy `benchmark-clinvar`, `benchmark-all` and `all` recipes
require two quoted arguments: the representation list and the nine `--setting`
flags; there is no implicit setting family. Missing/insufficient tests fail rather
than shrink the family. Duplicate annotations use column-wise consensus: a
strict-label conflict does not discard an agreed coarse label. Input cohort,
calibration and model identity still require independent checks.

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
| `10c_report_hits.py` | Calibrated batch scores → shared-support hits, cohort summaries and denominators. |
| `11_summarize_across_reps.py` | Existing `10` summaries → cross-representation tables and plots. |

For matched ClinVar comparisons, reuse the completed `10c` report:
`python scripts/10_benchmark_clinvar.py --cohort-report PATH_TO_REPORT`.
This reads the report's task, nine settings and `complete_pair_any` means without
re-averaging or counting an allele once per pair. Outputs go to a new `clinvar`
(XGB) or `clinvar_PA` (PA) subdirectory; overrides of task/settings and existing
output directories are rejected. Calibration/model lineage remains a precondition.

For the legacy raw-score route, prepare the intended biological-replicate batches (see
[batch identifiers](#batch-identifiers)) and select representations explicitly;
one batch is not a complete paired comparison. `10` defaults to
four-fold XGBoost summaries; `--fold-mode t4-only` selects T4-test metrics.
`--pa` reads PA results instead, and ignores `--fold-mode`. The current PA
reader looks under unsuffixed representation directories, not the `vit_t4`
output above: do not rename those outputs to imply compatible evaluation.
`11` consumes the matching `10` output directory, not features directly.

### HPA reference-localization analysis

HPA matches reference-cell profiles to gene-level labels in
`annotations/hpa_gene_localization_table.parquet`. Use the same CPU environment
and `PYTHONPATH`:

```bash
# Choose a fresh output directory. Small null size is for a usage check only.
.pixi/envs/default/bin/python scripts/10b_benchmark_hpa.py \
    --representations morphem --batches 2024_02_06_Batch_8 \
    --null-size 32 --max-workers 1 \
    --output-dir data/processed/benchmark/hpa-smoke
```

Outputs include `summary/ap_scores_pooled.parquet`, per-channel CSV summaries,
per-representation tables, PNG heatmaps/distributions/PCA plots, and provenance
sidecars. Generated results stay outside source control. As with PA, 32 null
draws are for checking execution, not significance testing.

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
with different scoring settings.

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

Downloaded inputs and generated results live under `data/`, which Git ignores:

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

| Batch | ID                 |
|-------|--------------------|
| 7     | 2024_01_23_Batch_7  |
| 8     | 2024_02_06_Batch_8  |
| 13    | 2025_01_27_Batch_13 |
| 14    | 2025_01_28_Batch_14 |
| 15    | 2025_03_17_Batch_15 |
| 16    | 2025_03_17_Batch_16 |

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

New representations require feature admission, preprocessing and channel
selection to be configured and checked.

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
  metadata requirements and verify the resulting identities and feature schema.

## Reproducibility

- Both bundle and `--sample` downloads default to commit
  `74f63113a76b4a832285da308f3df2932266e456` of `anonymous-xyz96/MisLocus`.
  Use `--revision` with a full commit ID to select another version.
  `PROT_LOC_BENCHMARK_HF_REPO` also selects a dataset; a different dataset
  requires its own explicit `--revision`.
- Record the dataset commit, code revision, environment lock, annotation
  versions and scoring settings. Control calibration must match the scoring
  backend and settings. Use a fresh `data/` directory when changing datasets
  or versions so files from different releases are not mixed.
