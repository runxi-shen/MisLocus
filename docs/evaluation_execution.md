# Evaluation execution

These commands use the existing classification folds, control definitions,
model hyperparameters and hit rules. Published features must already be
downloaded and the repository package/dependencies installed. This guide covers
execution of individual scoring scripts, not orchestration of a full analysis.

## XGBoost allocation and parallelism

CPU is the default. `MISLOCUS_CLASSIFIER_BACKEND=cpu` makes that choice explicit;
`gpu` explicitly requests GPU execution. The legacy `auto` value now selects
CPU. Unknown backend values raise an error.

```bash
MISLOCUS_CLASSIFIER_BACKEND=cpu pixi run -e default python scripts/09_classify.py \
    --batch 2025_03_17_Batch_15 --representation subcell_finetuned_mae \
    --workers 2 --threads 1
```

`--workers` is the number of concurrent classifiers (default 2); `--threads`
is the XGBoost thread count per classifier (default 1). Budget for their
product and for dataframe/library overhead. Both counts must be positive.
The shared model configuration remains unchanged; the command supplies its
thread limit explicitly.

GPU runs require the GPU environment, a CUDA-enabled XGBoost build, and one
allocated GPU identified by its full UUID. Obtain the UUID from the scheduler
or the device assigned to your job; use that value below.

```bash
CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
pixi run -e gpu python scripts/09_classify.py \
    --batch 2025_03_17_Batch_15 --representation subcell_finetuned_mae \
    --gpu --workers 1 --threads 1
```

The placeholder UUID must be replaced. All classifier tasks use the selected
visible GPU rather than distributing folds over other GPUs. Missing allocation,
missing CUDA support, or an actual fitted backend different from the requested
one causes an error. Runtime CUDA libraries must be supplied by the selected
environment. CPU and GPU results are not claimed to be numerically identical.

## Reusing control calibration

A default `--scope all` run fits controls and experimental classifiers, retains
their combined `metrics.csv`, and saves matched NC/PC calibration under
`data/processed/classification/{representation}/{batch}/controls/`.

Alternatively, in a fresh output location, run controls first and then reuse
them with exactly the same input, channels, folds, backend and XGBoost settings:

```bash
pixi run -e default python scripts/09_classify.py \
    --batch 2025_03_17_Batch_15 --representation subcell_finetuned_mae \
    --scope control --workers 2 --threads 1

pixi run -e default python scripts/09_classify.py \
    --batch 2025_03_17_Batch_15 --representation subcell_finetuned_mae \
    --scope exp --workers 2 --threads 1 \
    --calibration-dir data/processed/classification/subcell_finetuned_mae/2025_03_17_Batch_15/controls
```

`--scope control` now writes its predictions, metrics and calibration receipt
inside `controls/`; experimental outputs remain in the parent batch directory.
An experimental-only or allele-subset run requires matching calibration.
Calibration checks bind the feature-file hash, feature channels, batch,
representation, layout/unseen mode, model parameters, configured controls,
runtime versions, XGBoost library bytes, source code and backend. Changed
control files, thresholds or context are rejected before experimental fitting.
Changing worker concurrency alone does not invalidate calibration; changing
XGBoost threads does, because threads are part of the model parameters.

Completed or failed experimental output directories are not overwritten.
Failed attempts remain available for inspection, and provenance is recorded
only after successful execution. Use a separate project/data location for a
new experiment. These output guards are not a general resume mechanism.

## PA and HPA worker limits

`09c_classify_PA.py` and `10b_benchmark_hpa.py` accept `--max-workers 2`
and `--blas-threads 1` (the defaults). The limits cover copairs worker pools,
DuckDB connections used during scoring, and BLAS threads. Each mean-AP call
retains the separate temporary null cache introduced by the evaluation fixes.

The helper temporarily changes process-wide copairs/DuckDB hooks and restores
them on return or error. Run independent scoring jobs in separate processes;
do not overlap these helper calls from different threads in one process.
Other libraries' input-loading pools and total process memory still need an
external job allocation. These flags are not a whole-process resource cap.

Worker limits do not redefine the scientific calculations, but changes in
parallel numerical execution can produce small floating-point differences.
Full-data equality or GPU validation requires separate evidence.
