# Justfile for prot-loc-benchmark (minimal dataset companion)
#
# Quick tour:
#   just download-sample     # ~1.2 GB browseable sample (8 alleles × 2 batches)
#   just inspect-sample      # show what's in data/sample/
#   just download-batch      # ~3 GB CellProfiler features for one batch
#   just classify-batch      # XGBoost + PA mAP on that one batch
#   just download-all        # full mirror of every rep × every batch
#
# Run `just` (no arguments) to see every recipe with its short description.

set dotenv-load := true

# Comma-separated batch list. Defaults to every public MisLocus batch shipped
# in the dataset bundle. Override on the command line to subset, e.g.:
#   BATCHES=2025_01_27_Batch_13,2025_01_28_Batch_14 just preprocess cellprofiler
BATCHES := "2024_01_23_Batch_7,2024_02_06_Batch_8,2025_01_27_Batch_13,2025_01_28_Batch_14,2025_03_17_Batch_15,2025_03_17_Batch_16"

# Reps that flow through the cross-rep summary by default. Override per-call.
DEFAULT_REPS := "cellprofiler cytoself subcell_portable_bg_vit vit"

# ============================================================================
# Setup
# ============================================================================

# Show available recipes
default:
    @just --list

# Install all pixi environments
install:
    pixi install

# ============================================================================
# Showcase — the five commands you'll use 90% of the time
# ============================================================================

# Download the small browseable sample (~1.2 GB) into data/sample/{batch}/{allele}/.
download-sample:
    pixi run python scripts/00_download_dataset.py --sample

# Print the structure of the downloaded sample (alleles per batch, manifest schema, sizes).
inspect-sample:
    pixi run python scripts/00c_inspect_sample.py

# Download one rep's features for one batch (default: cellprofiler + Batch_13, ~3 GB).
download-batch BATCH="2025_01_27_Batch_13" REP="cellprofiler":
    pixi run python scripts/00_download_dataset.py --rep {{REP}} --batch {{BATCH}}

# Run XGBoost + phenotypic-activity mAP on one batch (default: cellprofiler + Batch_13). Requires `download-batch` first.
classify-batch BATCH="2025_01_27_Batch_13" REP="cellprofiler":
    # CONDA_OVERRIDE_CUDA: lets pixi resolve the gpu env's `__cuda`
    # virtual package on hosts where it isn't auto-detected.
    # CUDA_VERSION: tells xgboost which CUDA runtime to bind at load time;
    # without it xgboost emits "Device is changed from GPU to CPU".
    CONDA_OVERRIDE_CUDA=12.0 CUDA_VERSION=12.0 pixi run -e gpu python scripts/09_classify.py \
        --batch {{BATCH}} --representation {{REP}} --gpu
    pixi run python scripts/09c_classify_PA.py \
        --batch {{BATCH}} --representation {{REP}}

# Download the full bundle: features for every rep × every batch + crop tarballs (hundreds of GB).
download-all *EXTRA:
    pixi run python scripts/00_download_dataset.py {{EXTRA}}

# ============================================================================
# Aux downloads
# ============================================================================

# Download SubCell pretrained encoder weights from CZI's public S3 bucket (~2 GB; only needed for 08a / 08c).
download-subcell *EXTRA:
    pixi run python scripts/00b_download_subcell_weights.py {{EXTRA}}

# ============================================================================
# Advanced — multi-batch / multi-rep loops
# ============================================================================

# Preprocess one rep (e.g. cellprofiler, cytoself, subcell_portable_bg_vit, vit) across all BATCHES.
preprocess REP:
    #!/usr/bin/env bash
    set -euo pipefail
    for batch in $(echo {{BATCHES}} | tr ',' ' '); do
        echo "=== preprocess {{REP}} $batch ==="
        pixi run python scripts/06_preprocess_profiles.py --representation {{REP}} --batch $batch
    done

# Preprocess all DEFAULT_REPS in sequence.
preprocess-all REPS=DEFAULT_REPS:
    #!/usr/bin/env bash
    set -euo pipefail
    for rep in {{REPS}}; do
        just preprocess "$rep"
    done

# XGBoost classification for one rep across all BATCHES (GPU env).
classify REP EXTRA_ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    # CONDA_OVERRIDE_CUDA lets pixi activate the gpu env on hosts where
    # the __cuda virtual package isn't auto-detected. CUDA_VERSION pins
    # xgboost's runtime CUDA load so it stays on GPU.
    export CONDA_OVERRIDE_CUDA=12.0
    export CUDA_VERSION=12.0
    for batch in $(echo {{BATCHES}} | tr ',' ' '); do
        echo "=== classify {{REP}} $batch ==="
        pixi run -e gpu python scripts/09_classify.py \
            --batch $batch --representation {{REP}} --gpu {{EXTRA_ARGS}}
    done

# Phenotypic activity (copairs mAP-vs-ref) for one rep across all BATCHES.
classify-pa REP EXTRA_ARGS="":
    #!/usr/bin/env bash
    set -euo pipefail
    for batch in $(echo {{BATCHES}} | tr ',' ' '); do
        echo "=== classify-PA {{REP}} $batch ==="
        pixi run python scripts/09c_classify_PA.py \
            --batch $batch --representation {{REP}} {{EXTRA_ARGS}}
    done

# Run both classifiers for all DEFAULT_REPS.
classify-all REPS=DEFAULT_REPS:
    #!/usr/bin/env bash
    set -euo pipefail
    for rep in {{REPS}}; do
        just classify "$rep"
        just classify-pa "$rep"
    done

# ClinVar Pathogenic-vs-Benign Wilcoxon + cross-rep summary.
benchmark-clinvar REPS=DEFAULT_REPS:
    #!/usr/bin/env bash
    set -euo pipefail
    dataset_dir=$(pixi run python -c 'from prot_loc_benchmark.config import CLINVAR_BENCHMARK_DIR; print(CLINVAR_BENCHMARK_DIR)')
    pixi run python scripts/10_benchmark_clinvar.py --representations {{REPS}}
    n_reps=$(echo "{{REPS}}" | wc -w)
    if [ "$n_reps" -ge 2 ]; then
        pixi run python scripts/11_summarize_across_reps.py \
            --dataset-dir "$dataset_dir" \
            --representations {{REPS}}
    else
        echo "Skipping cross-rep summary (only $n_reps rep)."
    fi

# HPA per-organelle phenotypic consistency benchmark (cross-batch pooled).
benchmark-hpa REPS=DEFAULT_REPS:
    pixi run python scripts/10b_benchmark_hpa.py --representations {{REPS}}

# Run all benchmarks (ClinVar + HPA + cross-rep summary).
benchmark-all REPS=DEFAULT_REPS:
    just benchmark-clinvar "{{REPS}}"
    just benchmark-hpa "{{REPS}}"

# preprocess-all + classify-all + benchmark-all.
all REPS=DEFAULT_REPS:
    just preprocess-all "{{REPS}}"
    just classify-all "{{REPS}}"
    just benchmark-all "{{REPS}}"

# Wipe outputs only under the configured data root (MISLOCUS_DATA_ROOT or repo/data).
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    data_dir=$(pixi run python -c 'from prot_loc_benchmark.config import DATA_DIR; print(DATA_DIR)')
    rm -rf -- "$data_dir"/interim/cellprofiler/*/features.parquet
    rm -rf -- "$data_dir"/interim/cellprofiler/*/normalized.parquet
    rm -rf -- "$data_dir"/interim/cytoself/*/features.parquet
    rm -rf -- "$data_dir"/interim/subcell_portable_*/*/features.parquet
    rm -rf -- "$data_dir"/interim/vit/*/features.parquet
    rm -rf -- "$data_dir"/processed/classification
    rm -rf -- "$data_dir"/processed/classification_PA
    rm -rf -- "$data_dir"/processed/benchmark
