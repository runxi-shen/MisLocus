# Convenience recipes for downloads and analysis.
# See docs/dataset_bundle.md for direct CPU commands and input requirements.
# These recipes use pixi run (which may install the project package), GPU
# classification, and older representation defaults. `all` preprocesses inputs:
# DO NOT use it on already-cleaned published features. `clean` deletes outputs.
# `download-sample` supplies browseable crops, not a scoring fixture.
# Run `just` (no arguments) to list recipes; inspect each before execution.

set dotenv-load := true

# Comma-separated batch list. Defaults to every public MisLocus batch shipped
# in the dataset bundle. Override on the command line to subset, e.g.:
#   just BATCHES=2025_01_27_Batch_13,2025_01_28_Batch_14 preprocess cellprofiler
BATCHES := "2024_01_23_Batch_7,2024_02_06_Batch_8,2025_01_27_Batch_13,2025_01_28_Batch_14,2025_03_17_Batch_15,2025_03_17_Batch_16"

# Recipe defaults. Select representations present in your input data.
DEFAULT_REPS := "cellprofiler cytoself subcell_portable_bg_vit vit"

# ============================================================================
# Setup
# ============================================================================

# Show available recipes
default:
    @just --list

# Full installation; not the README's dependency-only default setup.
install:
    pixi install

# ============================================================================
# Legacy single-batch and download recipes
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

# GPU XGBoost + PA on a complete batch. Requires downloaded features.
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

# Raw profiles/embeddings only: preprocess DEFAULT_REPS. Never use on published features.
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

# ClinVar + cross-rep summary. SETTINGS must supply nine --setting REP CHANNEL flags.
benchmark-clinvar REPS SETTINGS:
    #!/usr/bin/env bash
    set -euo pipefail
    pixi run python scripts/10_benchmark_clinvar.py --representations {{REPS}} {{SETTINGS}}
    n_reps=$(echo "{{REPS}}" | wc -w)
    if [ "$n_reps" -ge 2 ]; then
        pixi run python scripts/11_summarize_across_reps.py \
            --dataset-dir data/processed/benchmark/clinvar/full_dataset \
            --representations {{REPS}}
    else
        echo "Skipping cross-rep summary (only $n_reps rep)."
    fi

# HPA per-organelle phenotypic consistency benchmark (cross-batch pooled).
benchmark-hpa REPS=DEFAULT_REPS:
    pixi run python scripts/10b_benchmark_hpa.py --representations {{REPS}}

# Legacy benchmark chain; see the dataset guide for input and protocol requirements.
benchmark-all REPS SETTINGS:
    just benchmark-clinvar "{{REPS}}" "{{SETTINGS}}"
    just benchmark-hpa "{{REPS}}"

# Raw-input chain: preprocess-all + GPU classify-all + benchmark-all. Not for cleaned HF features.
all REPS SETTINGS:
    just preprocess-all "{{REPS}}"
    just classify-all "{{REPS}}"
    just benchmark-all "{{REPS}}" "{{SETTINGS}}"

# Wipe per-batch features + classification + benchmark outputs.
clean:
    rm -rf data/interim/cellprofiler/*/features.parquet
    rm -rf data/interim/cellprofiler/*/normalized.parquet
    rm -rf data/interim/cytoself/*/features.parquet
    rm -rf data/interim/subcell_portable_*/*/features.parquet
    rm -rf data/interim/vit/*/features.parquet
    rm -rf data/processed/classification
    rm -rf data/processed/classification_PA
    rm -rf data/processed/benchmark
