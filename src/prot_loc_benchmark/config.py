"""
Configuration and path management for the benchmark pipeline.

Minimum-dataset variant: assumes the QC'd cell-crop bundle has already been
extracted into ``data/``. Server-specific paths, AWS download config, raw-image
paths, and site-QC / crop-extraction parameters live in the upstream branch.
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# REPOSITORY PATHS
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Writable downstream data can live separately from the frozen release and annotations.
DATA_DIR = Path(os.environ.get("MISLOCUS_DATA_ROOT", REPO_ROOT / "data")).expanduser()
if not DATA_DIR.is_absolute():
    raise ValueError("MISLOCUS_DATA_ROOT must be a nonempty absolute path")
DATA_DIR = DATA_DIR.resolve()
# Small reference annotations shipped in-repo (under ~11 MB total). Kept out
# of data/ because data/ is dataset payload that the download script populates;
# annotations/ is curated metadata that needs to live alongside the code.
ANNOTATIONS_DIR = REPO_ROOT / "annotations"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Interim subdirectories (per representation)
CELLPROFILER_DIR = INTERIM_DIR / "cellprofiler"
SUBCELL_DIR = INTERIM_DIR / "subcell"
CYTOSELF_DIR = INTERIM_DIR / "cytoself"
VIT_DIR = INTERIM_DIR / "vit"

# Processed subdirectories
CLASSIFICATION_DIR = PROCESSED_DIR / "classification"
BENCHMARK_DIR = PROCESSED_DIR / "benchmark"

# Crop / manifest inputs (shipped in the dataset bundle)
CROP_MANIFEST_DIR = INTERIM_DIR / "crop_manifest"
SINGLE_CELL_CROPS_DIR = INTERIM_DIR / "single_cell_crops"
# Raw release filenames, NOT a model's input-channel ordering.
CELL_CROP_CHANNEL_FILES = ("agp.npy", "dna.npy", "gfp.npy", "mito.npy")

# ============================================================================
# BATCH CONFIGURATION
# ============================================================================

# Default focus batches (override in scripts via --batch).
FOCUS_BATCHES = [
    "2025_01_27_Batch_13",
    "2025_01_28_Batch_14",
    "2025_03_17_Batch_15",
    "2025_03_17_Batch_16",
]

# All batches available in the dataset bundle.
ALL_PUBLIC_BATCHES = [
    "2024_01_23_Batch_7",
    "2024_02_06_Batch_8",
    "2025_01_27_Batch_13",
    "2025_01_28_Batch_14",
    "2025_03_17_Batch_15",
    "2025_03_17_Batch_16",
]

REPRESENTATIONS = ["cellprofiler", "subcell", "cytoself", "vit"]

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

PREPROCESS_NAN_THRESHOLD = 100  # features with >N NaN rows → drop feature
PREPROCESS_OUTLIER_THRESHOLD = 100.0  # clip ±T, drop features with 99th pct > T
PREPROCESS_VARIANT_ACV_THRESHOLD = 1e-3  # abs_coef_var threshold for feature selection
PREPROCESS_CC_THRESHOLD = 20  # drop wells with fewer cells than this

# ── Control definitions ──────────────────────────────────────────────────
# Complementary positive controls (cPC): gene symbols AND specific alleles
# with known localization changes upon mutation. Cross-batch MisLocus constants.
#
# Matching logic in annotate_controls():
#   - Gene symbols (e.g. "KRAS") → matched by Metadata_symbol (gene-level),
#     so ALL alleles of that gene become cPC.
#   - Specific alleles (e.g. "ALKT1151M") → matched by Metadata_gene_allele
#     (allele-level exact match).
#
# cPC cells are still trained as experimental unknowns for classification.
# Only NC and PC are true controls with special treatment.
CPC_GENE_ALLELES = [
    "ABL1",
    "ADA",
    "ALDH2",
    "ALK",
    "BRD4",
    "CHRM3",
    "CSK",
    "CYP2A6",
    "GHSR",
    "KCNQ2",
    "KRAS",
    "LPAR1",
    "LYN",
    "OPRM1",
    "PAK1",
    "PLP1",
    "PMP22",
    "PRKCE",
    "PTK2B",
    "RB1",
    "TNF",
    "ALK_Thr1151Met",
]

# Per-batch control allele identifiers (TC/NC/PC are allele-level exact matches)
BATCH_CONTROLS = {
    "2024_01_23_Batch_7": {
        "TC": ["EGFP"],
        "NC": ["RHEB", "MAPK9", "PRKACB", "SLIRP"],
        "PC": ["ALK", "ALK_Arg1275Gln"],
    },
    "2024_02_06_Batch_8": {
        "TC": ["EGFP"],
        "NC": ["RHEB", "MAPK9", "PRKACB", "SLIRP"],
        "PC": ["ALK", "ALK_Arg1275Gln"],
    },
    "2025_01_27_Batch_13": {
        "TC": ["eGFP"],
        "NC": ["RHEB", "MAPK9", "PRKACB", "SLIRP"],
        "PC": ["ALK", "ALK_Arg1275Gln", "PTK2B"],
    },
    "2025_01_28_Batch_14": {
        "TC": ["eGFP"],
        "NC": ["RHEB", "MAPK9", "PRKACB", "SLIRP"],
        "PC": ["ALK", "ALK_Arg1275Gln", "PTK2B"],
    },
    "2025_03_17_Batch_15": {
        "TC": ["eGFP"],
        "NC": ["RHEB", "MAPK9", "PRKACB", "SLIRP"],
        "PC": ["ALK", "ALK_Arg1275Gln", "PTK2B"],
    },
    "2025_03_17_Batch_16": {
        "TC": ["eGFP"],
        "NC": ["RHEB", "MAPK9", "PRKACB", "SLIRP"],
        "PC": ["ALK", "ALK_Arg1275Gln", "PTK2B"],
    },
}

# ============================================================================
# CLASSIFICATION CONFIGURATION
# ============================================================================

# Batch plate layout → CV strategy
BATCH_LAYOUT: dict[str, str] = {
    "2024_01_23_Batch_7": "single_rep",
    "2024_02_06_Batch_8": "single_rep",
    "2025_01_27_Batch_13": "single_rep",
    "2025_01_28_Batch_14": "single_rep",
    "2025_03_17_Batch_15": "single_rep",
    "2025_03_17_Batch_16": "single_rep",
}

# Biological replicate batch pairs for benchmark averaging
BIOREP_PAIRS: dict[str, tuple[str, str]] = {
    "pair_78": ("2024_01_23_Batch_7", "2024_02_06_Batch_8"),
    "pair_1314": ("2025_01_27_Batch_13", "2025_01_28_Batch_14"),
    "pair_1516": ("2025_03_17_Batch_15", "2025_03_17_Batch_16"),
}

# Merged allele collection (ClinVar / structure / predictor annotations).
ALLELE_COLLECTION_PATH = ANNOTATIONS_DIR / "full_allele_collection.parquet"

# Channels to benchmark per representation. Add an entry here when registering
# a new representation; see "Adding a new representation" in the README.
BENCHMARK_CHANNELS: dict[str, list[str]] = {
    "cellprofiler": ["DNA", "Mito", "AGP", "GFP", "Morph", "ALL"],
    "cytoself": ["combined"],
    "vit": ["DNA", "Mito", "AGP", "GFP", "Morph", "ALL"],
}

# Benchmark output directories
CLINVAR_BENCHMARK_DIR = BENCHMARK_DIR / "clinvar" / "full_dataset"
CLINVAR_SINGLE_FOLD_DIR = BENCHMARK_DIR / "clinvar" / "single_fold"
HPA_BENCHMARK_DIR = BENCHMARK_DIR / "hpa_consistency"

# XGBoost parameters
XGBOOST_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "n_estimators": 150,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "n_jobs": 4,
}

# Minimum cell count per class in a classifier
MIN_CELL_COUNT = 50

# Maximum training class imbalance ratio for inclusion in aggregation
MAX_IMBALANCE_RATIO = 3.0

# Minimum number of classifiers (folds) per allele for aggregation
MIN_CLASSIFIERS = 2

# Control null distribution percentile for hit threshold
NULL_PERCENTILE = 95

# CellProfiler feature channel patterns (used by classification/channels.py)
CP_CHANNEL_PATTERNS: dict[str, list[str]] = {
    "GFP": ["_GFP"],
    "DNA": ["_DNA"],
    "AGP": ["_AGP"],
    "Mito": ["_Mito"],
}

# Representation → interim feature file name (output of 06_preprocess_profiles).
# Add an entry here when registering a new representation.
REP_FEATURE_FILES: dict[str, str] = {
    "cellprofiler": "features.parquet",
    "cytoself": "features.parquet",
    "subcell": "embeddings.parquet",
    "vit": "features.parquet",
}

# Representation → raw, un-preprocessed feature file name.
# 06_preprocess_profiles reads from this file and writes features.parquet.
REP_RAW_FILES: dict[str, str] = {
    "cytoself": "latent_codes.parquet",
    "subcell": "embeddings.parquet",
    "vit": "embeddings.parquet",
}

# Classification output roots
CLASSIFICATION_OUTPUT_DIR = PROCESSED_DIR / "classification"
CLASSIFICATION_PA_DIR = PROCESSED_DIR / "classification_PA"

# ============================================================================
# SUBCELL REPRESENTATION REGISTRATION
# ============================================================================
# The frozen-SubCell extractor (08a) and fine-tuned SubCell pipeline (08c/08d)
# emit several rep variants, one per channel-config × model-type combination.
# Register them here so REP_FEATURE_FILES / REP_RAW_FILES / BENCHMARK_CHANNELS
# all stay in sync.

_PREPROCESSED_SUBCELL_REPS: set[str] = {
    "subcell_portable_bg_vit",
    "subcell_portable_rbg_mae",
    "subcell_portable_rbg_vit",
    "subcell_portable_rybg_mae",
    "subcell_finetuned_vit",
    "subcell_finetuned_mae",
}


def _register_subcell_rep(rep: str) -> None:
    REP_FEATURE_FILES[rep] = "features.parquet" if rep in _PREPROCESSED_SUBCELL_REPS else "embeddings.parquet"
    REP_RAW_FILES[rep] = "embeddings.parquet"
    BENCHMARK_CHANNELS[rep] = ["EMBED"]


# ============================================================================
# HPA REFERENCE TABLE
# ============================================================================

HPA_GENE_LOCALIZATION_PATH = ANNOTATIONS_DIR / "hpa_gene_localization_table.parquet"


def load_hpa_labels(threshold: float = 1.0) -> dict[str, list[str]]:
    """Gene symbol → list of HPA organelle labels at reliability ≥ threshold.

    HPA scoring: Enhanced=3, Supported=2, Approved=1, Uncertain=0.5. Genes
    with no organelle above ``threshold`` are excluded.
    """
    import polars as pl

    hpa = pl.read_parquet(str(HPA_GENE_LOCALIZATION_PATH))
    loc_cols = [c for c in hpa.columns if c not in ("Gene", "Gene name")]
    labels: dict[str, list[str]] = {}
    for row in hpa.iter_rows(named=True):
        gene = row["Gene name"]
        if gene is None:
            continue
        organelles = [c for c in loc_cols if (row[c] or 0) >= threshold]
        if organelles:
            labels[gene] = organelles
    return labels


# ============================================================================
# CYTOSELF CONFIGURATION
# ============================================================================

CYTOSELF_MANIFEST_DIR = CYTOSELF_DIR / "manifests"
CYTOSELF_MODELS_DIR = CYTOSELF_DIR / "models"
CYTOSELF_EVAL_DIR = CYTOSELF_DIR / "eval"

# Channel configuration matching upstream Cytoself (DataManagerOpenCell):
#   ch0: pro     = GFP (target protein fluorescence)
#   ch1: nuc     = DNA (nucleus fluorescence)
#   ch2: nucdist = EDT of nuclear mask (computed on-the-fly from DNA)
CYTOSELF_CHANNEL_FILES = ["mito", "agp", "dna", "gfp"]
CYTOSELF_GFP_INDEX = 3
CYTOSELF_DNA_INDEX = 2
CYTOSELF_INTENSITY_ADJ = {"pro": 1.0, "nuc": 1.0, "nucdist": 0.01}

# Technical replicate split strategy (T1+T2 → train, T3 → val, T4 → test)
TECH_REP_SPLIT = {1: "train", 2: "train", 3: "val", 4: "test"}

# Export embedding layers
CYTOSELF_EXPORT_LAYERS = ["vqvec2", "vqindhist1", "vqind1"]

# ============================================================================
# SUBCELL CONFIGURATION
# ============================================================================

# Pixel sizes for physical-scale rescaling (µm/pixel)
SUBCELL_PIXEL_SIZE_HPA = 0.0801
SUBCELL_PIXEL_SIZE_MISLOCUS = 0.598
SUBCELL_SCALE_FACTOR = SUBCELL_PIXEL_SIZE_MISLOCUS / SUBCELL_PIXEL_SIZE_HPA  # ~7.47

# Input/output sizes for HPA-scale preprocessing
SUBCELL_INPUT_CROP_SIZE = 128
SUBCELL_INFERENCE_CROP = 448
SUBCELL_RESCALED_SIZE = int(SUBCELL_INPUT_CROP_SIZE * SUBCELL_SCALE_FACTOR)  # 955

# Embedding dimension (2 attention pooling heads × 768 hidden size)
SUBCELL_EMBED_DIM = 1536

# Model weight cache
SUBCELL_WEIGHTS_DIR = INTERIM_DIR / "subcell_portable" / "weights"

# Channel configurations for benchmarking. Order follows SubCellPortable
# convention: R=microtubules, Y=ER, B=nuclei, G=protein. MisLocus mapping:
# AGP→R, Mito→Y, DNA→B, GFP→G.
SUBCELL_CHANNEL_CONFIGS: dict[str, list[str]] = {
    "bg": ["dna", "gfp"],
    "rbg": ["agp", "dna", "gfp"],
    "rybg": ["agp", "mito", "dna", "gfp"],
}

SUBCELL_CHANNEL_FILES = [f"{ch}.npy" for ch in SUBCELL_CHANNEL_CONFIGS["rybg"]]

SUBCELL_MODEL_TYPES: dict[str, str] = {
    "mae": "mae_contrast_supcon_model",
    "vit": "vit_supcon_model",
}

# Register frozen-SubCell variants
for _ch in SUBCELL_CHANNEL_CONFIGS:
    for _mt in SUBCELL_MODEL_TYPES:
        _register_subcell_rep(f"subcell_portable_{_ch}_{_mt}")

# Register fine-tuned SubCell variants (08c/08d)
for _ft in ["mae", "vit"]:
    _register_subcell_rep(f"subcell_finetuned_{_ft}")

# v2 comparisons always use the same downstream preprocessing, never raw vs processed.
for _ft in ["mae", "vit"]:
    for _rep in [f"subcell_frozen_rybg_v2_{_ft}", *[f"subcell_allele_rybg_v2_{_ft}_s{seed}" for seed in (42, 43, 44)]]:
        _PREPROCESSED_SUBCELL_REPS.add(_rep)
        _register_subcell_rep(_rep)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_num_cpus() -> int:
    """Get number of available CPUs."""
    return os.cpu_count() or 1


def ensure_dirs() -> None:
    """Create all required interim/processed directories."""
    for d in [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        CELLPROFILER_DIR,
        SUBCELL_DIR,
        CYTOSELF_DIR,
        VIT_DIR,
        CLASSIFICATION_DIR,
        CLASSIFICATION_OUTPUT_DIR,
        BENCHMARK_DIR,
        CROP_MANIFEST_DIR,
        SINGLE_CELL_CROPS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def get_batch_dir(batch_id: str, data_type: str) -> Path:
    """Get directory for a specific batch and representation type."""
    if data_type == "cellprofiler":
        return CELLPROFILER_DIR / batch_id
    elif data_type == "subcell":
        return SUBCELL_DIR / batch_id
    elif data_type == "cytoself":
        return CYTOSELF_DIR / batch_id
    elif data_type == "vit":
        return VIT_DIR / batch_id
    else:
        raise ValueError(f"Unknown data type: {data_type}")
