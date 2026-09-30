"""Split features by imaging channel for per-channel classification."""

from __future__ import annotations

import logging

from prot_loc_benchmark.config import REP_FEATURE_FILES, canonical_representation

logger = logging.getLogger(__name__)

# Keyword that identifies the protein/GFP channel in feature names
_GFP_KEYWORD = "_GFP"

# Keywords identifying non-protein stain channels
_STAIN_KEYWORDS: dict[str, str] = {
    "DNA": "_DNA",
    "AGP": "_AGP",
    "Mito": "_Mito",
}


def _has_gfp(col: str) -> bool:
    """Return True if a feature column name contains the GFP keyword."""
    return _GFP_KEYWORD.lower() in col.lower()


def get_feature_channels(
    feature_cols: list[str],
    representation: str,
) -> dict[str, list[str]]:
    """Map feature columns to channel groups for classification.

    Returns dict mapping channel name to list of feature column names.
    The channel set depends on the representation:

    - cellprofiler: {GFP, DNA, AGP, Mito, Morph, ALL}. DNA/AGP/Mito
      exclude features also containing "_GFP". Morph = all non-GFP.
    - intensity_only: {GFP, DNA, AGP, Mito, Morph} — one MeanIntensity
      feature per channel; Morph stacks DNA+AGP+Mito.
    - cell_count: {count} — single cell_count feature.
    - cytoself: {global, spectrum, combined} — VQ1 histogram, vqvec2,
      and both concatenated.
    - vit: per-channel groups parsed from "ViT_{channel}_{idx}" column
      names plus a "combined" group; falls back to {EMBED} if a single
      channel is present.
    - Other representations (e.g. subcell): {EMBED} containing all features.
    """
    representation = canonical_representation(representation)
    if representation not in REP_FEATURE_FILES and representation not in ("intensity_only", "cell_count"):
        raise ValueError(f"Unknown representation: {representation}")
    if representation == "intensity_only":
        _intensity_map = {
            "GFP": "Cells_Intensity_MeanIntensity_GFP",
            "DNA": "Cells_Intensity_MeanIntensity_DNA",
            "AGP": "Cells_Intensity_MeanIntensity_AGP",
            "Mito": "Cells_Intensity_MeanIntensity_Mito",
        }
        channels: dict[str, list[str]] = {}
        for ch_name, col_name in _intensity_map.items():
            if col_name in feature_cols:
                channels[ch_name] = [col_name]
        morph_cols = [
            _intensity_map[ch]
            for ch in ("DNA", "AGP", "Mito")
            if _intensity_map[ch] in feature_cols
        ]
        if morph_cols:
            channels["Morph"] = morph_cols
        for name, cols in channels.items():
            logger.info("Channel %s: %d features", name, len(cols))
        return channels

    if representation == "cell_count":
        channels = {"count": list(feature_cols)}
        for name, cols in channels.items():
            logger.info("Channel %s: %d features", name, len(cols))
        return channels

    if representation == "cytoself":
        global_cols = [c for c in feature_cols if c.startswith("Cytoself_global_")]
        spectrum_cols = [c for c in feature_cols if c.startswith("Cytoself_spectrum_")]
        channels: dict[str, list[str]] = {}
        if global_cols:
            channels["global"] = global_cols
        if spectrum_cols:
            channels["spectrum"] = spectrum_cols
        channels["combined"] = list(feature_cols)
        for name, cols in channels.items():
            logger.info("Channel %s: %d features", name, len(cols))
        return channels

    if representation == "vit":
        # Group by channel prefix: ViT_{channel}_{idx}
        # Channel names are normalized to CellProfiler casing so cross-rep
        # benchmarks line up (GFP/DNA/AGP/Mito + Morph/ALL).
        _vit_case = {"gfp": "GFP", "dna": "DNA", "agp": "AGP", "mito": "Mito"}
        channel_groups: dict[str, list[str]] = {}
        for col in feature_cols:
            parts = col.split("_", 2)  # ["ViT", "gfp", "0"]
            if len(parts) >= 3:
                ch = _vit_case.get(parts[1].lower(), parts[1])
                channel_groups.setdefault(ch, []).append(col)
        if len(channel_groups) > 1:
            channels: dict[str, list[str]] = dict(channel_groups)
            morph_cols = [
                c for ch in ("DNA", "AGP", "Mito") for c in channel_groups.get(ch, [])
            ]
            if morph_cols:
                channels["Morph"] = morph_cols
            channels["ALL"] = list(feature_cols)
        else:
            channels = {"EMBED": list(feature_cols)}
        for name, cols in channels.items():
            logger.info("Channel %s: %d features", name, len(cols))
        return channels

    if representation != "cellprofiler":
        return {"EMBED": list(feature_cols)}

    channels: dict[str, list[str]] = {}

    # GFP: all features with GFP keyword
    gfp_cols = [c for c in feature_cols if _has_gfp(c)]
    if gfp_cols:
        channels["GFP"] = gfp_cols

    # DNA, AGP, Mito: channel-specific features EXCLUDING GFP
    for channel_name, keyword in _STAIN_KEYWORDS.items():
        cols = [
            c
            for c in feature_cols
            if keyword.lower() in c.lower() and not _has_gfp(c)
        ]
        if cols:
            channels[channel_name] = cols

    # Morph: ALL non-GFP features (morphology without protein localization)
    morph_cols = [c for c in feature_cols if not _has_gfp(c)]
    if morph_cols:
        channels["Morph"] = morph_cols

    # ALL: every feature column combined
    channels["ALL"] = list(feature_cols)

    for name, cols in channels.items():
        logger.info("Channel %s: %d features", name, len(cols))

    return channels
