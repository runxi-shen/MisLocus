"""Hit/cohort reporting from calibrated batch scores; no null or BH recomputation."""

from __future__ import annotations

import polars as pl

from prot_loc_benchmark.config import BIOREP_PAIRS

SETTING = ["representation", "channel"]
KEY = SETTING + ["batch", "allele"]


def _batch_hits(scores: pl.DataFrame, task: str) -> pl.DataFrame:
    """Validate one input before schema union can invent missing columns/flags."""
    if task not in {"xgb", "pa"}:
        raise ValueError("task must be xgb or pa")
    columns = ("allele_var", "auroc_mean", "null_threshold") if task == "xgb" else (
        "Metadata_gene_allele", "mAP_vs_ref_norm", "null_threshold_p95"
    )
    required = SETTING + ["batch", *columns]
    if task == "pa":
        required.append("corrected_p_value_vs_ref")
    missing = set(required) - set(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if any(scores[c].dtype != pl.String or scores[c].is_null().any()
           or scores[c].str.strip_chars().eq("").any() for c in SETTING + ["batch", columns[0]]):
        raise ValueError("Identity columns must contain nonempty strings")
    rows = scores.with_columns(
        allele=pl.col(columns[0]), score=pl.col(columns[1]).cast(pl.Float64),
        threshold=pl.col(columns[2]).cast(pl.Float64),
    )
    reason = (pl.when(~pl.col("score").is_finite().fill_null(False))
              .then(pl.lit("nonfinite_score"))
              .when(~pl.col("threshold").is_finite().fill_null(False))
              .then(pl.lit("nonfinite_threshold")))
    if task == "pa":
        rows = rows.with_columns(pl.col("corrected_p_value_vs_ref").cast(pl.Float64))
        p = pl.col("corrected_p_value_vs_ref")
        reason = reason.when(~(p.is_finite() & p.is_between(0, 1)).fill_null(False)).then(
            pl.lit("invalid_corrected_p")
        )
    rows = rows.with_columns(
        exclusion_reason=reason.otherwise(pl.lit(None, dtype=pl.String)),
        threshold_pass=pl.col("score") > pl.col("threshold"),
    ).with_columns(eligible=pl.col("exclusion_reason").is_null())
    if task == "xgb":
        if rows.filter(pl.col("eligible") & (
            ~pl.col("score").is_between(0, 1) | ~pl.col("threshold").is_between(0, 1)
        )).height:
            raise ValueError("XGB AUROC scores and thresholds must be in [0, 1]")
    # Catch contradictory producer flags; never silently replace their meaning.
    comparisons = [("is_hit", pl.col("threshold_pass"))]
    if task == "pa":
        comparisons.append(("below_corrected_p_vs_ref", pl.col("corrected_p_value_vs_ref") < .05))
    for name, expected in comparisons:
        valid = rows.filter(pl.col("eligible"))
        if name in rows.columns and not valid.is_empty():
            if valid[name].dtype != pl.Boolean or valid[name].null_count() or valid.filter(
                pl.col(name) != expected
            ).height:
                raise ValueError(f"Producer {name} disagrees with calibrated score values")
    hit = pl.col("threshold_pass")
    if task == "pa":
        hit &= pl.col("corrected_p_value_vs_ref") < .05
    return rows.with_columns(
        reporting_hit=pl.when(pl.col("eligible")).then(hit).otherwise(None),
        observed=pl.lit(True),
    )


def hit_cohorts(
    scores: pl.DataFrame | list[pl.DataFrame],
    task: str,
    settings: list[tuple[str, str]],
    biorep_pairs: dict[str, tuple[str, str]] = BIOREP_PAIRS,
) -> dict[str, pl.DataFrame]:
    """Return batch audit, shared mask, allele summaries and setting denominators.

    Pass distinct input tables as a list: each schema/optional flag is checked
    before concatenation. Inputs use native XGB/PA columns plus representation,
    channel and batch. Thresholds must already match scoring backend/settings;
    this checks values, not calibration provenance or control adequacy.
    Complete-pair 'any'/'all' requires either/both batches within a pair to hit;
    any successful pair makes the allele a hit. Producer fields are preserved.
    """
    selected = pl.DataFrame(settings, schema=SETTING, orient="row")
    if selected.is_empty() or selected.is_duplicated().any():
        raise ValueError("Provide distinct, nonempty representation/channel settings")
    if any(selected[c].dtype != pl.String or selected[c].is_null().any()
           or selected[c].str.strip_chars().eq("").any() for c in SETTING):
        raise ValueError("Identity columns must contain nonempty strings")
    pair_map = {}
    for pair, batches in biorep_pairs.items():
        if len(batches) != 2 or len(set(batches)) != 2 or set(batches) & pair_map.keys():
            raise ValueError("Biological pairs must contain two distinct, non-overlapping batches")
        pair_map.update(dict.fromkeys(batches, pair))
    tables = [scores] if isinstance(scores, pl.DataFrame) else scores
    if not tables:
        raise ValueError("Provide at least one score table")
    rows = pl.concat([_batch_hits(frame, task) for frame in tables], how="diagonal_relaxed").join(
        selected, on=SETTING, how="semi",
    )
    if rows.select(KEY).is_duplicated().any():
        raise ValueError("Duplicate representation/channel/batch/allele score rows")
    if not set(rows["batch"]) <= pair_map.keys():
        raise ValueError("Scores contain batches outside the declared biological pairs")
    # The union of observed allele/batch keys is the audit universe; no invented
    # alleles or assumptions that every allele was assayed in all six batches.
    grid = selected.join(rows.select("batch", "allele").unique(), how="cross")
    audit = grid.join(rows, on=KEY, how="left").with_columns(
        observed=pl.col("observed").fill_null(False),
        eligible=pl.col("eligible").fill_null(False),
        exclusion_reason=pl.when(pl.col("observed").is_null())
        .then(pl.lit("missing_setting_row")).otherwise(pl.col("exclusion_reason")),
    ).with_columns(
        shared=pl.col("eligible").all().over("batch", "allele"),
        pair_name=pl.col("batch").replace_strict(pair_map, return_dtype=pl.String),
    )
    mask = audit.filter(pl.col("shared")).select("batch", "allele").unique().sort("batch", "allele")
    shared = audit.filter(pl.col("shared"))
    complete = shared.filter(
        pl.col("batch").n_unique().over(SETTING + ["allele", "pair_name"]) == 2
    )
    parts = []
    cohorts = ["any_available", "complete_pair_any", "complete_pair_all"]
    for cohort, data in zip(cohorts, [shared, complete, complete]):
        aggregate = pl.col("reporting_hit").all() if cohort.endswith("_all") else pl.col("reporting_hit").any()
        data = data.with_columns(pair_hit=aggregate.over(SETTING + ["allele", "pair_name"]))
        parts.append(data.group_by(SETTING + ["allele"]).agg(
            mean_score=pl.col("score").mean(), hit=pl.col("pair_hit").any(),
            n_batches=pl.col("batch").n_unique(),
        ).with_columns(cohort=pl.lit(cohort)))
    alleles = pl.concat(parts).sort("cohort", *SETTING, "allele")
    counts = alleles.group_by(["cohort", *SETTING]).agg(
        n_alleles=pl.len(), n_hits=pl.col("hit").sum(),
    )
    denominators = selected.join(pl.DataFrame({"cohort": cohorts}), how="cross").join(
        counts, on=["cohort", *SETTING], how="left",
    ).with_columns(pl.col("n_alleles", "n_hits").fill_null(0)).with_columns(
        hit_fraction=pl.when(pl.col("n_alleles") > 0)
        .then(pl.col("n_hits") / pl.col("n_alleles")).otherwise(None),
    ).sort("cohort", *SETTING)
    return {"batch_audit": audit.sort(KEY), "shared_mask": mask,
            "allele_summary": alleles, "denominators": denominators}
