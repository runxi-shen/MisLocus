"""Parallel classifier execution loop extracted from scripts/09_classify.py.

Encapsulates the task → thread-pool → ClassificationWriter pipeline so
multiple classification scripts can share one implementation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .io import ClassificationWriter
from .metrics import compute_classifier_metrics
from .train import train_and_predict

logger = logging.getLogger(__name__)


def _run_classifier(task: dict, task_device: str, xgb_params: dict | None = None) -> dict | None:
    """Train one classifier and return a results dict, or None on failure."""
    result = train_and_predict(
        task["train_df"],
        task["test_df"],
        task["ch_features"],
        device=task_device,
        xgb_params=xgb_params,
    )
    if result is None:
        return None

    preds, labels, importances = result
    pair = task["pair"]
    channel = task["channel"]
    fold = task["fold"]
    train_df = task["train_df"]
    test_df = task["test_df"]

    classifier_id = f"{pair.pair_id}__{channel}__fold{fold.fold_id}"
    m = compute_classifier_metrics(preds, labels)
    n_train_pos = int((train_df["Label"] == 1).sum())
    n_train_neg = int((train_df["Label"] == 0).sum())
    imbalance = (
        max(n_train_pos, n_train_neg) / max(min(n_train_pos, n_train_neg), 1)
    )

    return {
        "classifier_id": classifier_id,
        "pair": pair,
        "channel": channel,
        "fold": fold,
        "preds": preds,
        "labels": labels,
        "importances": importances,
        "metrics": m,
        "n_train_pos": n_train_pos,
        "n_train_neg": n_train_neg,
        "imbalance": imbalance,
        "test_df": test_df,
        "train_height": train_df.height,
    }


def run_classifier_tasks(
    tasks: list[dict],
    device: str,
    output_dir: Path,
    workers: int = 2,
    xgb_params: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict], int]:
    """Run a batch of classifier tasks in parallel and stream predictions.

    Parameters
    ----------
    tasks
        List of dicts, each with keys:
        ``pair``, ``channel``, ``ch_features``, ``fold``, ``train_df``, ``test_df``.
        The train/test dataframes are read-only — this function does not
        modify them or their task dictionaries.
    device
        One selected CPU or explicitly allocated GPU backend, shared by all tasks.
    output_dir
        Directory where ``predictions.parquet`` will be written.

    Returns
    -------
    (metrics_rows, importance_rows, info_rows, n_classifiers_run)
        Three lists of dicts ready to feed into ``pl.DataFrame`` for CSV output,
        plus a count of classifiers that produced results (excludes None returns).
    """
    metrics_rows: list[dict] = []
    importance_rows: list[dict] = []
    info_rows: list[dict] = []
    n_classifiers = 0

    if workers < 1:
        raise ValueError("workers must be positive")

    t_class = time.time()
    predictions_path = output_dir / "predictions.parquet"
    with ClassificationWriter(predictions_path) as writer:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_classifier, t, device, xgb_params): t
                for t in tasks
            }
            for future in as_completed(futures):
                r = future.result()
                if r is None:
                    raise ValueError("Classifier unavailable; refusing a partial calibrated run")

                n_classifiers += 1
                pair = r["pair"]
                channel = r["channel"]
                fold = r["fold"]
                test_df = r["test_df"]

                writer.write_predictions(
                    classifier_id=r["classifier_id"],
                    plates=test_df["Metadata_Plate"].to_numpy(),
                    wells=test_df["Metadata_well_position"].to_numpy(),
                    object_numbers=test_df["Metadata_ObjectNumber"].to_numpy(),
                    labels=r["labels"],
                    predictions=r["preds"],
                    channel=channel,
                    is_control=pair.is_control,
                )

                metrics_rows.append({
                    "classifier_id": r["classifier_id"],
                    "pair_id": pair.pair_id,
                    "gene": pair.gene,
                    "allele_ref": pair.allele_ref,
                    "allele_var": pair.allele_var,
                    "channel": channel,
                    "fold_id": fold.fold_id,
                    "is_control": pair.is_control,
                    "category": pair.category,
                    "n_train": r["train_height"],
                    "n_test": test_df.height,
                    "imbalance_ratio": r["imbalance"],
                    **r["metrics"],
                })

                top_feats = sorted(
                    r["importances"].items(), key=lambda x: x[1], reverse=True
                )[:50]
                for feat_name, feat_imp in top_feats:
                    importance_rows.append({
                        "classifier_id": r["classifier_id"],
                        "feature": feat_name,
                        "importance": feat_imp,
                    })

                info_rows.append({
                    "classifier_id": r["classifier_id"],
                    "pair_id": pair.pair_id,
                    "gene": pair.gene,
                    "allele_ref": pair.allele_ref,
                    "allele_var": pair.allele_var,
                    "channel": channel,
                    "fold_id": fold.fold_id,
                    "is_control": pair.is_control,
                    "category": pair.category,
                    "n_train_ref": r["n_train_pos"],
                    "n_train_var": r["n_train_neg"],
                    "n_test": test_df.height,
                    "train_plates": ",".join(fold.train_plates),
                    "test_plates": ",".join(fold.test_plates),
                    "test_wells": ",".join(fold.test_wells),
                })

    t_class_elapsed = time.time() - t_class
    logger.info(
        "Classifier phase: %d classifiers in %.1fs (%.2fs/classifier)",
        n_classifiers,
        t_class_elapsed,
        t_class_elapsed / max(n_classifiers, 1),
    )

    return metrics_rows, importance_rows, info_rows, n_classifiers
