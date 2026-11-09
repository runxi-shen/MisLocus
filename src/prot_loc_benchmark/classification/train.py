"""XGBoost training with explicit GPU allocation and no silent backend fallback."""
from __future__ import annotations

import json
import logging
import os
import re

import numpy as np
import polars as pl
import xgboost
from xgboost import XGBClassifier

from prot_loc_benchmark.config import XGBOOST_PARAMS

logger = logging.getLogger(__name__)


def select_device(backend: str | None = None) -> str:
    """CPU by default; GPU requires one explicitly allocated visible device."""
    backend = (backend or os.environ.get("MISLOCUS_CLASSIFIER_BACKEND", "cpu")).strip().lower()
    if backend in ("cpu", "auto"):
        return "cpu"
    if backend != "gpu":
        raise ValueError(f"Unknown classifier backend: {backend}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not re.fullmatch(r"GPU-[\da-fA-F]{8}(?:-[\da-fA-F]{4}){3}-[\da-fA-F]{12}", visible):
        raise ValueError("GPU execution requires one full GPU UUID allocation in CUDA_VISIBLE_DEVICES")
    if not xgboost.build_info().get("USE_CUDA", False):
        raise ValueError("Installed XGBoost has no CUDA support")
    return "cuda:0"


def train_and_predict(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str = "Label",
    device: str = "cpu",
    xgb_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    """Fit one classifier; refuse a requested/actual backend mismatch."""
    params = {**XGBOOST_PARAMS, **(xgb_params or {})}

    train_labels = train_df[label_col].to_numpy()
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        logger.warning("Single-class training data, skipping")
        return None

    imbalance = max(n_pos, n_neg) / min(n_pos, n_neg)
    if imbalance > 100:
        logger.warning("Extreme class imbalance %.0f:1, skipping", imbalance)
        return None

    params["scale_pos_weight"] = n_neg / n_pos
    params["device"] = device

    X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
    X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_labels
    y_test = test_df[label_col].to_numpy()

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    actual = json.loads(clf.get_booster().save_config())["learner"]["generic_param"]["device"]
    if actual != device:
        raise RuntimeError(f"XGBoost device fallback: requested {device}, used {actual}")

    if device == "cpu":
        preds = clf.predict_proba(X_test)[:, 1]
    else:
        preds = clf.get_booster().predict(xgboost.DMatrix(X_test, nthread=params["n_jobs"]))
    importances = dict(zip(feature_cols, clf.feature_importances_.tolist()))

    return preds, y_test, importances
