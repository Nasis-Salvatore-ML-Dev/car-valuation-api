"""
Compute PSI training baseline from Phase 1 training data.

Reads the training CSV, computes bin edges and expected proportions
for each monitored feature, and writes training_baseline.json.

Also samples 100 rows for the SHAP background dataset.

Run once before first deploy:
    python scripts/compute_baseline.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Input: Phase 1 preprocessed training data (after feature engineering)
TRAINING_DATA_PATH = "data/training/training_set.csv"
BASELINE_OUTPUT = "data/baselines/training_baseline.json"
SHAP_BACKGROUND_OUTPUT = "data/baselines/shap_background.pkl"

MONITORED_FEATURES = ["mileage", "engine_power", "car_age_years", "luxury_tier"]
N_BINS = 10
N_SHAP_BACKGROUND = 100


def compute_baseline(df: pd.DataFrame) -> dict:
    """
    Compute bin edges and expected proportions for each monitored feature.

    Fixed bins from training ensure PSI comparisons are consistent over time.
    """
    baseline = {}
    for feature in MONITORED_FEATURES:
        if feature not in df.columns:
            logger.warning("Feature %r not found in training data — skipping.", feature)
            continue

        values = df[feature].dropna().values

        # Use percentile-based bins to handle skewed distributions
        percentiles = np.linspace(0, 100, N_BINS + 1)
        bin_edges = np.unique(np.percentile(values, percentiles))

        # Ensure at least 2 edges (degenerate case: all values identical)
        if len(bin_edges) < 2:
            logger.warning("Feature %r has no variance — skipping.", feature)
            continue

        # Extend edges to capture outliers
        bin_edges[0] = bin_edges[0] - 1e-9
        bin_edges[-1] = bin_edges[-1] + 1e-9

        counts, _ = np.histogram(values, bins=bin_edges)
        proportions = (counts / counts.sum()).tolist()

        baseline[feature] = {
            "bin_edges": bin_edges.tolist(),
            "expected_proportions": proportions,
            "n_samples": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }

        logger.info(
            "Feature %r: %d bins, mean=%.2f, std=%.2f",
            feature, len(bin_edges) - 1, baseline[feature]["mean"], baseline[feature]["std"]
        )

    return baseline


def main() -> None:
    training_path = Path(TRAINING_DATA_PATH)
    if not training_path.exists():
        logger.error(
            "Training data not found at %s. "
            "Export the Phase 1 training set to this path and re-run.",
            training_path.resolve(),
        )
        sys.exit(1)

    logger.info("Loading training data from %s", training_path)
    df = pd.read_csv(training_path)
    logger.info("Training set shape: %s", df.shape)

    # Compute and save baseline
    baseline = compute_baseline(df)
    output_path = Path(BASELINE_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Baseline written to %s", output_path)

    # Sample SHAP background dataset
    # Use model feature columns only (drop target)
    feature_cols = [c for c in df.columns if c != "actual_price_log"]
    background = df[feature_cols].sample(
        n=min(N_SHAP_BACKGROUND, len(df)), random_state=42
    )
    bg_path = Path(SHAP_BACKGROUND_OUTPUT)
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(background, bg_path)
    logger.info(
        "SHAP background dataset saved: %d rows × %d cols → %s",
        len(background), len(background.columns), bg_path
    )

    logger.info("Baseline computation complete.")


if __name__ == "__main__":
    main()