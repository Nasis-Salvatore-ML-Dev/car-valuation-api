"""
SHAP explainability module for the Car Valuation API.

Uses TreeExplainer — the exact (not approximate) SHAP algorithm for
tree-based models including scikit-learn Random Forest. Initialised
once per container lifetime at startup.

Key concepts:
    - SHAP value for feature f on prediction p: the marginal contribution
      of feature f's actual value vs its expected value to the final prediction.
    - Unit: EUR (same unit as the prediction), after exp() back-transform.
    - Positive SHAP → feature pushes price UP.
    - Negative SHAP → feature pushes price DOWN.
    - Sum of all SHAP values ≈ predicted_price − expected_price.
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

_BACKGROUND_PATH_DEFAULT = "data/baselines/shap_background.pkl"
_BACKGROUND_PATH_ENV = "SHAP_BACKGROUND_PATH"

# Confidence proxy: RMSE from Phase 1 training (EUR)
_TRAINING_RMSE_EUR = 2875.0
# Wide prediction interval (interval_width > this → low confidence)
_WIDE_INTERVAL_THRESHOLD_EUR = 8000.0


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for the BMW pricing Random Forest.

    Attributes:
        _explainer:       shap.TreeExplainer instance (created once).
        _feature_names:   Column names matching model.feature_names_in_.
        _global_importance: Cached mean |SHAP| per feature over background.
    """

    def __init__(self, bundle) -> None:
        """
        Initialise and pre-compute global importance.

        Args:
            bundle: ModelBundle — must have .model and .encoders.
        """
        background = self._load_background()

        logger.info("Initialising TreeExplainer with background dataset shape=%s", background.shape)
        self._explainer = shap.TreeExplainer(
            bundle.model,
            data=background,
            feature_perturbation="interventional",
        )
        self._feature_names: list[str] = list(background.columns)

        # Pre-compute global importance over background
        logger.info("Computing global SHAP importance over background set.")
        bg_shap = self._explainer.shap_values(background)
        mean_abs = np.abs(bg_shap).mean(axis=0)

        # Convert log-scale SHAP → EUR-scale (approximate: multiply by median price)
        # For RF regression on log(price), SHAP values are in log-EUR units.
        # We report them in log-EUR for consistency with the prediction.
        self._global_importance: dict[str, float] = dict(
            sorted(
                zip(self._feature_names, mean_abs.tolist(), strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        logger.info(
            "Top 3 global drivers: %s",
            list(self._global_importance.items())[:3],
        )

    def explain(self, features_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute SHAP values for a single prediction.

        Args:
            features_df: One-row DataFrame matching model.feature_names_in_.

        Returns:
            dict mapping feature_name → shap_value (log-EUR units).
            Positive = contribution toward higher price.
            Negative = contribution toward lower price.
        """
        if len(features_df) != 1:
            raise ValueError(f"explain() expects exactly 1 row; got {len(features_df)}.")

        shap_vals = self._explainer.shap_values(features_df)

        # shap_vals shape: (1, n_features)
        if hasattr(shap_vals, "values"):
            # shap.Explanation object (newer shap versions)
            raw = shap_vals.values[0]
        elif isinstance(shap_vals, list):
            # Multi-output: take first output (regression → single output)
            raw = shap_vals[0][0]
        else:
            raw = shap_vals[0]

        return {name: float(val) for name, val in zip(self._feature_names, raw, strict=False)}

    def global_importance(self) -> dict[str, float]:
        """
        Return cached global feature importance (mean |SHAP| over background).

        Returns:
            dict mapping feature_name → mean_abs_shap, sorted descending.
        """
        return self._global_importance

    def confidence_score(self, predicted_price: float, bundle) -> float:
        """
        Proxy confidence score based on prediction interval width.

        A wide interval relative to the predicted price indicates the model
        is uncertain. We map interval_width → [0, 1] confidence.

        Args:
            predicted_price: EUR price output from the model.
            bundle:          ModelBundle (used for RMSE reference).

        Returns:
            float in [0, 1]. Higher = more confident.
        """
        interval_width = 2 * 1.96 * _TRAINING_RMSE_EUR
        # Normalise: smaller width relative to price → higher confidence
        normalised_width = interval_width / (predicted_price + 1e-9)
        # Map to [0, 1]: width=0 → score=1.0, width=2.0 → score~0
        confidence = float(max(0.0, min(1.0, 1.0 - normalised_width)))
        return confidence

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_background() -> pd.DataFrame:
        """
        Load the SHAP background reference dataset.

        Background dataset: 100 rows sampled from the training set at
        training time. Stored as a pickled DataFrame. Required by
        TreeExplainer with interventional perturbation.

        Returns:
            pd.DataFrame with the same columns as the model expects.

        Raises:
            FileNotFoundError: if background file is missing.
        """
        path_str = os.environ.get(_BACKGROUND_PATH_ENV, _BACKGROUND_PATH_DEFAULT)
        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(
                f"SHAP background dataset not found at {path.resolve()}. "
                "Run scripts/compute_baseline.py to generate it."
            )

        background: pd.DataFrame = joblib.load(path)
        logger.info("SHAP background loaded: shape=%s", background.shape)
        return background
