"""
Bias testing suite for the Car Valuation API.

Computes MAE broken down by vehicle segment to detect systematic
performance disparities. A model that is accurate on average but
badly miscalibrated on specific segments creates unfair outcomes
(e.g. consistently undervaluing luxury cars, hurting sellers).

Segments tested:
    - Luxury vs. economy (luxury_tier >= 3 vs < 3)
    - Old vs. new (car_age_years > 5 vs <= 5)
    - High mileage vs. low (mileage > 150K vs <= 150K)
    - Fuel type (diesel, petrol, hybrid, electric)

Alert threshold: segment MAE > 1.5× overall MAE → flagged.

The bias report is:
    1. Computed at container startup from Phase 1 validation set.
    2. Returned by the /metrics endpoint.
    3. Included in the auto-generated model card.
    4. Re-run on every CD pipeline execution.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VALIDATION_DATA_PATH = "data/validation/validation_set.csv"
_BIAS_REPORT_PATH = "data/reports/bias_report.json"
_MODEL_CARD_PATH = "model_card.json"

_FLAG_THRESHOLD = 1.5  # segment MAE / overall MAE > this → flagged


class BiasTestSuite:
    """
    Runs MAE bias analysis across vehicle segments.

    The suite loads from a cached report if available (generated during CD),
    falling back to computing from the validation set at startup.
    """

    def __init__(self, bundle) -> None:
        self._bundle = bundle
        self._report: dict | None = None
        self._load_or_compute()

    def cached_report(self) -> dict:
        """Return the cached bias report (computed once at startup)."""
        if self._report is None:
            self._report = self._empty_report()
        return self._report

    def run(self, validation_df: pd.DataFrame | None = None) -> dict:
        """
        Run bias tests on a validation DataFrame.

        Args:
            validation_df: Optional pre-loaded validation data.
                           If None, loads from _VALIDATION_DATA_PATH.

        Returns:
            Bias report dict matching BiasReportResponse schema.
        """
        if validation_df is None:
            validation_df = self._load_validation_data()

        if validation_df is None or len(validation_df) == 0:
            logger.warning("No validation data available — returning empty bias report.")
            return self._empty_report()

        # Predictions on validation set
        features = validation_df.drop(columns=["actual_price_log"], errors="ignore")
        actual_log = validation_df.get("actual_price_log")

        if actual_log is None:
            logger.warning("Validation set missing 'actual_price_log' column.")
            return self._empty_report()

        predicted_log = self._bundle.model.predict(features)
        actual_eur = np.exp(actual_log.values)
        predicted_eur = np.exp(predicted_log)

        overall_mae = float(np.mean(np.abs(actual_eur - predicted_eur)))
        overall_rmse = float(np.sqrt(np.mean((actual_eur - predicted_eur) ** 2)))
        r2 = float(
            1
            - np.sum((actual_eur - predicted_eur) ** 2)
            / np.sum((actual_eur - actual_eur.mean()) ** 2)
        )

        segments = self._compute_segments(validation_df, actual_eur, predicted_eur, overall_mae)

        flagged_segments = [s for s in segments if s["flagged"]]
        recommendation = None
        if flagged_segments:
            names = [s["segment_name"] for s in flagged_segments]
            recommendation = (
                f"Segments with MAE > 1.5× overall: {names}. "
                "Investigate whether additional features or segment-specific models would help. "
                "Document limitations in the model card."
            )

        report = {
            "model_version": self._bundle.version,
            "overall_mae_eur": round(overall_mae, 2),
            "overall_rmse_eur": round(overall_rmse, 2),
            "overall_r2": round(r2, 4),
            "bias_segments": segments,
            "model_card_s3_uri": f"s3://car-valuation-models/{self._bundle.version}/model_card.json",
            "computed_at": datetime.now(UTC).isoformat(),
            "recommendation": recommendation,
        }

        # Cache and persist
        self._report = report
        self._save_report(report)
        return report

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_or_compute(self) -> None:
        """Load cached report or compute from validation data."""
        report_path = Path(_BIAS_REPORT_PATH)
        if report_path.exists():
            with open(report_path) as f:
                self._report = json.load(f)
            logger.info("Bias report loaded from cache: %s", report_path)
        else:
            logger.info("No cached bias report — computing from validation set.")
            self._report = self.run()

    @staticmethod
    def _load_validation_data() -> pd.DataFrame | None:
        path = Path(_VALIDATION_DATA_PATH)
        if not path.exists():
            logger.warning("Validation set not found at %s.", path)
            return None
        return pd.read_csv(path)

    def _compute_segments(
        self,
        df: pd.DataFrame,
        actual_eur: np.ndarray,
        predicted_eur: np.ndarray,
        overall_mae: float,
    ) -> list[dict]:
        segments = []
        abs_errors = np.abs(actual_eur - predicted_eur)

        def _add_segment(name: str, description: str, mask: np.ndarray) -> None:
            n = int(mask.sum())
            if n < 10:
                return
            seg_mae = float(np.mean(abs_errors[mask]))
            ratio = seg_mae / (overall_mae + 1e-9)
            segments.append(
                {
                    "segment_name": name,
                    "segment_description": description,
                    "n_samples": n,
                    "mae_eur": round(seg_mae, 2),
                    "relative_to_overall": round(ratio, 3),
                    "flagged": ratio > _FLAG_THRESHOLD,
                }
            )

        if "luxury_tier" in df.columns:
            luxury_mask = df["luxury_tier"].values >= 3
            _add_segment(
                "luxury", "luxury_tier >= 3 (5 Series, 7 Series, X5, M-series)", luxury_mask
            )
            _add_segment("economy", "luxury_tier < 3 (1 Series, 3 Series, X1, X3)", ~luxury_mask)

        if "car_age_years" in df.columns:
            old_mask = df["car_age_years"].values > 5
            _add_segment("old_vehicles", "car_age_years > 5", old_mask)
            _add_segment("new_vehicles", "car_age_years <= 5", ~old_mask)

        if "mileage" in df.columns:
            high_mileage_mask = df["mileage"].values > 150_000
            _add_segment("high_mileage", "mileage > 150,000 km", high_mileage_mask)
            _add_segment("low_mileage", "mileage <= 150,000 km", ~high_mileage_mask)

        if "fuel" in df.columns:
            for fuel_type in ["diesel", "petrol", "hybrid_petrol", "electro"]:
                mask = df["fuel"].values == fuel_type
                if mask.sum() >= 10:
                    _add_segment(
                        f"fuel_{fuel_type}",
                        f"fuel type = {fuel_type}",
                        mask,
                    )

        return segments

    @staticmethod
    def _save_report(report: dict) -> None:
        path = Path(_BIAS_REPORT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Bias report saved to %s.", path)

    def _empty_report(self) -> dict:
        return {
            "model_version": self._bundle.version,
            "overall_mae_eur": 0.0,
            "overall_rmse_eur": 0.0,
            "overall_r2": 0.0,
            "bias_segments": [],
            "model_card_s3_uri": "",
            "computed_at": datetime.now(UTC).isoformat(),
            "recommendation": "Validation data not available at startup.",
        }
