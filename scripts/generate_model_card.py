"""
Generate model_card.json following Mitchell et al. (2019) format.

The model card is auto-generated from pipeline artefacts so it is always
in sync with the deployed model. Never manually edited.

Run as part of the CD pipeline:
    python scripts/generate_model_card.py

Outputs:
    model_card.json (committed to repo root, uploaded to S3 on deploy)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_VERSION = os.environ.get("MODEL_VERSION", "rand_forest_v1")
BIAS_REPORT_PATH = "data/reports/bias_report.json"
OUTPUT_PATH = "model_card.json"


def load_bias_report() -> dict:
    path = Path(BIAS_REPORT_PATH)
    if not path.exists():
        logger.warning("Bias report not found — model card will have placeholder metrics.")
        return {}
    with open(path) as f:
        return json.load(f)


def generate_model_card(bias_report: dict) -> dict:
    return {
        "model_card_version": "1.0",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "model_details": {
            "name": "BMW Car Valuation Model",
            "version": MODEL_VERSION,
            "type": "RandomForestRegressor (scikit-learn)",
            "task": "Regression — predict used BMW vehicle price in EUR",
            "training_framework": "scikit-learn 1.x",
            "serialisation_formats": ["sklearn pkl", "ONNX"],
            "developers": ["Salvatore — Phase 1 BMW Pricing Project"],
            "contact": "See GitHub repository",
            "license": "MIT",
        },
        "intended_use": {
            "primary_use": (
                "Automated price estimation for used BMW vehicles sold at Aures Holdings "
                "(Resulmatic platform). Intended to assist pricing analysts, not replace them."
            ),
            "intended_users": ["Pricing analysts", "Fleet managers", "Used car buyers"],
            "out_of_scope": [
                "Pricing non-BMW vehicles (model not trained on other manufacturers)",
                "Real-time market pricing (model is retrained periodically, not live)",
                "Vehicles with mileage > 300,000 km (outside training distribution)",
                "Vehicles manufactured before 2005 (sparse training data)",
            ],
        },
        "factors": {
            "relevant_factors": [
                "Vehicle age (car_age_years)",
                "Mileage (mileage in km)",
                "Engine power (HP)",
                "BMW model series (1/3/5/7/X series)",
                "Fuel type",
                "Body type (car_type)",
            ],
            "evaluation_factors": [
                "Vehicle segment (luxury vs. economy)",
                "Vehicle age group (old vs. new)",
                "Mileage bracket (high vs. low)",
                "Fuel type",
            ],
        },
        "metrics": {
            "performance_measures": ["MAE (EUR)", "RMSE (EUR)", "R²", "MAPE (%)"],
            "overall": {
                "mae_eur": bias_report.get("overall_mae_eur", "N/A"),
                "rmse_eur": bias_report.get("overall_rmse_eur", "N/A"),
                "r2": bias_report.get("overall_r2", "N/A"),
            },
            "phase_1_training_results": {
                "mae_eur": 1946,
                "rmse_eur": 2875,
                "r2": 0.776,
                "mape_pct": 20.48,
                "note": (
                    "Custom business metrics (Tail Rate, TC-APE) did not meet targets. "
                    "Dataset of 3,874 rows lacks information density for tighter targets. "
                    "Honest engineering decision: deploy at achievable performance rather than overfit."
                ),
            },
            "segment_breakdown": bias_report.get("bias_segments", []),
        },
        "evaluation_data": {
            "dataset": "Aures Holdings BMW transaction records (3,874 rows, 17 raw features)",
            "split": "60/20/20 chronological train/val/test split",
            "preprocessing": "Feature engineering: 38 features including BMW series encoding, luxury tier, temporal features, target encoding for categorical variables",
        },
        "training_data": {
            "description": "BMW used car transactions from the Resulmatic platform",
            "size": "3,874 rows after cleaning",
            "features": 38,
            "date_range": "2020–2024 (approximate)",
        },
        "quantitative_analyses": {
            "unitary_results": "See segment_breakdown above",
            "intersectional_results": "Not computed in this version. Recommended for v2.",
        },
        "ethical_considerations": {
            "data": [
                "No personally identifiable information (PII) in training data",
                "Vehicles from a single country market — cross-market generalisability not validated",
            ],
            "human_life": [
                "Model is advisory. No automated financial decisions are made without human review.",
                "Low-confidence predictions are routed to human override queue.",
            ],
            "mitigations": [
                "Bias testing by vehicle segment on every deployment",
                "SHAP explainability on every prediction",
                "Human override endpoint for flagged predictions",
                "Immutable DynamoDB audit log for all predictions",
            ],
        },
        "caveats_and_recommendations": {
            "caveats": [
                "Model trained on pre-2025 data. Used car market disrupted by COVID and EV transition.",
                "Luxury segment MAE is higher in absolute EUR terms — budget for human review there.",
                "MAPE of 20.48% is acceptable for a baseline but insufficient for automated transactions above €30K.",
            ],
            "recommendations": [
                "Retrain quarterly with fresh transaction data",
                "Add market index features (used car price indices) to capture macro shifts",
                "Consider segment-specific models for luxury (tier >= 3)",
                "Validate PSI thresholds empirically with 6 months of production data",
            ],
        },
        "mlops": {
            "deployment_platform": "AWS Lambda (us-east-1)",
            "api_gateway": "AWS API Gateway HTTP API",
            "audit_log": "DynamoDB — append-only, immutable",
            "monitoring": "CloudWatch — latency p50/p99, error rate, PSI drift metrics",
            "drift_detection": "PSI (Population Stability Index) on 4 features",
            "explainability": "SHAP TreeExplainer — per-prediction and global",
            "ci_cd": "GitHub Actions — push to main → ECR → Lambda",
        },
    }


def main() -> None:
    bias_report = load_bias_report()
    card = generate_model_card(bias_report)

    output = Path(OUTPUT_PATH)
    with open(output, "w") as f:
        json.dump(card, f, indent=2)

    logger.info("Model card written to %s", output)
    logger.info("Overall MAE: %.0f EUR", card["metrics"]["overall"].get("mae_eur") or 0)

    flagged = [s for s in card["metrics"]["segment_breakdown"] if s.get("flagged")]
    if flagged:
        logger.warning(
            "Flagged segments (MAE > 1.5x overall): %s",
            [s["segment_name"] for s in flagged],
        )


if __name__ == "__main__":
    main()