"""Feature engineering. Extracted from Phase 1 app.py on Day 1."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_feature_dataframe(payload, encoders, target_encodings) -> pd.DataFrame:
    """
    Preprocess input features to match EXACT training format

    Model expects 38 features in this EXACT order:
    1-17: Raw features (maker_key, model_key, mileage, ... sold_at)
    18-35: Engineered features (bmw_series, luxury_tier, ... is_old_car)
    36-38: Target-encoded features (fuel_encoded, color_encoded, car_type_encoded)

    CRITICAL: Model wants BOTH raw categoricals (label-encoded) AND target-encoded versions!
    """

    # Convert to dict (Pydantic V2)
    data = payload.model_dump()

    # Calculate car age
    reg_date = pd.to_datetime(data["registration_date"])
    sold_date = pd.to_datetime(data["sold_at"])
    data["car_age_years"] = (sold_date - reg_date).days / 365.25

    # Extract BMW series and luxury indicators
    model_key_upper = data["model_key"].upper()
    if "X5" in model_key_upper:
        bmw_series = "X5"
        luxury_tier = 3
        is_luxury = 1
    elif "X3" in model_key_upper:
        bmw_series = "X3"
        luxury_tier = 2
        is_luxury = 0
    elif "X1" in model_key_upper:
        bmw_series = "X1"
        luxury_tier = 1
        is_luxury = 0
    elif "7" in model_key_upper:
        bmw_series = "7_series"
        luxury_tier = 4
        is_luxury = 1
    elif "5" in model_key_upper:
        bmw_series = "5_series"
        luxury_tier = 3
        is_luxury = 1
    elif "3" in model_key_upper:
        bmw_series = "3_series"
        luxury_tier = 2
        is_luxury = 0
    elif "1" in model_key_upper:
        bmw_series = "1_series"
        luxury_tier = 1
        is_luxury = 0
    else:
        bmw_series = "other_series"
        luxury_tier = 2
        is_luxury = 0

    data["bmw_series"] = bmw_series
    data["luxury_tier"] = luxury_tier
    data["is_luxury"] = is_luxury
    data["is_performance"] = 1 if "M" in model_key_upper else 0

    # Temporal features
    data["registration_year"] = reg_date.year
    data["registration_month"] = reg_date.month
    data["registration_quarter"] = reg_date.quarter
    data["is_summer_registration"] = 1 if reg_date.month in [6, 7, 8] else 0
    data["is_year_end_registration"] = 1 if reg_date.month in [11, 12] else 0

    # Interaction features
    data["age_mileage_interaction"] = data["car_age_years"] * data["mileage"] / 10000
    data["mileage_per_power"] = data["mileage"] / (data["engine_power"] + 1)
    data["annual_mileage"] = data["mileage"] / (data["car_age_years"] + 1)
    data["power_age_ratio"] = data["engine_power"] / (data["car_age_years"] + 1)
    data["luxury_mileage_interaction"] = is_luxury * data["mileage"] / 10000
    data["luxury_age_interaction"] = is_luxury * data["car_age_years"]
    data["is_high_mileage"] = 1 if (data["mileage"] / (data["car_age_years"] + 1)) > 25000 else 0
    data["is_old_car"] = 1 if data["car_age_years"] > 8 else 0

    # Add maker_key (always BMW)
    data["maker_key"] = "BMW"

    # Convert binary features to integers
    for i in range(1, 9):
        feature_name = f"feature_{i}"
        data[feature_name] = int(data.get(feature_name, False))

    # Create DataFrame
    df = pd.DataFrame([data])

    # ------------------------------------------------------------------
    # Apply label encoders to raw categorical columns
    # This includes: maker_key, model_key, fuel, paint_color, car_type
    # ------------------------------------------------------------------
    if encoders:
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    # Map each value to its integer label; unseen categories get -1
                    df[col] = (
                        df[col]
                        .astype(str)
                        .apply(
                            lambda x, enc=encoder: enc.transform([x])[0]
                            if x in enc.classes_
                            else -1
                        )
                    )
                except Exception as e:
                    logger.warning(f"Encoding error for column {col}: {e}")
                    df[col] = -1
    else:
        # Fallback if no encoders (should not happen)
        logger.warning("No encoders found; using default mapping.")
        for col in ["maker_key", "model_key", "fuel", "paint_color", "car_type"]:
            if col in df.columns:
                df[col] = 0

    # ------------------------------------------------------------------
    # Create target-encoded columns using saved mappings
    # These are: fuel_encoded, color_encoded, car_type_encoded
    # ------------------------------------------------------------------
    if target_encodings:
        # Fuel
        fuel_map = target_encodings.get("fuel", {})
        df["fuel_encoded"] = (
            df["fuel"].map(fuel_map).fillna(np.mean(list(fuel_map.values())) if fuel_map else 0)
        )

        # Color (paint_color)
        color_map = target_encodings.get("color", {})
        df["color_encoded"] = (
            df["paint_color"]
            .map(color_map)
            .fillna(np.mean(list(color_map.values())) if color_map else 0)
        )

        # Car type
        car_type_map = target_encodings.get("car_type", {})
        df["car_type_encoded"] = (
            df["car_type"]
            .map(car_type_map)
            .fillna(np.mean(list(car_type_map.values())) if car_type_map else 0)
        )
    else:
        # Fallback (should not happen)
        logger.warning("Target encodings missing; using zeros.")
        df["fuel_encoded"] = 0
        df["color_encoded"] = 0
        df["car_type_encoded"] = 0

    # Ensure numeric columns are float
    numeric_cols = [
        "mileage",
        "engine_power",
        "car_age_years",
        "luxury_tier",
        "is_luxury",
        "is_performance",
        "registration_year",
        "registration_month",
        "registration_quarter",
        "is_summer_registration",
        "is_year_end_registration",
        "age_mileage_interaction",
        "mileage_per_power",
        "annual_mileage",
        "power_age_ratio",
        "luxury_mileage_interaction",
        "luxury_age_interaction",
        "is_high_mileage",
        "is_old_car",
        "fuel_encoded",
        "color_encoded",
        "car_type_encoded",
    ] + [f"feature_{i}" for i in range(1, 9)]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # CRITICAL: Model expects features in specific order
    # Reorder columns to match model's feature_names_in_
    expected_order = [
        "maker_key",
        "model_key",
        "mileage",
        "engine_power",
        "registration_date",
        "fuel",
        "paint_color",
        "car_type",
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6",
        "feature_7",
        "feature_8",
        "sold_at",
        "bmw_series",
        "luxury_tier",
        "is_luxury",
        "is_performance",
        "registration_year",
        "registration_month",
        "registration_quarter",
        "is_summer_registration",
        "is_year_end_registration",
        "car_age_years",
        "age_mileage_interaction",
        "mileage_per_power",
        "annual_mileage",
        "power_age_ratio",
        "luxury_mileage_interaction",
        "luxury_age_interaction",
        "is_high_mileage",
        "is_old_car",
        "fuel_encoded",
        "color_encoded",
        "car_type_encoded",
    ]

    # Reorder DataFrame to match expected feature order
    df = df[expected_order]

    return df
