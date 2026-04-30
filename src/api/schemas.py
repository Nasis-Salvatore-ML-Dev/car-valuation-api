"""Pydantic request/response schemas for the Car Valuation API."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Input schema for /predict endpoint."""

    model_key: str = Field(..., examples=["320d"], description="BMW model identifier")
    mileage: float = Field(..., ge=0, examples=[120000], description="Mileage in kilometres")
    engine_power: float = Field(..., gt=0, examples=[184], description="Engine power in HP")
    registration_date: str = Field(
        ..., examples=["2015-03-01"], description="Vehicle registration date (YYYY-MM-DD)"
    )
    fuel: str = Field(..., examples=["diesel"])
    paint_color: str = Field(..., examples=["black"])
    car_type: str = Field(..., examples=["sedan"])
    sold_at: str = Field(..., examples=["2020-06-15"], description="Sale date (YYYY-MM-DD)")

    feature_1: bool = Field(default=False)
    feature_2: bool = Field(default=False)
    feature_3: bool = Field(default=False)
    feature_4: bool = Field(default=False)
    feature_5: bool = Field(default=False)
    feature_6: bool = Field(default=False)
    feature_7: bool = Field(default=False)
    feature_8: bool = Field(default=False)

    @field_validator("registration_date", "sold_at")
    @classmethod
    def _validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Date must be YYYY-MM-DD") from exc
        return v

    @field_validator("fuel")
    @classmethod
    def _validate_fuel(cls, v: str) -> str:
        allowed = {"diesel", "petrol", "hybrid_petrol", "electro"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"fuel must be one of {sorted(allowed)}")
        return v

    @field_validator("car_type")
    @classmethod
    def _validate_car_type(cls, v: str) -> str:
        allowed = {"sedan", "suv", "coupe", "convertible", "estate", "hatchback", "van", "subcompact"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"car_type must be one of {sorted(allowed)}")
        return v


class OverrideRequest(BaseModel):
    """Input schema for /override endpoint."""

    prediction_id: str = Field(..., description="UUID from the /predict response")
    reason: str | None = Field(default=None, description="Why this prediction needs review")
    notes: str | None = Field(default=None, description="Reviewer notes")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Output schema for /predict endpoint."""

    prediction_id: str = Field(..., description="Unique identifier for this prediction (UUID)")
    predicted_price: float = Field(..., description="Predicted price in EUR")
    confidence_interval: dict[str, float] = Field(
        ..., description="95% confidence interval: {lower, upper}"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence (0=low, 1=high)"
    )
    shap_values: dict[str, float] = Field(
        ...,
        description=(
            "Per-feature SHAP contributions in EUR. "
            "Positive = pushes price up. Negative = pushes price down."
        ),
    )
    model_version: str = Field(..., description="Model version that served this prediction")
    processing_time_ms: float = Field(..., description="End-to-end latency in milliseconds")
    flagged_for_review: bool = Field(
        ...,
        description="True if confidence < 0.6 and routed to human override queue",
    )


class ExplainResponse(BaseModel):
    """Output schema for /explain/{prediction_id} endpoint."""

    prediction_id: str
    predicted_price: float
    model_version: str
    shap_waterfall: list[dict] = Field(
        ...,
        description=(
            "Waterfall chart data sorted by |SHAP| descending. "
            "Each item: {feature, value, shap_contribution}."
        ),
    )
    top_positive_drivers: list[dict] = Field(..., description="Top 3 features increasing price")
    top_negative_drivers: list[dict] = Field(..., description="Top 3 features decreasing price")


class ExplainGlobalResponse(BaseModel):
    """Output schema for /explain/global endpoint."""

    model_version: str
    feature_importance: dict[str, float] = Field(
        ..., description="Mean absolute SHAP value per feature, sorted descending"
    )
    explanation: str


class OverrideResponse(BaseModel):
    """Output schema for /override endpoint."""

    prediction_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """Output schema for /health endpoint."""

    status: str
    model_version: str
    model_loaded: bool
    timestamp: str


class DriftFeatureReport(BaseModel):
    """PSI result for a single feature."""

    feature: str
    psi: float
    status: str = Field(..., description="stable | monitor | action_required")
    expected_distribution: list[float]
    actual_distribution: list[float]
    bin_edges: list[float]


class DriftReportResponse(BaseModel):
    """Output schema for /drift endpoint."""

    computed_at: str
    n_recent_predictions: int
    overall_status: str = Field(..., description="stable | monitor | action_required")
    features: list[DriftFeatureReport]
    recommendation: str


class BiasSegmentResult(BaseModel):
    """MAE result for one bias segment."""

    segment_name: str
    segment_description: str
    n_samples: int
    mae_eur: float
    relative_to_overall: float = Field(..., description="Segment MAE / overall MAE")
    flagged: bool = Field(..., description="True if MAE > 1.5x overall MAE")


class BiasReportResponse(BaseModel):
    """Output schema for /metrics endpoint."""

    model_version: str
    overall_mae_eur: float
    overall_rmse_eur: float
    overall_r2: float
    bias_segments: list[BiasSegmentResult]
    model_card_s3_uri: str
    computed_at: str
    recommendation: str | None