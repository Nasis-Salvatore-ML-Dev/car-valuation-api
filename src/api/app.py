"""
Car Valuation API — production FastAPI application.

Deployed on AWS Lambda via Mangum adapter. Provides car price predictions
with SHAP explainability, drift detection, human override routing,
and immutable DynamoDB audit logging.

Endpoints:
    POST /predict           — Price prediction with SHAP values
    GET  /explain/{id}      — Full SHAP breakdown for a logged prediction
    GET  /explain/global    — Global feature importance (mean |SHAP|)
    POST /override          — Flag prediction for human review
    GET  /drift             — Current PSI drift report
    GET  /health            — Liveness + model version
    GET  /metrics           — Bias report + model card
"""

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from src.api.middleware import LoggingMiddleware
from src.api.schemas import (
    BiasReportResponse,
    DriftReportResponse,
    ExplainGlobalResponse,
    ExplainResponse,
    HealthResponse,
    OverrideRequest,
    OverrideResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.explainability.shap_explainer import SHAPExplainer
from src.monitoring.audit_logger import AuditLogger
from src.monitoring.bias_tester import BiasTestSuite
from src.monitoring.drift import DriftMonitor
from src.utils.model_loader import ModelBundle, load_model_bundle

# ---------------------------------------------------------------------------
# Logging — STDERR for errors, STDOUT for info
# ---------------------------------------------------------------------------
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(logging.INFO)
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[_stdout_handler, _stderr_handler],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (loaded once per Lambda container lifetime)
# ---------------------------------------------------------------------------
_model_bundle: ModelBundle | None = None
_shap_explainer: SHAPExplainer | None = None
_drift_monitor: DriftMonitor | None = None
_audit_logger: AuditLogger | None = None
_bias_suite: BiasTestSuite | None = None


def _get_model() -> ModelBundle:
    if _model_bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service initialising.",
        )
    return _model_bundle


# ---------------------------------------------------------------------------
# Lifespan: initialise all heavy objects once per container
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_bundle, _shap_explainer, _drift_monitor, _audit_logger, _bias_suite

    logger.info("Container startup — loading model and explainer.")
    try:
        _model_bundle = load_model_bundle()
        logger.info("Model bundle loaded: %s", _model_bundle.version)
    except Exception:
        logger.exception("Fatal: model loading failed.")
        raise

    # SHAP is non-critical — service degrades gracefully without it
    try:
        _shap_explainer = SHAPExplainer(_model_bundle)
        logger.info("SHAP explainer initialised.")
    except Exception:
        _shap_explainer = None
        logger.exception(
            "SHAP explainer failed to initialise. "
            "Predictions will be served without SHAP values."
        )

    try:
        _drift_monitor = DriftMonitor()
        logger.info("Drift monitor initialised.")

        _audit_logger = AuditLogger()
        logger.info("Audit logger initialised.")

        _bias_suite = BiasTestSuite(_model_bundle)
        logger.info("Bias test suite initialised.")
    except Exception:
        logger.exception("Fatal: monitoring components failed to initialise.")
        raise

    yield  # application serves requests here

    logger.info("Container shutdown.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Car Valuation API",
    description=(
        "Production-grade car price prediction API with SHAP explainability, "
        "PSI drift detection, and DynamoDB audit logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness check. Returns 503 if model is not loaded."""
    bundle = _get_model()
    return HealthResponse(
        status="healthy",
        model_version=bundle.version,
        model_loaded=True,
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
)
async def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    """
    Predict BMW car price with SHAP explainability.

    Every prediction is:
    - SHAP-explained (per-feature contribution) when available
    - Written to DynamoDB audit log (append-only)
    - Auto-routed to override queue if confidence is low
    """
    start_ts = time.perf_counter()
    prediction_id = str(uuid.uuid4())

    bundle = _get_model()

    try:
        features_df = bundle.preprocess(payload)
        predicted_price_log = bundle.model.predict(features_df)[0]
        predicted_price = float(bundle.inverse_transform_price(predicted_price_log))

        # SHAP: graceful degradation if explainer is unavailable
        if _shap_explainer is not None:
            shap_values = _shap_explainer.explain(features_df)
            confidence_score = _shap_explainer.confidence_score(predicted_price, bundle)
        else:
            shap_values = {}
            confidence_score = 0.5
            logger.warning(
                "prediction_id=%s served without SHAP (explainer unavailable)",
                prediction_id,
            )

        latency_ms = (time.perf_counter() - start_ts) * 1000

        # Async-style: audit log write (DynamoDB PutItem, <5ms on warm)
        await _audit_logger.write(
            prediction_id=prediction_id,
            input_features=payload.model_dump(),
            predicted_price=predicted_price,
            shap_values=shap_values,
            model_version=bundle.version,
            confidence_score=confidence_score,
            request_ip=request.client.host if request.client else "unknown",
            latency_ms=latency_ms,
        )

        # Route to override queue if confidence low
        low_confidence = confidence_score < 0.6
        if low_confidence:
            await _audit_logger.flag_override(prediction_id=prediction_id, reason="low_confidence")
            logger.warning(
                "prediction_id=%s routed to override queue (confidence=%.2f)",
                prediction_id,
                confidence_score,
            )

        rmse_eur = 2875.0  # Phase 1 training RMSE
        margin = 1.96 * rmse_eur

        logger.info(
            "prediction_id=%s price=%.0f confidence=%.2f latency=%.1fms",
            prediction_id,
            predicted_price,
            confidence_score,
            latency_ms,
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            predicted_price=predicted_price,
            confidence_interval={
                "lower": max(0.0, predicted_price - margin),
                "upper": predicted_price + margin,
            },
            confidence_score=confidence_score,
            shap_values=shap_values,
            model_version=bundle.version,
            processing_time_ms=latency_ms,
            flagged_for_review=low_confidence,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for prediction_id=%s", prediction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.get(
    "/explain/{prediction_id}",
    response_model=ExplainResponse,
    tags=["explainability"],
)
async def explain_prediction(prediction_id: str) -> ExplainResponse:
    """
    Return full SHAP breakdown for a logged prediction.

    Fetches the original inputs and SHAP values from DynamoDB.
    Returns waterfall-chart-ready data (sorted by |SHAP| descending).
    """
    record = await _audit_logger.fetch(prediction_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id!r} not found in audit log.",
        )

    shap_raw: dict[str, float] = record["shap_values"]
    waterfall = sorted(
        [
            {"feature": k, "value": record["input_features"].get(k), "shap_contribution": v}
            for k, v in shap_raw.items()
        ],
        key=lambda x: abs(x["shap_contribution"]),
        reverse=True,
    )

    return ExplainResponse(
        prediction_id=prediction_id,
        predicted_price=record["predicted_price"],
        model_version=record["model_version"],
        shap_waterfall=waterfall,
        top_positive_drivers=[w for w in waterfall if w["shap_contribution"] > 0][:3],
        top_negative_drivers=[w for w in waterfall if w["shap_contribution"] < 0][:3],
    )


@app.get(
    "/explain/global",
    response_model=ExplainGlobalResponse,
    tags=["explainability"],
)
async def explain_global() -> ExplainGlobalResponse:
    """
    Global feature importance: mean absolute SHAP value per feature.

    Computed at container startup over the background reference dataset.
    Represents which features drive the model across the population.
    """
    if _shap_explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer is unavailable. Global importance cannot be computed.",
        )
    importance = _shap_explainer.global_importance()
    return ExplainGlobalResponse(
        model_version=_get_model().version,
        feature_importance=importance,
        explanation=(
            "Mean absolute SHAP value per feature across the background reference dataset. "
            "Higher = more influential in the model's predictions."
        ),
    )


@app.post(
    "/override",
    response_model=OverrideResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["human-oversight"],
)
async def override(payload: OverrideRequest) -> OverrideResponse:
    """
    Flag a prediction for human review.

    Predictions are auto-flagged when confidence < 0.6.
    This endpoint allows manual flagging of any prediction.
    EU AI Act Article 14: human oversight for high-stakes decisions.
    """
    record = await _audit_logger.fetch(payload.prediction_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {payload.prediction_id!r} not found.",
        )

    await _audit_logger.flag_override(
        prediction_id=payload.prediction_id,
        reason=payload.reason or "manual_review_requested",
        reviewer_notes=payload.notes,
    )

    return OverrideResponse(
        prediction_id=payload.prediction_id,
        status="queued_for_review",
        message="Prediction flagged. A human reviewer will assess within 24h.",
    )


@app.get(
    "/drift",
    response_model=DriftReportResponse,
    tags=["monitoring"],
)
async def drift_report() -> DriftReportResponse:
    """
    Current PSI drift report.

    Computes Population Stability Index between training baseline
    and recent predictions stored in DynamoDB.
    PSI < 0.1 = stable. 0.1–0.2 = monitor. > 0.2 = action required.
    """
    recent_records = await _audit_logger.fetch_recent(limit=500)
    report = _drift_monitor.compute_report(recent_records)

    # Push to CloudWatch for alarming
    _drift_monitor.publish_to_cloudwatch(report)

    return DriftReportResponse(**report)


@app.get(
    "/metrics",
    response_model=BiasReportResponse,
    tags=["monitoring"],
)
async def metrics() -> BiasReportResponse:
    """
    Bias report and model card.

    Returns MAE breakdown by vehicle segment (luxury vs economy,
    old vs new, high-mileage vs low). Flags subgroups where MAE
    exceeds 1.5x the overall model MAE.
    """
    report = _bias_suite.cached_report()
    return BiasReportResponse(**report)


# ---------------------------------------------------------------------------
# Lambda handler (Mangum wraps the ASGI app)
# ---------------------------------------------------------------------------
handler = Mangum(app, lifespan="on")
