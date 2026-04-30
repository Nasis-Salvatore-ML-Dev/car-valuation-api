"""
Model loader for the Car Valuation API.

Loads the trained Random Forest bundle (model + encoders + target encodings)
from S3 or local filesystem depending on the MODEL_PATH environment variable.

ModelBundle is the single object passed around the application — it owns
preprocessing, prediction, and price back-transformation.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Environment variable: s3://bucket/key or local path
_MODEL_PATH_ENV = "MODEL_PATH"
_MODEL_VERSION_ENV = "MODEL_VERSION"
_DEFAULT_LOCAL_PATH = "models/rand_forest_v1.pkl"


@dataclass
class ModelBundle:
    """
    Container for all model artefacts needed at inference time.

    Attributes:
        model:           Trained sklearn RandomForestRegressor.
        encoders:        Label encoders keyed by column name.
        target_encodings: Mean-target encodings keyed by column name.
        version:         Model version string (e.g. "rand_forest_v1").
    """

    model: object
    encoders: dict
    target_encodings: dict
    version: str

    def preprocess(self, payload) -> pd.DataFrame:
        """
        Transform a PredictionRequest into the feature DataFrame the model expects.

        Mirrors the feature engineering logic from Phase 1 exactly.
        """
        from src.api.preprocessing import build_feature_dataframe
        return build_feature_dataframe(payload, self.encoders, self.target_encodings)

    def inverse_transform_price(self, log_price: float) -> float:
        """Convert log-scale model output back to EUR."""
        return float(np.exp(log_price))


def load_model_bundle() -> ModelBundle:
    """
    Load the model bundle from S3 or local path.

    Resolution order:
        1. MODEL_PATH env var (supports s3:// and local paths)
        2. Default local path: models/rand_forest_v1.pkl

    Returns:
        ModelBundle: loaded and validated bundle ready for inference.

    Raises:
        FileNotFoundError: if model artefact cannot be located.
        RuntimeError: if artefact format is unrecognised.
    """
    model_path = os.environ.get(_MODEL_PATH_ENV, _DEFAULT_LOCAL_PATH)
    version = os.environ.get(_MODEL_VERSION_ENV, "rand_forest_v1")

    logger.info("Loading model from: %s", model_path)

    if model_path.startswith("s3://"):
        raw = _load_from_s3(model_path)
    else:
        raw = _load_from_local(model_path)

    if isinstance(raw, dict):
        model = raw["model"]
        encoders = raw.get("encoders", {})
        target_encodings = raw.get("target_encodings", {})
        logger.info(
            "Loaded model dict: %d encoders, %d target encodings",
            len(encoders),
            len(target_encodings),
        )
    else:
        # Legacy format: bare model object
        model = raw
        encoders = {}
        target_encodings = {}
        logger.warning("Legacy model format: no encoders or target encodings found.")

    bundle = ModelBundle(
        model=model,
        encoders=encoders,
        target_encodings=target_encodings,
        version=version,
    )

    # Sanity check: model must have predict()
    if not hasattr(bundle.model, "predict"):
        raise RuntimeError("Loaded artefact does not have a predict() method.")

    logger.info("Model bundle ready: version=%s", bundle.version)
    return bundle


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_from_local(path: str) -> object:
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"Model not found at local path: {local_path.resolve()}")
    return joblib.load(local_path)


def _load_from_s3(s3_uri: str) -> object:
    """
    Download model from S3 to /tmp (Lambda writable path) and load.

    Args:
        s3_uri: Full S3 URI, e.g. s3://my-bucket/v1.0/rand_forest_v1.pkl

    Returns:
        Deserialized model object.
    """
    import io

    import boto3

    # Parse s3://bucket/key
    without_scheme = s3_uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    filename = Path(key).name
    local_tmp = Path("/tmp") / filename

    logger.info("Downloading s3://%s/%s → %s", bucket, key, local_tmp)

    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    s3.download_fileobj(bucket, key, buffer)
    buffer.seek(0)

    obj = joblib.load(buffer)
    logger.info("Model downloaded and loaded from S3.")
    return obj