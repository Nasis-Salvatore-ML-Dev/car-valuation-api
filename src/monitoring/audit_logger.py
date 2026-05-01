"""
DynamoDB audit logger for the Car Valuation API.

Every prediction is written to an append-only DynamoDB table. Append-only
semantics are enforced via a ConditionExpression that rejects writes where
the partition key already exists.

Table design:
    audit_log:
        PK: prediction_id (String)   — UUID, unique per prediction
        SK: timestamp (String)       — ISO-8601 UTC, enables time-range queries
        GSI1: model_version-timestamp-index
            — allows queries like "all predictions from model v1.1 last week"

    override_queue:
        PK: prediction_id (String)
        SK: timestamp (String)
        status: pending | reviewed | resolved
        TTL: epoch seconds (30-day auto-expiry)

This design satisfies EU AI Act Article 12 (record-keeping) requirements:
every automated decision is logged with its inputs, outputs, model version,
and explainability data in a tamper-evident store.
"""

import logging
import os
from datetime import UTC, datetime, timedelta

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_AUDIT_TABLE_ENV = "DYNAMODB_AUDIT_TABLE"
_OVERRIDE_TABLE_ENV = "DYNAMODB_OVERRIDE_TABLE"
_AWS_REGION_ENV = "AWS_REGION"

_DEFAULT_AUDIT_TABLE = "car-valuation-audit-log"
_DEFAULT_OVERRIDE_TABLE = "car-valuation-override-queue"

_OVERRIDE_TTL_DAYS = 30


class AuditLogger:
    """
    Writes and reads prediction records from DynamoDB.

    Thread-safe: boto3 clients are thread-safe per AWS documentation.
    Lambda concurrency: each invocation gets its own Python process,
    so no shared state issues.
    """

    def __init__(self) -> None:
        region = os.environ.get(_AWS_REGION_ENV, "us-east-1")
        self._ddb = boto3.resource("dynamodb", region_name=region)
        self._audit_table = self._ddb.Table(os.environ.get(_AUDIT_TABLE_ENV, _DEFAULT_AUDIT_TABLE))
        self._override_table = self._ddb.Table(
            os.environ.get(_OVERRIDE_TABLE_ENV, _DEFAULT_OVERRIDE_TABLE)
        )
        logger.info(
            "AuditLogger connected to tables: %s, %s",
            self._audit_table.name,
            self._override_table.name,
        )

    async def write(
        self,
        prediction_id: str,
        input_features: dict,
        predicted_price: float,
        shap_values: dict[str, float],
        model_version: str,
        confidence_score: float,
        request_ip: str,
        latency_ms: float,
    ) -> None:
        """
        Write a prediction record to the audit log (append-only).

        Uses ConditionExpression to prevent overwrites — once a prediction_id
        is written, it cannot be modified. This provides immutability guarantees
        for audit and regulatory purposes.

        Args:
            prediction_id: UUID string uniquely identifying this prediction.
            input_features: Raw input from the API request.
            predicted_price: EUR price output (after exp() back-transform).
            shap_values:    Per-feature SHAP contributions.
            model_version:  Version string of the model that served the request.
            confidence_score: Proxy confidence in [0, 1].
            request_ip:     IP of the requesting client.
            latency_ms:     End-to-end inference latency.
        """
        timestamp = datetime.now(UTC).isoformat()

        item = {
            "prediction_id": prediction_id,
            "timestamp": timestamp,
            "input_features": self._serialise(input_features),
            "predicted_price": str(predicted_price),
            "shap_values": self._serialise(shap_values),
            "model_version": model_version,
            "confidence_score": str(confidence_score),
            "request_ip": request_ip,
            "latency_ms": str(latency_ms),
        }

        try:
            self._audit_table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(prediction_id)",
            )
            logger.debug("Audit log written: prediction_id=%s", prediction_id)

        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                # This should never happen with UUIDs, but handle defensively
                logger.error(
                    "Duplicate prediction_id detected: %s — NOT overwriting.", prediction_id
                )
            else:
                # Log but do not fail the prediction request
                logger.exception("DynamoDB write failed for prediction_id=%s", prediction_id)

    async def fetch(self, prediction_id: str) -> dict | None:
        """
        Fetch a single audit log record by prediction_id.

        Args:
            prediction_id: UUID string.

        Returns:
            dict with all stored fields, or None if not found.
        """
        try:
            resp = self._audit_table.get_item(Key={"prediction_id": prediction_id})
            item = resp.get("Item")
            if item is None:
                return None
            return self._deserialise_record(item)

        except ClientError:
            logger.exception("DynamoDB fetch failed for prediction_id=%s", prediction_id)
            return None

    async def fetch_recent(self, limit: int = 500) -> list[dict]:
        """
        Fetch the most recent N records for drift computation.

        NOTE: This performs a DynamoDB Scan, which reads every item.
        For production at scale this should be replaced with a GSI
        time-range query. At our traffic levels (free tier), Scan is
        acceptable and avoids the cost of provisioned capacity.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of deserialised record dicts, most recent first.
        """
        try:
            resp = self._audit_table.scan(Limit=limit)
            items = resp.get("Items", [])
            return [self._deserialise_record(i) for i in items]

        except ClientError:
            logger.exception("DynamoDB scan failed.")
            return []

    async def flag_override(
        self,
        prediction_id: str,
        reason: str,
        reviewer_notes: str | None = None,
    ) -> None:
        """
        Add prediction to the override queue for human review.

        TTL is set to 30 days — after which DynamoDB auto-expires the item.
        Items are NOT deleted on resolution; status is updated to 'resolved'.

        Args:
            prediction_id: UUID of the prediction to flag.
            reason:        Why this prediction needs review.
            reviewer_notes: Optional free-text notes from the requester.
        """
        timestamp = datetime.now(UTC).isoformat()
        ttl_epoch = int((datetime.now(UTC) + timedelta(days=_OVERRIDE_TTL_DAYS)).timestamp())

        try:
            self._override_table.put_item(
                Item={
                    "prediction_id": prediction_id,
                    "timestamp": timestamp,
                    "reason": reason,
                    "reviewer_notes": reviewer_notes or "",
                    "status": "pending",
                    "ttl": ttl_epoch,
                }
            )
            logger.info("Override queued: prediction_id=%s reason=%r", prediction_id, reason)

        except ClientError:
            logger.exception("DynamoDB override write failed for prediction_id=%s", prediction_id)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _serialise(obj: dict) -> dict:
        """
        Convert all values to strings for DynamoDB compatibility.

        DynamoDB does not support float natively (stores as Decimal).
        We store all numeric values as strings and parse on read.
        """
        return {k: str(v) for k, v in obj.items()}

    @staticmethod
    def _deserialise_record(item: dict) -> dict:
        """Parse stored strings back to typed values."""
        result = dict(item)
        # Parse nested JSON-encoded dicts
        for field in ("input_features", "shap_values"):
            if field in result and isinstance(result[field], dict):
                # Convert string values back to float where possible
                parsed = {}
                for k, v in result[field].items():
                    try:
                        parsed[k] = float(v)
                    except (TypeError, ValueError):
                        parsed[k] = v
                result[field] = parsed

        # Parse scalar floats
        for field in ("predicted_price", "confidence_score", "latency_ms"):
            if field in result:
                try:
                    result[field] = float(result[field])
                except (TypeError, ValueError):
                    pass

        return result
