#!/usr/bin/env bash
# Create DynamoDB tables for the Car Valuation API.
# Idempotent: safe to run multiple times.
# Usage: bash infra/scripts/create_dynamodb.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
AUDIT_TABLE="car-valuation-audit-log"
OVERRIDE_TABLE="car-valuation-override-queue"

echo "Creating DynamoDB tables in region: $REGION"

# ── audit_log table ──────────────────────────────────────────────────────────
if aws dynamodb describe-table --table-name "$AUDIT_TABLE" --region "$REGION" \
   > /dev/null 2>&1; then
    echo "[SKIP] Table '$AUDIT_TABLE' already exists."
else
    aws dynamodb create-table \
        --table-name "$AUDIT_TABLE" \
        --attribute-definitions \
            AttributeName=prediction_id,AttributeType=S \
            AttributeName=timestamp,AttributeType=S \
            AttributeName=model_version,AttributeType=S \
        --key-schema \
            AttributeName=prediction_id,KeyType=HASH \
            AttributeName=timestamp,KeyType=RANGE \
        --global-secondary-indexes '[
            {
                "IndexName": "model_version-timestamp-index",
                "KeySchema": [
                    {"AttributeName": "model_version", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"}
                ],
                "Projection": {"ProjectionType": "ALL"}
            }
        ]' \
        --billing-mode PAY_PER_REQUEST \
        --region "$REGION"
    echo "[CREATED] Table '$AUDIT_TABLE'."
fi

# ── override_queue table ─────────────────────────────────────────────────────
if aws dynamodb describe-table --table-name "$OVERRIDE_TABLE" --region "$REGION" \
   > /dev/null 2>&1; then
    echo "[SKIP] Table '$OVERRIDE_TABLE' already exists."
else
    aws dynamodb create-table \
        --table-name "$OVERRIDE_TABLE" \
        --attribute-definitions \
            AttributeName=prediction_id,AttributeType=S \
            AttributeName=timestamp,AttributeType=S \
        --key-schema \
            AttributeName=prediction_id,KeyType=HASH \
            AttributeName=timestamp,KeyType=RANGE \
        --billing-mode PAY_PER_REQUEST \
        --region "$REGION"

    # Enable TTL for 30-day auto-expiry
    aws dynamodb update-time-to-live \
        --table-name "$OVERRIDE_TABLE" \
        --time-to-live-specification Enabled=true,AttributeName=ttl \
        --region "$REGION"

    echo "[CREATED] Table '$OVERRIDE_TABLE' with 30-day TTL."
fi

echo "DynamoDB setup complete."