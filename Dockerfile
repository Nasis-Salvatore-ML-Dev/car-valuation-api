# ============================================================================
# Car Valuation API — Lambda Container Image
# Multi-stage build: builder installs deps, runtime is minimal
# Target: AWS Lambda container image format (arm64 for cost efficiency)
# ============================================================================

# Stage 1: Build — install all dependencies including compile-time deps
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some packages (numpy, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --target=/build/packages -r requirements.txt

# ============================================================================
# Stage 2: Runtime — minimal image for Lambda
# Lambda base image provides the runtime interface client (RIC)
# ============================================================================
FROM public.ecr.aws/lambda/python:3.11

# Copy installed packages from builder
COPY --from=builder /build/packages ${LAMBDA_TASK_ROOT}

# Copy application code
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/ ${LAMBDA_TASK_ROOT}/models/
COPY data/baselines/ ${LAMBDA_TASK_ROOT}/data/baselines/
COPY data/reports/ ${LAMBDA_TASK_ROOT}/data/reports/
COPY model_card.json ${LAMBDA_TASK_ROOT}/

# Copy __init__ files to make packages importable
COPY src/__init__.py ${LAMBDA_TASK_ROOT}/src/__init__.py

# Environment defaults (overridden by Lambda environment variables)
ENV MODEL_PATH=models/rand_forest_v1.pkl \
    MODEL_VERSION=rand_forest_v1 \
    AWS_DEFAULT_REGION=us-east-1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build metadata (visible in Lambda console)
ARG GIT_SHA=unknown
LABEL git.sha="${GIT_SHA}" \
      project="car-valuation-api" \
      maintainer="salvo"

# Lambda handler: module.function
# Mangum wraps FastAPI as a Lambda-compatible ASGI handler
CMD ["src.api.app.handler"]