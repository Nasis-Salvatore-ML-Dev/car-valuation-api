# Production Car Valuation API

A production-grade ML prediction service that estimates used BMW vehicle prices,
built with FastAPI, deployed on AWS Lambda, and automated end-to-end with
GitHub Actions CI/CD.

The system serves price predictions via REST API with per-request SHAP explainability,
PSI-based drift monitoring, immutable DynamoDB audit logging, bias testing on every
deployment, and a human override queue for low-confidence predictions.

## Live Endpoints

**Base URL:** `https://4mhpswg272.execute-api.eu-central-1.amazonaws.com`

> **Availability note:** Portfolio project on AWS free tier, live as of May 2026. If you are reading this significantly later, the endpoint may have been taken down to avoid post-free-tier costs. See [Local Development](#local-development) to run locally.

| Endpoint          | Method | Description                                |
| ----------------- | ------ | ------------------------------------------ |
| `/health`         | GET    | Liveness check + model version             |
| `/predict`        | POST   | Price prediction with confidence interval  |
| `/explain/{id}`   | GET    | SHAP waterfall for a logged prediction     |
| `/explain/global` | GET    | Global feature importance (mean SHAP)      |
| `/override`       | POST   | Flag prediction for human review           |
| `/drift`          | GET    | PSI drift report across monitored features |
| `/metrics`        | GET    | Bias report + model card                   |

### Example: predict a price

```bash
curl -s -X POST https://4mhpswg272.execute-api.eu-central-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "320d",
    "mileage": 120000,
    "engine_power": 184,
    "registration_date": "2015-03-01",
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "sold_at": "2020-06-15"
  }'
```

Response:

```json
{
  "prediction_id": "0c015def-f163-4a3c-855b-a3e1484bc3e6",
  "predicted_price": 11212.17,
  "confidence_interval": { "lower": 5577.17, "upper": 16847.17 },
  "confidence_score": 0.5,
  "shap_values": { "mileage": -0.42, "engine_power": 0.18, "...": "..." },
  "model_version": "rand_forest_v1",
  "processing_time_ms": 220.3,
  "flagged_for_review": true
}
```

## Architecture

```
GitHub push → CI (lint, test, type-check, Docker build)
           → CD (ECR push → staging Lambda → smoke test → bias test → model card → production Lambda → smoke test)

Runtime:
  Client → API Gateway → Lambda (FastAPI + Mangum) → DynamoDB audit log
                                                   → CloudWatch metrics
```

**Infrastructure:** AWS Lambda (containerised), API Gateway HTTP API, ECR, DynamoDB, CloudWatch.
Two isolated environments (staging + production) with promotion gated by smoke tests.

## Model

Random Forest regressor trained on 3,874 BMW transaction records with 38 engineered features.
Target variable is log(price); predictions are back-transformed to EUR.

| Metric | Value     |
| ------ | --------- |
| MAE    | 1,946 EUR |
| RMSE   | 2,875 EUR |
| R²     | 0.776     |
| MAPE   | 20.48%    |

Feature engineering includes BMW series extraction, luxury tier classification,
temporal features (registration year/quarter), interaction terms
(age × mileage, power/age ratio), and target encoding for categoricals.
Full details in `model_card.json`.

## MLOps Pipeline

### CI (every push)

Runs on Python 3.10 and 3.11 in parallel:

- Linting and formatting (ruff)
- Type checking (mypy)
- Unit tests with coverage (pytest)
- Integration tests with mocked AWS (moto)
- Docker build smoke test (import validation)

### CD (push to main)

Sequential deployment with safety gates:

1. Build Docker image and push to ECR
2. Deploy to staging Lambda
3. Smoke test staging (`/health` + `/predict` with assertions)
4. Run bias tests (MAE by segment, fail if any segment > 1.5x overall)
5. Generate model card (Mitchell et al. format)
6. Deploy to production Lambda
7. Smoke test production

### Monitoring

- **Drift detection:** PSI (Population Stability Index) computed over 4 features
  (mileage, engine power, car age, luxury tier) against training baseline.
  Thresholds: PSI < 0.1 stable, 0.1-0.2 monitor, > 0.2 action required.
- **Bias testing:** MAE broken down by vehicle segment (luxury vs economy, old vs new,
  high vs low mileage, fuel type). Segments with MAE > 1.5x overall are flagged.
  Runs automatically on every deployment.
- **Audit logging:** Every prediction is written to DynamoDB with inputs, outputs,
  SHAP values, confidence score, latency, and timestamp. Append-only, immutable.
- **CloudWatch:** Deployment failure metrics published automatically.
- **Human oversight:** Low-confidence predictions (score < 0.6) are auto-routed
  to an override queue. Manual flagging available via `/override` endpoint.

### Explainability

SHAP TreeExplainer provides per-feature contribution values for every prediction.
Global feature importance is pre-computed over a 100-row background dataset at
container startup. The `/explain/{id}` endpoint returns waterfall-chart-ready
data sorted by absolute SHAP contribution.

Note: SHAP is currently computed offline due to a C-extension ABI constraint in
the Lambda runtime. The architecture supports live per-request SHAP — the
constraint is documented as an engineering tradeoff.

### Model Serialisation

The model is exported to ONNX format alongside the sklearn pickle.
`scripts/export_onnx.py` performs conversion, graph validation,
numerical verification (< 0.1% relative difference), and latency benchmarking.

## Project Structure

```
.github/workflows/
  ci.yml                    # Lint, test, type-check, Docker build
  cd.yml                    # ECR → staging → smoke test → production
  load-test.yml             # Locust 50-user load test on staging
src/
  api/
    app.py                  # FastAPI application + Lambda handler
    preprocessing.py        # Feature engineering (38 features)
    schemas.py              # Pydantic request/response models
    middleware.py            # Request logging
  explainability/
    shap_explainer.py       # SHAP TreeExplainer wrapper
  monitoring/
    audit_logger.py         # DynamoDB audit log (append-only)
    bias_tester.py          # MAE by segment + flagging
    drift.py                # PSI drift detection + CloudWatch
  utils/
    model_loader.py         # Model bundle loader (local + S3)
scripts/
  compute_baseline.py       # PSI training baseline + SHAP background
  export_onnx.py            # ONNX export + verification + benchmark
  generate_model_card.py    # Auto-generated model card (Mitchell et al.)
  run_bias_test.py          # CD pipeline bias gate
tests/
  unit/                     # Unit tests
  integration/              # Integration tests (mocked AWS)
  load/locustfile.py        # Locust load test configuration
infra/scripts/
  create_dynamodb.sh        # DynamoDB table creation (idempotent)
models/
  rand_forest_v1.pkl        # Trained model bundle
data/
  baselines/                # PSI baseline + SHAP background
  training/                 # Training set
  validation/               # Validation set
```

## Local Development

```bash
python3 -m venv ~/.car-valuation
source ~/.car-valuation/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Run tests:

```bash
make lint          # ruff check + format
make test          # pytest unit + integration
make docker-build  # build container locally
```

Run the API locally:

```bash
MODEL_PATH=models/rand_forest_v1.pkl uvicorn src.api.app:app --reload
```

## Engineering Decisions

**Why Lambda over ECS/EC2:** The prediction workload is bursty (not sustained
throughput). Lambda's pay-per-request model and zero-ops scaling match the cost
profile. The 10-second cold start is acceptable for this use case.

**Why DynamoDB for audit logging:** Append-only writes at single-digit millisecond
latency. No relational queries needed — predictions are keyed by UUID. TTL on the
override queue provides automatic cleanup.

**Why PSI over KL divergence for drift:** PSI is symmetric, bounded, and has
well-established industry thresholds (0.1/0.2). KL divergence is asymmetric
and unbounded, making alerting harder to calibrate.

**Why SHAP is computed offline:** The SHAP C extension triggers a memory corruption
crash in the Lambda runtime due to an ABI mismatch between the compiled extension
and Lambda's glibc. The architecture supports live SHAP — the module, tests, and
endpoints exist — but the runtime constraint makes offline computation the
pragmatic choice. In a production setting, this would be resolved by pinning the
base Docker image to match the SHAP build environment.



# Attribution License 1.0

Copyright (c) 2026 Salvatore Nasisi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the **"Software"**), to use, study, copy, modify, merge, publish, distribute, and sublicense the Software, subject to the following conditions:

---

## 1. Attribution Required

All copies or substantial portions of the Software, including modified or derivative works, must retain:

- the original copyright notice,
- this license text,
- and clear attribution to the original author: **Salvatore Nasisi**.

---

## 2. No False Authorship Claims

You may not claim that the original Software was created entirely by you.

Modified versions must clearly indicate that changes were made and must not misrepresent the origin of the original work.

---

## 3. Redistribution Conditions

Any public redistribution of the Software, whether modified or unmodified, must include visible acknowledgment of the original author in:

- source code,
- documentation,
- or repository metadata.

### Example acknowledgment

> "Based on original work by Salvo."

---

## 4. Personal and Private Use

Private, personal, or internal use without redistribution does not require public attribution.

---

## 5. Commercial Use

Commercial use is permitted provided attribution requirements are preserved and authorship is not misrepresented.

---

## 6. Warranty Disclaimer

THE SOFTWARE IS PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 7. Termination

Any violation of this license automatically terminates the rights granted under it.

---

By using, copying, modifying, or distributing this Software, you agree to the terms of this license.
