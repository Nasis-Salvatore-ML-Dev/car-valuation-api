# Car Valuation API — Project Runbook

This runbook documents every step taken to build and deploy the Car Valuation API.
Following these steps from scratch will reproduce the entire project.

## Prerequisites

- macOS with Python 3.11+
- VSCode
- GitHub account
- AWS account (free tier sufficient)
- AWS CLI v2 installed

---

## Phase 1: Local Environment and Repository Setup

### 1.1 Create virtual environment

```bash
python3.11 -m venv ~/.car-valuation
source ~/.car-valuation/bin/activate
pip install --upgrade pip
```

### 1.2 Create GitHub repository

- Name: `car-valuation-api`
- Public repo, no template
- Clone locally

### 1.3 Build directory structure

```
car-valuation-api/
├── .github/workflows/       # CI/CD pipelines
├── data/
│   ├── baselines/           # PSI baseline + SHAP background
│   ├── reports/             # Bias reports
│   ├── training/            # Training CSV
│   └── validation/          # Validation CSV
├── infra/scripts/           # AWS infrastructure scripts
├── models/                  # Trained model artifacts
├── scripts/                 # Pipeline scripts (bias test, ONNX export, etc.)
├── src/
│   ├── api/                 # FastAPI app, schemas, preprocessing
│   ├── explainability/      # SHAP explainer
│   ├── monitoring/          # Audit logger, bias tester, drift monitor
│   └── utils/               # Model loader
├── tests/
│   ├── unit/
│   ├── integration/
│   └── load/
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

### 1.4 Install dependencies

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Key packages: fastapi, mangum, scikit-learn==1.7.2, shap, onnx, onnxruntime,
skl2onnx, boto3, locust, ruff, mypy, pytest, moto.

### 1.5 Place model artifact

Copy the trained Random Forest bundle from Phase 1 into `models/rand_forest_v1.pkl`.
The bundle is a dict with keys: `model`, `encoders`, `target_encodings`.

Ensure the pickle was saved with the same scikit-learn version specified in
`requirements.txt`. If versions differ, re-pickle:

```bash
python3 -c "
import joblib, sklearn
print(sklearn.__version__)
bundle = joblib.load('models/rand_forest_v1.pkl')
joblib.dump(bundle, 'models/rand_forest_v1.pkl')
"
```

---

## Phase 2: AWS Infrastructure Setup

### 2.1 Create IAM deploy user

AWS Console → IAM → Users → Create user: `deploy-user`

Attach policies:

- AmazonDynamoDBFullAccess
- AmazonS3FullAccess
- AmazonEC2ContainerRegistryFullAccess
- AWSLambda_FullAccess
- CloudWatchFullAccess
- AmazonAPIGatewayAdministrator

Create access key (CLI use case). Save the key ID and secret.

### 2.2 Configure AWS CLI

```bash
aws configure set aws_access_key_id YOUR_KEY_ID
aws configure set aws_secret_access_key YOUR_SECRET_KEY
aws configure set default.region us-east-1
```

### 2.3 Create ECR repository

```bash
aws ecr create-repository --repository-name car-valuation-api --region us-east-1
```

### 2.4 Create IAM role for Lambda

```bash
aws iam create-role \
  --role-name car-valuation-lambda-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy --role-name car-valuation-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam attach-role-policy --role-name car-valuation-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name car-valuation-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchFullAccess
aws iam attach-role-policy --role-name car-valuation-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### 2.5 Create Lambda functions

```bash
ECR_URI=$(aws ecr describe-repositories --repository-name car-valuation-api \
  --query 'repositories[0].repositoryUri' --output text)
ROLE_ARN=$(aws iam get-role --role-name car-valuation-lambda-role \
  --query 'Role.Arn' --output text)

sleep 10  # Wait for role propagation

# Staging
aws lambda create-function \
  --function-name car-valuation-api-staging \
  --package-type Image \
  --code ImageUri=$ECR_URI:latest \
  --role $ROLE_ARN \
  --memory-size 2048 \
  --timeout 300

# Production
aws lambda create-function \
  --function-name car-valuation-api \
  --package-type Image \
  --code ImageUri=$ECR_URI:latest \
  --role $ROLE_ARN \
  --memory-size 2048 \
  --timeout 300
```

### 2.6 Create API Gateway (staging)

AWS Console → API Gateway → Create API → HTTP API

1. Add integration: Lambda → `car-valuation-api-staging`
2. Route: `$default` (catch-all)
3. Stage: `$default`
4. Copy the invoke URL

### 2.7 Create API Gateway (production)

Repeat for production, integrating with `car-valuation-api`.

If the `$default` route is missing, create it:

```bash
# Get API ID
aws apigatewayv2 get-apis --query 'Items[?Name==`car-valuation-api-production`].ApiId' --output text

# Get integration ID
aws apigatewayv2 get-integrations --api-id <API_ID> --query 'Items[0].IntegrationId' --output text

# Create route
aws apigatewayv2 create-route \
  --api-id <API_ID> \
  --route-key '$default' \
  --target integrations/<INTEGRATION_ID>
```

If production returns "Internal Server Error", add Lambda invoke permission:

```bash
aws lambda add-permission \
  --function-name car-valuation-api \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:us-east-1:<ACCOUNT_ID>:<API_ID>/*"
```

### 2.8 Create DynamoDB tables

```bash
bash infra/scripts/create_dynamodb.sh
```

Creates `car-valuation-audit-log` and `car-valuation-override-queue` tables
with PAY_PER_REQUEST billing.

### 2.9 Add GitHub secrets

Repository → Settings → Secrets and variables → Actions:

| Secret                | Value                             |
| --------------------- | --------------------------------- |
| AWS_ACCESS_KEY_ID     | IAM deploy user key ID            |
| AWS_SECRET_ACCESS_KEY | IAM deploy user secret            |
| STAGING_API_URL       | Staging API Gateway invoke URL    |
| PROD_API_URL          | Production API Gateway invoke URL |

---

## Phase 3: Application Code

### 3.1 FastAPI application (`src/api/app.py`)

Core endpoints: `/health`, `/predict`, `/explain/{id}`, `/explain/global`,
`/override`, `/drift`, `/metrics`.

Deployed on Lambda via Mangum adapter. Lifespan initialises model bundle,
drift monitor, audit logger, and bias test suite once per container.

### 3.2 Feature engineering (`src/api/preprocessing.py`)

Transforms raw input (model key, mileage, engine power, dates, fuel, color, type)
into 38 features matching the training format:

- BMW series extraction and luxury tier classification
- Temporal features (registration year/month/quarter, seasonal flags)
- Interaction terms (age × mileage, mileage/power, annual mileage)
- Label encoding for categoricals
- Target encoding for fuel, color, car type

### 3.3 Model loader (`src/utils/model_loader.py`)

Loads model bundle from local path or S3. Supports dict format
(model + encoders + target encodings) and legacy bare model.

### 3.4 SHAP explainer (`src/explainability/shap_explainer.py`)

TreeExplainer with `tree_path_dependent` perturbation and `check_additivity=False`.
Pre-computes global feature importance over 100-row background set.

Note: SHAP is disabled in Lambda due to C-extension ABI incompatibility.
The code is fully functional and tested locally.

### 3.5 Monitoring modules

- `audit_logger.py`: DynamoDB PutItem for every prediction, append-only
- `bias_tester.py`: MAE by segment, flags if > 1.5x overall
- `drift.py`: PSI computation + CloudWatch metric publishing

### 3.6 Pydantic schemas (`src/api/schemas.py`)

Request/response validation with field descriptions and validators.
Fuel type and car type are validated against allowed sets.

---

## Phase 4: CI/CD Pipelines

### 4.1 CI pipeline (`.github/workflows/ci.yml`)

Triggered on every push to any branch. Runs on Python 3.10 and 3.11:

1. Install dependencies
2. Lint (ruff check)
3. Format check (ruff format --check)
4. Type check (mypy, non-blocking)
5. Unit tests (pytest with coverage)
6. Integration tests (moto-mocked AWS)
7. Docker build smoke test

### 4.2 CD pipeline (`.github/workflows/cd.yml`)

Triggered on push to main:

1. Build Docker image → push to ECR (commit SHA tag + latest)
2. Deploy to staging Lambda
3. Smoke test staging: `/health` returns 200, `/predict` returns valid response
4. Run bias test (`scripts/run_bias_test.py`)
5. Generate model card (`scripts/generate_model_card.py`)
6. Deploy to production Lambda
7. Smoke test production: `/health` returns 200

Concurrency group prevents simultaneous deployments.

### 4.3 Load test pipeline (`.github/workflows/load-test.yml`)

Triggered on push to staging branch:

1. Warm up staging endpoint (3 requests)
2. Run Locust: 50 users, 5/s spawn rate, 60s duration
3. Assert: p99 < 2000ms, error rate < 1%
4. Upload HTML report as artifact

---

## Phase 5: Pipeline Scripts

### 5.1 Compute training baseline

```bash
python scripts/compute_baseline.py
```

Reads `data/training/training_set.csv`, computes PSI bin edges and expected
proportions for monitored features, samples 100 rows for SHAP background.
Outputs: `data/baselines/training_baseline.json`, `data/baselines/shap_background.pkl`.

### 5.2 Export to ONNX

```bash
python scripts/export_onnx.py
```

Converts sklearn model to ONNX, validates graph, verifies numerical consistency
(< 0.1% relative difference), benchmarks inference latency.

### 5.3 Run bias test

```bash
PYTHONPATH=. MODEL_PATH=models/rand_forest_v1.pkl python scripts/run_bias_test.py
```

Loads model, computes MAE by segment, exits with code 1 if any segment flagged.

### 5.4 Generate model card

```bash
MODEL_PATH=models/rand_forest_v1.pkl python scripts/generate_model_card.py
```

Produces `model_card.json` following Mitchell et al. (2019) format. Includes
bias segments, performance metrics, intended use, ethical considerations.

---

## Troubleshooting

### Lambda returns 500 on startup

Check CloudWatch logs:

```bash
aws logs tail /aws/lambda/car-valuation-api-staging --since 5m
```

Common causes:

- scikit-learn version mismatch between pickle and runtime → pin version in requirements.txt
- SHAP C-extension crash → remove SHAP from Lambda startup path
- Missing DynamoDB tables → run `create_dynamodb.sh`

### Smoke test times out

Lambda cold start can exceed 50 seconds. Increase retry count in cd.yml or
bump Lambda timeout:

```bash
aws lambda update-function-configuration \
  --function-name car-valuation-api-staging \
  --timeout 300 --memory-size 2048
```

### API Gateway returns 404

Routes not configured. Create a `$default` catch-all route:

```bash
aws apigatewayv2 create-route --api-id <ID> --route-key '$default' --target integrations/<INT_ID>
```

### API Gateway returns "Internal Server Error" with no Lambda logs

Missing invoke permission:

```bash
aws lambda add-permission --function-name <NAME> \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:us-east-1:<ACCOUNT>:<API_ID>/*"
```

### Ruff format/lint failures in CI

```bash
ruff check --fix src/ tests/ scripts/
ruff format src/ tests/ scripts/
```

### sklearn version mismatch warnings

Re-pickle the model with the correct version:

```bash
pip install scikit-learn==1.7.2
python3 -c "
import joblib
bundle = joblib.load('models/rand_forest_v1.pkl')
joblib.dump(bundle, 'models/rand_forest_v1.pkl')
"
```

---

## Lessons Learned

1. **scikit-learn pickle compatibility is fragile.** A model pickled with 1.7.2 and
   loaded by 1.6.1 unpickles "successfully" but produces corrupted internal tree
   structures. SHAP's additivity check catches this. Pin the exact version everywhere.

2. **SHAP's C extension can crash the entire Lambda process.** `munmap_chunk(): invalid pointer`
   is a C-level segfault that cannot be caught by Python try/except. The fix is to
   isolate SHAP in a subprocess or disable it in constrained runtimes.

3. **API Gateway routes are not created automatically.** Creating an HTTP API with a
   Lambda integration does not guarantee a `$default` route exists. Always verify
   with `get-routes`.

4. **Lambda invoke permissions must be explicitly granted.** API Gateway needs
   `lambda:InvokeFunction` permission even if the Lambda role has broad policies.

5. **CI/CD pipelines surface issues that local testing misses.** The ruff formatting
   and lint checks caught trailing newlines, unused imports, and import ordering
   issues that worked fine locally but failed in CI.

6. **Graceful degradation is essential.** Making SHAP non-fatal in the lifespan
   (service starts without it) is better than making it a hard dependency that
   kills the entire deployment.
