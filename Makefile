.PHONY: install lint format test build run-local load-test clean deploy-infra baseline

# ─── Variables ────────────────────────────────────────────────────────────────
PYTHON      := python3.11
PORT        := 8000
IMAGE_NAME  := car-valuation-api
IMAGE_TAG   := latest

# ─── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

# ─── Code quality ─────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

type-check:
	mypy src/ --ignore-missing-imports

# ─── Testing ──────────────────────────────────────────────────────────────────

test:
	pytest tests/unit/ tests/integration/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=80 \
		-v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ─── Container ────────────────────────────────────────────────────────────────

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "Image size:"
	@docker image inspect $(IMAGE_NAME):$(IMAGE_TAG) --format='{{.Size}}' | \
		awk '{printf "%.0f MB\n", $$1/1024/1024}'

run-local:
	docker run --rm \
		-p $(PORT):8080 \
		-e MODEL_PATH=models/rand_forest_v1.pkl \
		-e MODEL_VERSION=rand_forest_v1 \
		-e AWS_DEFAULT_REGION=us-east-1 \
		-v $(PWD)/models:/var/task/models:ro \
		-v $(PWD)/data:/var/task/data:ro \
		$(IMAGE_NAME):$(IMAGE_TAG)

# ─── Load testing ─────────────────────────────────────────────────────────────

load-test:
	@if [ -z "$(URL)" ]; then \
		echo "Usage: make load-test URL=https://your-api-gateway-url"; \
		exit 1; \
	fi
	locust -f tests/load/locustfile.py \
		--headless \
		--users 50 \
		--spawn-rate 5 \
		--run-time 60s \
		--host $(URL) \
		--html tests/load/report.html \
		--csv tests/load/results

# ─── Data and model artefacts ─────────────────────────────────────────────────

baseline:
	$(PYTHON) scripts/compute_baseline.py
	@echo "Baseline written to data/baselines/"

bias-test:
	$(PYTHON) scripts/run_bias_test.py
	@echo "Bias report written to data/reports/bias_report.json"

model-card:
	$(PYTHON) scripts/generate_model_card.py
	@echo "Model card written to model_card.json"

onnx-export:
	$(PYTHON) scripts/export_onnx.py
	$(PYTHON) scripts/verify_onnx.py

benchmark:
	$(PYTHON) scripts/benchmark.py

# ─── Infrastructure ───────────────────────────────────────────────────────────

deploy-infra:
	bash infra/scripts/create_dynamodb.sh
	bash infra/scripts/create_s3_baseline.sh
	@echo "AWS infrastructure ready."

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache
	rm -f tests/load/report.html tests/load/results_*.csv

# ─── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo "Car Valuation API — Makefile targets"
	@echo ""
	@echo "  install        Install all dependencies"
	@echo "  lint           Run ruff linter"
	@echo "  format         Auto-format with ruff"
	@echo "  test           Run unit + integration tests with coverage"
	@echo "  build          Build Docker image"
	@echo "  run-local      Run container locally on :$(PORT)"
	@echo "  load-test      Run Locust: make load-test URL=https://..."
	@echo "  baseline       Compute PSI baseline from training data"
	@echo "  bias-test      Run bias test suite"
	@echo "  model-card     Generate model_card.json"
	@echo "  onnx-export    Export model to ONNX and verify"
	@echo "  benchmark      Compare pkl vs ONNX latency"
	@echo "  deploy-infra   Create DynamoDB tables and S3 bucket"
	@echo "  clean          Remove build artefacts"