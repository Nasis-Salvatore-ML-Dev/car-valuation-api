FROM public.ecr.aws/lambda/python:3.11 AS builder

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir --target=/build/packages -r requirements.txt

FROM public.ecr.aws/lambda/python:3.11

COPY --from=builder /build/packages ${LAMBDA_TASK_ROOT}

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/ ${LAMBDA_TASK_ROOT}/models/
COPY model_card.json ${LAMBDA_TASK_ROOT}/

RUN mkdir -p ${LAMBDA_TASK_ROOT}/data/baselines \
             ${LAMBDA_TASK_ROOT}/data/reports \
             ${LAMBDA_TASK_ROOT}/data/training \
             ${LAMBDA_TASK_ROOT}/data/validation

COPY src/__init__.py ${LAMBDA_TASK_ROOT}/src/__init__.py

ENV MODEL_PATH=models/rand_forest_v1.pkl \
    MODEL_VERSION=rand_forest_v1 \
    AWS_DEFAULT_REGION=us-east-1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG GIT_SHA=unknown
LABEL git.sha="${GIT_SHA}" \
      project="car-valuation-api"

CMD ["src.api.app.handler"]
