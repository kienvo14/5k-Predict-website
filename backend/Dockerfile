# 5K Predictor — FastAPI backend for Google Cloud Run
# Build context is this backend/ folder (matches render.yaml rootDir: backend).
FROM python:3.12-slim

# Faster, cleaner Python in a container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first so this layer caches unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the training data the model is fit from at startup (features.csv)
COPY . .

# Cloud Run sends traffic to $PORT (default 8080). Bind to it.
# Single worker: the model is trained in-process at import time, and Cloud Run
# scales by adding instances, not workers.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
