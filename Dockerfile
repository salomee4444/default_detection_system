FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_TOKEN=change-me \
    PRED_THRESHOLD=0.30 \
    MODEL_URL=""

# Install deps first (cache-friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port (Render provides $PORT)
EXPOSE 8000

# On container start: fetch model if missing, then run API
CMD ["sh", "-c", "set -e; \
  mkdir -p artifacts; \
  if [ ! -f artifacts/model_pipeline.joblib ]; then \
    if [ -z \"$MODEL_URL\" ]; then \
      echo 'ERROR: MODEL_URL not set and artifacts/model_pipeline.joblib missing' >&2; exit 1; \
    fi; \
    echo 'Downloading model artifact...'; \
    curl -L \"$MODEL_URL\" -o artifacts/model_pipeline.joblib; \
  fi; \
  uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]