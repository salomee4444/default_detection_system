# --- Runtime image ---
    FROM python:3.11-slim

    # System deps (optional but useful for scientific libs)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
      && rm -rf /var/lib/apt/lists/*
    
    # Workdir
    WORKDIR /app
    
    # Env (include PYTHONPATH so unpickling can import `spec_encoder`)
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH=/app \
        API_TOKEN=change-me \
        PRED_THRESHOLD=0.30
    
    # Install deps first (better layer caching)
    COPY requirements.txt ./
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt
    
    # Copy project
    COPY . .
    
    # Fail fast if model artifact missing
    RUN test -f artifacts/model_pipeline.joblib || (echo "Missing artifacts/model_pipeline.joblib. Train & commit it first." && exit 1)
    
    # Expose (Render will set $PORT; fallback to 8000 locally)
    EXPOSE 8000
    
    # Start server (bind to $PORT if present)
    CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]