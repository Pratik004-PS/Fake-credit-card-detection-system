# ── Dockerfile (root level — used by Render) ────────────────────────────────
# Build: docker build -t fraud-detection-api .
# Run:   docker run -p 8000:8000 fraud-detection-api

FROM python:3.11-slim

# System dependencies (LightGBM / XGBoost need libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY config/   ./config/
COPY src/      ./src/
COPY api/      ./api/
COPY artifacts/ ./artifacts/

# Create logs directory (may be read-only on Render — logger handles gracefully)
RUN mkdir -p logs

# Render injects $PORT; default to 8000 for local Docker usage
ENV PORT=8000
ENV PYTHONPATH=/app

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
