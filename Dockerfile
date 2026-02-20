# syntax=docker/dockerfile:1

# Time-Series Forecasting API - production-grade multi-stage image

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


# Builder image: includes build tooling for scientific Python stack
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better layer caching
COPY requirements.txt .

# Create virtualenv and install all Python dependencies into it
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# Runtime image: minimal surface area, no build tools
FROM base AS runtime

# Use the virtualenv from the builder
ENV PATH="/opt/venv/bin:$PATH"

# Install only small runtime utilities (e.g. curl for container healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtualenv and application code
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser api/ ./api/

USER appuser

EXPOSE 8000

# Make host/port configurable via env vars for better environment isolation
ENV UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

CMD ["sh", "-c", "uvicorn api.main:app --host ${UVICORN_HOST} --port ${UVICORN_PORT}"]
