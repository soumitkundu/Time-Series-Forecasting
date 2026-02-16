# Time-Series Forecasting API - production image
FROM python:3.13-slim

WORKDIR /app

# Install system deps if needed (optional; remove if not required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better layer caching
COPY requirements.txt .

# Install Python dependencies (use --no-cache-dir to keep image smaller)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (data/ and models/ are mounted at runtime or built separately)
COPY api/ ./api/

# Non-root user for security (optional)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Production: single worker; for more workers use gunicorn + uvicorn workers
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
