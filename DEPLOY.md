# Deploying the Time-Series Forecasting API

## Prerequisites

1. **Train models and produce artifacts** (once, or when retraining):

   ```bash
   python -m src.pipeline
   ```

   This creates:
   - `data/df_nvidia_stock_history.csv` (raw)
   - `data/df_nvidia_cleaned.csv` (cleaned)
   - `models/model_lstm.keras`
   - `models/scaler_lstm.joblib`
   - `models/model_xgb.joblib`
   - `models/metrics.json`

2. **Docker** (for containerized deployment): [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/).

---

## Local run (no Docker)

From the project root with the virtual environment activated:

```bash
uvicorn api.main:app --reload --host localhost --port 8000
```

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Readiness (models/data present): http://localhost:8000/ready

---

## Docker deployment

### Build and run with Docker Compose (recommended)

`data/` and `models/` are mounted from the host, so they must exist and contain the pipeline outputs before starting the API.

```bash
# 1. Ensure artifacts exist
python -m src.pipeline

# 2. Build and start the API
docker compose up --build
```

The API is available at http://localhost:8000.

### Run with plain Docker

```bash
# Build
docker build -t time-series-api .

# Run (mount current dir's data and models)
docker run -p 8000:8000 -v "%cd%\models:/app/models:ro" -v "%cd%\data:/app/data:ro" time-series-api
```

On Linux/macOS use `$(pwd)` instead of `%cd%`.

### Optional: bake data and models into the image

If you prefer to ship a self-contained image without volume mounts:

1. Remove or comment out the `data/` and `models/` lines in `.dockerignore`.
2. In the Dockerfile, add back:
   ```dockerfile
   COPY data/ ./data/
   COPY models/ ./models/
   ```
3. Run the pipeline, then build: `docker compose up --build` (you can drop the `volumes` section from `docker-compose.yml` for this image).

---

## Endpoints

| Method | Path                | Description                            |
| ------ | ------------------- | -------------------------------------- |
| GET    | `/health`           | Liveness probe                         |
| GET    | `/ready`            | Readiness (503 if models/data missing) |
| GET    | `/predict/next_day` | Next-day close prediction              |
| GET    | `/docs`             | Swagger UI                             |
| GET    | `/redoc`            | ReDoc                                  |

**Query parameters for `/predict/next_day`:**

- `model`: `lstm` or `xgboost` (default: `xgboost`).

Example:

```bash
curl "http://localhost:8000/predict/next_day?model=xgboost"
curl "http://localhost:8000/predict/next_day?model=lstm"
```

---

## Production notes

- **Workers**: The default `CMD` runs a single uvicorn process. For more throughput, use Gunicorn with uvicorn workers, e.g.:

  ```dockerfile
  RUN pip install gunicorn
  CMD ["gunicorn", "api.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000"]
  ```

- **Readiness**: Use `/ready` in Kubernetes or other orchestrators so traffic is sent only when models and data are present.

- **Secrets**: The API does not use API keys or auth by default. Add middleware or reverse-proxy auth if you expose it publicly.
