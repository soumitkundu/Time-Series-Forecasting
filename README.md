# Time-Series-Forecasting on NVIDIA Stock

This is an implementation on different Time Series Models to predict the `close` value of NVIDIA Stock

### Project Architecture & Modular Structure:

```
nvidia-forecasting/
├── data/ # Versioned data (managed by DVC)
├── models/ # Saved .pkl or .h5 files
├── src/ # Core logic
│ ├── ingestion.py # Fetches data from yfinance
│ ├── features.py # Technical indicators (RSI, MACD)
│ ├── preprocess.py # Scaling and sequence creation
│ ├── train.py # Model training logic
│ └── pipeline.py # The Orchestrator
├── api/ # Deployment
│ └── main.py # FastAPI code
└── requirements.txt
```
