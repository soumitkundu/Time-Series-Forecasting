import React, { useState, useEffect } from 'react';
import './App.css';
import { healthCheck, readinessCheck, predictNextDay } from './services/api';
import StatusIndicator from './components/StatusIndicator';
import PredictionCard from './components/PredictionCard';

function App() {
  const [healthStatus, setHealthStatus] = useState(null);
  const [readinessStatus, setReadinessStatus] = useState(null);
  const [selectedModel, setSelectedModel] = useState('xgboost');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkStatus = async () => {
    const health = await healthCheck();
    const ready = await readinessCheck();
    setHealthStatus(health);
    setReadinessStatus(ready);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    const result = await predictNextDay(selectedModel);
    
    if (result.success) {
      setPrediction(result.data);
    } else {
      setError(result.error || 'Failed to get prediction');
    }
    
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>📈 Time-Series Forecasting</h1>
        <p>Predict next-day stock prices using LSTM or XGBoost models</p>
      </header>

      <div className="status-container">
        <StatusIndicator 
          label="Health" 
          status={healthStatus} 
          isReady={healthStatus?.success}
        />
        <StatusIndicator 
          label="Readiness" 
          status={readinessStatus} 
          isReady={readinessStatus?.success}
        />
      </div>

      <div className="main-content">
        <div className="model-selector">
          <h2>Select Model</h2>
          <div className="model-buttons">
            <button
              className={`model-btn ${selectedModel === 'xgboost' ? 'active' : ''}`}
              onClick={() => setSelectedModel('xgboost')}
            >
              XGBoost
            </button>
            <button
              className={`model-btn ${selectedModel === 'lstm' ? 'active' : ''}`}
              onClick={() => setSelectedModel('lstm')}
            >
              LSTM
            </button>
          </div>
        </div>

        <div className="prediction-section">
          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={loading || !readinessStatus?.success}
          >
            {loading ? 'Predicting...' : 'Predict Next Day Close'}
          </button>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          {prediction && (
            <PredictionCard prediction={prediction} />
          )}
        </div>
      </div>

      <footer className="app-footer">
        <p>Time-Series Forecasting API - Powered by FastAPI & React</p>
      </footer>
    </div>
  );
}

export default App;
