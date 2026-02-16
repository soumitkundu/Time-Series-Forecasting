import React from 'react';
import './PredictionCard.css';

const PredictionCard = ({ prediction }) => {
  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  const getModelDisplayName = (model) => {
    return model === 'lstm' ? 'LSTM' : 'XGBoost';
  };

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <h3>Prediction Result</h3>
        <span className="model-badge">{getModelDisplayName(prediction.model)}</span>
      </div>
      <div className="prediction-content">
        <div className="prediction-label">Next Day Close Price</div>
        <div className="prediction-value">{formatPrice(prediction.next_day_close)}</div>
      </div>
      <div className="prediction-footer">
        <small>Predicted using {getModelDisplayName(prediction.model)} model</small>
      </div>
    </div>
  );
};

export default PredictionCard;
