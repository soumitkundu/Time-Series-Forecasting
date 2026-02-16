import React from 'react';
import './StatusIndicator.css';

const StatusIndicator = ({ label, status, isReady }) => {
  const getStatusColor = () => {
    if (!status) return '#gray';
    if (status.success) return '#4caf50';
    return '#f44336';
  };

  const getStatusText = () => {
    if (!status) return 'Checking...';
    if (status.success) return 'OK';
    return 'Error';
  };

  return (
    <div className="status-indicator">
      <div className="status-label">{label}</div>
      <div className="status-content">
        <div 
          className="status-dot" 
          style={{ backgroundColor: getStatusColor() }}
        />
        <span className="status-text">{getStatusText()}</span>
      </div>
      {status && !status.success && status.error && (
        <div className="status-error">{status.error}</div>
      )}
    </div>
  );
};

export default StatusIndicator;
