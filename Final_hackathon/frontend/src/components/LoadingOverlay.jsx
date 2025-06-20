import React from 'react';

const LoadingOverlay = ({ message = 'Loading...' }) => {
  return (
    <div className="loading-overlay">
      <div className="flex flex-col items-center space-y-4">
        <div className="loading-spinner" />
        <p className="text-lg font-medium text-gray-700 animate-pulse">{message}</p>
      </div>
    </div>
  );
};

export default LoadingOverlay;