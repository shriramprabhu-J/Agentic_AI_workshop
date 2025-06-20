// AgentStatus.jsx
import React, { useState, useEffect } from "react";
import CodeInputBox from "./CodeInputBox";
import AnalysisResult from "../pages/AnalysisResult"; // Make sure this path is correct

const AgentStatus = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState(null);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [serverStatus, setServerStatus] = useState('checking'); // 'checking', 'online', 'offline'

  // Check backend server status on mount
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          setServerStatus('online');
        } else {
          setServerStatus('offline');
        }
      } catch (error) {
        setServerStatus('offline');
      }
    };

    checkServerStatus();
  }, []);

  const handleCodeSubmit = async ({ code, language }) => {
    try {
      setIsAnalyzing(true);
      setError(null);
      setAnalysisResults(null);
      
      // Check server status before sending
      if (serverStatus === 'offline') {
        throw new Error('Backend server is offline. Please start the server and try again.');
      }
      
      const response = await fetch('http://localhost:8000/api/analyze-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code,
          language,
          learner_id: 'user123'
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Analysis failed');
      }

      const data = await response.json();
      setAnalysisResults(data);
    } catch (error) {
      console.error('Analysis failed:', error);
      setError(error.message || 'Failed to analyze code. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    let interval;
    if (isAnalyzing) {
      interval = setInterval(() => {
        setLoadingProgress(prev => prev < 90 ? prev + 10 : prev);
      }, 1000);
    } else {
      setLoadingProgress(0);
    }
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-gray-900 pt-20">
      <div className="container mx-auto px-4 py-8 max-w-7xl relative">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-white mb-3">Code Analysis</h1>
          <p className="text-lg text-gray-300">Submit your code and get instant feedback</p>
          
          {/* Server status indicator */}
          <div className="mt-4 flex justify-center">
            <div className={`flex items-center px-4 py-2 rounded-full ${
              serverStatus === 'online' ? 'bg-green-500/20' : 
              serverStatus === 'offline' ? 'bg-red-500/20' : 'bg-yellow-500/20'
            }`}>
              <div className={`w-3 h-3 rounded-full mr-2 ${
                serverStatus === 'online' ? 'bg-green-500' : 
                serverStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
              }`} />
              <span className="text-sm">
                {serverStatus === 'online' ? 'Backend connected' : 
                 serverStatus === 'offline' ? 'Backend offline' : 'Checking backend...'}
              </span>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start relative">
          <div className="backdrop-blur-md bg-white/10 rounded-lg shadow-lg overflow-hidden border border-white/20">
            <div className="bg-white/20 p-4 border-b border-white/20">
              <h2 className="text-lg font-semibold text-white">Code Editor</h2>
            </div>
            <div className="p-4">
              <CodeInputBox 
                onSubmit={handleCodeSubmit} 
                disabled={isAnalyzing || serverStatus !== 'online'} 
              />
              
              {serverStatus === 'offline' && (
                <div className="mt-4 p-4 bg-red-500/10 rounded-md text-center">
                  <p className="text-red-300 text-sm">
                    Backend server is offline. Please ensure your server is running at http://localhost:8000
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="backdrop-blur-md bg-white/10 rounded-lg shadow-lg border border-white/20">
            <div className="bg-white/20 p-4 border-b border-white/20">
              <h2 className="text-lg font-semibold text-white">Analysis Results</h2>
            </div>
            <div className="p-4">
              {error ? (
                <div className="bg-red-500/10 backdrop-blur-sm border border-red-500/20 rounded-md p-4 text-center">
                  <p className="text-red-400 text-sm">{error}</p>
                  <button 
                    onClick={() => setError(null)}
                    className="mt-2 px-3 py-1 bg-red-500/20 text-red-400 rounded text-sm hover:bg-red-500/30 transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              ) : isAnalyzing ? (
                <div className="flex flex-col items-center justify-center py-8 space-y-4">
                  <div className="w-full max-w-xs bg-white/10 rounded-full h-2 overflow-hidden">
                    <div 
                      className="h-full bg-indigo-600 transition-all duration-500"
                      style={{ width: `${loadingProgress}%` }}
                    />
                  </div>
                  <p className="text-indigo-300 animate-pulse">Analyzing your code...</p>
                </div>
              ) : analysisResults ? (
                <AnalysisResult result={analysisResults} />
              ) : (
                <div className="backdrop-blur-sm bg-white/5 rounded-md p-6 text-center border border-white/10">
                  <p className="text-gray-300 text-sm">Submit your code to see the analysis results</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentStatus;