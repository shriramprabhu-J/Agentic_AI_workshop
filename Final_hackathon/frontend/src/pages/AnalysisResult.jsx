import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import HintTrail from '../components/HintTrail';
import LoadingOverlay from '../components/LoadingOverlay';

const IssueCard = ({ title, issues, type }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getIcon = () => {
    switch (type) {
      case 'syntax':
        return '🔍';
      case 'logic':
        return '🧠';
      case 'optimization':
        return '⚡';
      default:
        return '📝';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm hover:shadow-md transition-all duration-300"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <span className="text-2xl">{getIcon()}</span>
          <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-gray-500 hover:text-gray-700 transition-colors"
        >
          {isExpanded ? '−' : '+'}
        </button>
      </div>
      {isExpanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="space-y-3"
        >
          {issues.map((issue, index) => (
            <div
              key={index}
              className="p-3 bg-gray-50 rounded-md border border-gray-200"
            >
              {type === 'syntax' && (
                <div className="text-gray-700">
                  <span className="text-indigo-600 font-medium">Line {issue.line}:</span> {issue.message}
                  {issue.fix_suggestion && (
                    <div className="mt-2 text-sm text-gray-600 bg-indigo-50 p-2 rounded">
                      <span className="font-medium">Suggestion:</span> {issue.fix_suggestion}
                    </div>
                  )}
                </div>
              )}
              {type === 'logic' && (
                <div className="text-gray-700">
                  <div className="font-medium text-indigo-600">{issue.context}</div>
                  <div className="mt-1">{issue.explanation}</div>
                </div>
              )}
              {type === 'optimization' && (
                <div className="text-gray-700">
                  <div className="font-medium text-indigo-600">Current Code:</div>
                  <pre className="mt-1 p-2 bg-gray-50 rounded border border-gray-200 font-mono text-sm">{issue.original_code}</pre>
                  <div className="font-medium text-indigo-600 mt-3">Optimized Version:</div>
                  <pre className="mt-1 p-2 bg-gray-50 rounded border border-gray-200 font-mono text-sm">{issue.optimized_code}</pre>
                  <div className="mt-2 text-sm bg-green-50 p-2 rounded">{issue.rationale}</div>
                </div>
              )}
            </div>
          ))}
        </motion.div>
      )}
    </motion.div>
  );
};

const AnalysisResult = () => {
  const { sessionId } = useParams();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAnalysisResult = async () => {
      try {
        const response = await axios.get(`/api/hints/${sessionId}`);
        setResult(response.data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    if (sessionId) {
      fetchAnalysisResult();
    }
  }, [sessionId]);

  if (loading) {
    return <LoadingOverlay message="Loading analysis results..." />;
  }

  if (error) {
    return (
      <div className="p-6 text-center">
        <div className="text-red-500 font-medium mb-2">Error</div>
        <div className="text-gray-600">{error}</div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="p-6 text-center">
        <div className="text-gray-600">No analysis results found</div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Analysis Results</h2>
      
      {result.syntax_issues?.length > 0 && (
        <IssueCard
          title="Syntax Issues"
          issues={result.syntax_issues}
          type="syntax"
        />
      )}
      {result.logic_flaws?.length > 0 && (
        <IssueCard
          title="Logic Flaws"
          issues={result.logic_flaws}
          type="logic"
        />
      )}
      {result.optimizations?.length > 0 && (
        <IssueCard
          title="Optimization Suggestions"
          issues={result.optimizations}
          type="optimization"
        />
      )}
      {result.hint_trail?.length > 0 && (
        <div className="mt-8">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Guided Hints</h3>
          <HintTrail hints={result.hint_trail.map(h => h.hint)} />
        </div>
      )}
      {result.final_fix && (
        <div className="mt-8 p-6 bg-white rounded-lg shadow-lg border border-gray-200">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Suggested Fix</h3>
          <pre className="bg-gray-50 p-4 rounded-md overflow-x-auto">
            <code className="text-sm text-gray-800">{result.final_fix}</code>
          </pre>
        </div>
      )}
    </div>
  );
};

export default AnalysisResult;