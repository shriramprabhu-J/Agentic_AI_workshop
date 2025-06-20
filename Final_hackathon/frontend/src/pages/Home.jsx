import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import CodeInputBox from '../components/CodeInputBox';
import LoadingOverlay from '../components/LoadingOverlay';

const Home = () => {
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleCodeSubmit = async (code, language) => {
    setIsSubmitting(true);
    setError(null);
    
    try {
      const response = await axios.post("http://localhost:8000/analyze-code", {
        learner_id: 'user123',
        code: code,  // ensure these variables have values
        language: language
    }, {
        headers: {
            'Content-Type': 'application/json'  // explicitly set headers
        }
    });
      
      // Navigate to analysis result page with the session ID
      navigate(`/analysis/${response.data.session_id}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze code');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-indigo-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold text-gray-900 mb-4"
          >
            Code Analysis Assistant
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-gray-600"
          >
            Get instant feedback on your code with AI-powered analysis
          </motion.p>
        </div>

        {/* Main Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white rounded-xl shadow-lg p-6 md:p-8"
        >
          <div className="mb-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-2">Submit Your Code</h2>
            <p className="text-gray-600">Paste your code below and get detailed analysis including syntax issues, logic flaws, and optimization suggestions.</p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600">{error}</p>
            </div>
          )}

          <CodeInputBox onSubmit={handleCodeSubmit} />
        </motion.div>

        {/* Features Section */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            {
              icon: '🔍',
              title: 'Syntax Analysis',
              description: 'Catch syntax errors and get instant fix suggestions'
            },
            {
              icon: '🧠',
              title: 'Logic Check',
              description: 'Identify logical flaws and potential bugs in your code'
            },
            {
              icon: '⚡',
              title: 'Optimization Tips',
              description: 'Get suggestions to improve your code performance'
            }
          ].map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + index * 0.1 }}
              className="bg-white rounded-lg p-6 shadow-md border border-gray-100"
            >
              <div className="text-3xl mb-4">{feature.icon}</div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {isSubmitting && <LoadingOverlay message="Analyzing your code..." />}
    </div>
  );
};

export default Home;