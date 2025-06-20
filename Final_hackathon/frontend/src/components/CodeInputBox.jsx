// CodeInputBox.js
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';


const CodeInputBox = ({ onSubmit, disabled }) => {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('python');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const languages = [
    { value: 'python', label: 'Python' },
    { value: 'javascript', label: 'JavaScript' },
    { value: 'java', label: 'Java' },
    { value: 'cpp', label: 'C++' },
    { value: 'ruby', label: 'Ruby' }
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!code.trim() || isSubmitting) return;
    
    setIsSubmitting(true);
    try {
      await onSubmit({ code, language });
      setCode('');
    } catch (error) {
      console.error('Error submitting code:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const supportedLanguages = [
    { value: 'javascript', label: 'JavaScript' },
    { value: 'python', label: 'Python' },
    { value: 'java', label: 'Java' },
    { value: 'cpp', label: 'C++' },
    { value: 'ruby', label: 'Ruby' }
  ];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-800">Code Analysis</h2>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="px-4 py-2 border border-gray-200 rounded-md text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          disabled={disabled || isSubmitting}
        >
          {supportedLanguages.map(lang => (
            <option key={lang.value} value={lang.value}>
              {lang.label}
            </option>
          ))}
        </select>
      </div>

      <div className="relative">
        <SyntaxHighlighter
          language={language}
          style={oneDark}
          customStyle={{
            margin: 0,
            padding: '1rem',
            borderRadius: '0.5rem',
            minHeight: '200px'
          }}
          className="font-mono text-sm"
        >
          {code}
        </SyntaxHighlighter>
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          placeholder="Enter your code here..."
          className="absolute inset-0 w-full h-full font-mono text-sm bg-transparent text-transparent caret-white resize-none p-4 focus:outline-none"
          spellCheck="false"
          disabled={disabled || isSubmitting}
        />
      </div>

      <motion.button
        onClick={handleSubmit}
        disabled={isSubmitting || !code.trim() || disabled}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className={`w-full py-3 rounded-lg font-medium text-white
          ${isSubmitting || !code.trim() || disabled
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700'
          }
        `}
      >
        {isSubmitting ? (
          <div className="flex items-center justify-center space-x-2">
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            <span>Analyzing...</span>
          </div>
        ) : (
          'Analyze Code'
        )}
      </motion.button>
    </div>
  );
};

export default CodeInputBox;