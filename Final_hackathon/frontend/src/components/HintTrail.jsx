import React from 'react';
import { motion } from 'framer-motion';

const HintTrail = ({ hints }) => {
  return (
    <div className="space-y-4">
      {hints.map((hint, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.2 }}
          className="flex items-start space-x-4"
        >
          <div className="flex-shrink-0 w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center text-white font-medium shadow-md">
            {index + 1}
          </div>
          <div className="flex-grow bg-white rounded-lg p-4 border border-gray-200 shadow-sm hover:shadow-md transition-shadow duration-200">
            <p className="text-gray-700">{hint}</p>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

export default HintTrail;