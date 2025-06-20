import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const HistoryItem = ({ item }) => (
  <div className="bg-white rounded-lg p-4 shadow-md border border-gray-100">
    <div className="flex justify-between items-start mb-2">
      <h4 className="font-medium text-gray-800">{item.query_type}</h4>
      <span className="text-sm text-gray-500">
        {new Date(item.timestamp).toLocaleString()}
      </span>
    </div>
    <p className="text-sm text-gray-600 mb-3 line-clamp-2">{item.query}</p>
    <div className="flex justify-between items-center text-sm">
      <span className={`px-2 py-1 rounded ${item.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
        {item.status}
      </span>
      <span className="text-gray-500">{item.processing_time}s</span>
    </div>
  </div>
);

const History = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch('/api/history');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
          throw new Error('Response is not JSON');
        }
        const data = await response.json();
        setHistory(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching history:', error);
        setLoading(false);
      }
    };

    if (isOpen) {
      fetchHistory();
    }
  }, [isOpen]);

  const drawerVariants = {
    open: {
      x: 0,
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 30
      }
    },
    closed: {
      x: '100%',
      transition: {
        type: 'spring',
        stiffness: 300,
        damping: 30
      }
    }
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed right-0 top-1/2 -translate-y-1/2 bg-indigo-600 text-white px-3 py-6 rounded-l-lg shadow-lg transform hover:scale-105 transition-transform z-50"
      >
        <span className="writing-mode-vertical">
          {isOpen ? 'Close History' : 'View History'}
        </span>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial="closed"
            animate="open"
            exit="closed"
            variants={drawerVariants}
            className="fixed right-0 top-0 h-full w-96 bg-white shadow-2xl z-40 overflow-hidden"
          >
            <div className="h-full flex flex-col">
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-800">Query History</h2>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {loading ? (
                  <div className="flex justify-center items-center h-full">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
                  </div>
                ) : history.length > 0 ? (
                  history.map((item) => (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <HistoryItem item={item} />
                    </motion.div>
                  ))
                ) : (
                  <div className="text-center text-gray-500 mt-8">
                    No history available
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsOpen(false)}
            className="fixed inset-0 bg-black/20 backdrop-blur-sm z-30"
          />
        )}
      </AnimatePresence>

      <style>{`
        .writing-mode-vertical {
          writing-mode: vertical-rl;
          text-orientation: mixed;
        }
      `}</style>
    </>
  );
};

export default History;