import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import LoadingOverlay from './components/LoadingOverlay';

const Home = lazy(() => import('./pages/Home'));
const AnalysisResult = lazy(() => import('./pages/AnalysisResult'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const History = lazy(() => import('./components/History'));

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-white">
        <Navbar />
        
        <Suspense fallback={<LoadingOverlay message="Loading..." />}>
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/submit" element={<Home />} />
              <Route path="/analysis/:sessionId" element={<AnalysisResult />} />
              <Route path="/history" element={<History />} />
              <Route path="/dashboard" element={<Dashboard />} />
            </Routes>
          </main>
        </Suspense>
      </div>
    </BrowserRouter>
  );
}

export default App;
