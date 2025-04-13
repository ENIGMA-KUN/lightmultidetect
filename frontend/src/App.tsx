import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/common/Navbar';
import Footer from './components/common/Footer';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import DetectionPage from './pages/DetectionPage';
import ResultPage from './pages/ResultPage';
import NotFoundPage from './pages/NotFoundPage';

import { DetectionProvider } from './contexts/DetectionContext';

const App: React.FC = () => {
    return (
        <DetectionProvider>
            <Router>
                <div className="flex flex-col min-h-screen bg-gray-900 text-white">
                    <Navbar />

                    <main className="flex-grow container mx-auto px-4 py-6">
                        <Routes>
                            <Route path="/" element={<HomePage />} />
                            <Route path="/about" element={<AboutPage />} />
                            <Route path="/detect" element={<DetectionPage />} />
                            <Route path="/result/:resultId" element={<ResultPage />} />
                            <Route path="*" element={<NotFoundPage />} />
                        </Routes>
                    </main>

                    <Footer />
                </div>
            </Router>
        </DetectionProvider>
    );
};

export default App;