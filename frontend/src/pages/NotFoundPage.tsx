import React from 'react';
import { Link } from 'react-router-dom';
import Button from '../components/common/Button';

const NotFoundPage: React.FC = () => {
    return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4">
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-8 max-w-md">
                <div className="mb-6">
                    <svg className="h-24 w-24 text-blue-500 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                </div>

                <h1 className="text-4xl font-bold text-white mb-4">404</h1>
                <h2 className="text-2xl font-bold text-white mb-4">Page Not Found</h2>

                <p className="text-gray-300 mb-6">
                    The page you are looking for might have been removed, had its name changed, or is temporarily unavailable.
                </p>

                <Link to="/">
                    <Button variant="primary">
                        Return to Home
                    </Button>
                </Link>
            </div>
        </div>
    );
};

export default NotFoundPage;