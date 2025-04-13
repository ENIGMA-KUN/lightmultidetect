import React from 'react';
import { Link } from 'react-router-dom';
import Button from '../components/common/Button';

const HomePage: React.FC = () => {
    return (
        <div className="flex flex-col space-y-12">
            {/* Hero Section */}
            <section className="text-center py-16 px-4 relative overflow-hidden">
                {/* Background Grid Effect */}
                <div className="absolute inset-0 grid-bg z-0"></div>

                <div className="relative z-10">
                    <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
                            DeepFake Detection Platform
                        </span>
                    </h1>

                    <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
                        A comprehensive platform that detects manipulated content across images, audio, and video
                        using state-of-the-art AI technology.
                    </p>

                    <div className="flex flex-col sm:flex-row justify-center gap-4">
                        <Link to="/detect">
                            <Button variant="primary" size="lg">
                                Start Detection
                            </Button>
                        </Link>

                        <Link to="/about">
                            <Button variant="secondary" size="lg">
                                Learn More
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-12">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold mb-4">Powerful Detection Capabilities</h2>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Our platform combines multiple specialized models to detect various types of manipulated content.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <FeatureCard
                        title="Image Analysis"
                        description="Detect face swaps, warping, and GAN-generated content in images using frequency analysis and artifact detection."
                        icon={
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                        }
                    />

                    <FeatureCard
                        title="Audio Analysis"
                        description="Identify synthetic speech and voice cloning through spectral analysis and voice characteristic verification."
                        icon={
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                            </svg>
                        }
                    />

                    <FeatureCard
                        title="Video Analysis"
                        description="Detect facial reenactments, lip-sync inconsistencies, and temporal artifacts across video frames."
                        icon={
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                        }
                    />
                </div>
            </section>

            {/* How It Works Section */}
            <section className="py-12 bg-gray-800 bg-opacity-50 rounded-xl">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold mb-4">How It Works</h2>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Our platform uses a sophisticated multi-stage process to detect manipulated content.
                    </p>
                </div>

                <div className="container mx-auto px-4">
                    <div className="flex flex-col md:flex-row items-start justify-center gap-8">
                        <StepCard
                            number={1}
                            title="Upload Content"
                            description="Upload the image, audio, or video you want to analyze. We support common formats and maintain your privacy."
                        />

                        <StepCard
                            number={2}
                            title="AI Analysis"
                            description="Our advanced AI models analyze the content using specialized techniques for each media type."
                        />

                        <StepCard
                            number={3}
                            title="Review Results"
                            description="Get a detailed report with confidence scores, visualizations, and explanations of detected manipulations."
                        />
                    </div>
                </div>

                <div className="text-center mt-12">
                    <Link to="/detect">
                        <Button variant="primary" size="lg">
                            Try It Now
                        </Button>
                    </Link>
                </div>
            </section>

            {/* Technology Section */}
            <section className="py-12">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold mb-4">Powered by Advanced Technology</h2>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Our platform combines state-of-the-art models and techniques to provide accurate detection.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <TechCard
                        title="Image Models"
                        items={["XceptionNet", "EfficientNet", "Frequency Analysis"]}
                    />

                    <TechCard
                        title="Audio Models"
                        items={["Wav2Vec 2.0", "RawNet2", "Spectral Analysis"]}
                    />

                    <TechCard
                        title="Video Models"
                        items={["3D Convolutional Networks", "Two-Stream Networks", "Temporal Consistency"]}
                    />

                    <TechCard
                        title="Ensemble Techniques"
                        items={["Multi-Model Voting", "Confidence Calibration", "Cross-Modal Verification"]}
                    />
                </div>
            </section>
        </div>
    );
};

interface FeatureCardProps {
    title: string;
    description: string;
    icon: React.ReactNode;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ title, description, icon }) => {
    return (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 transition-all duration-300 hover:border-blue-500 hover:shadow-lg hover:shadow-blue-900/20">
            <div className="flex flex-col items-center text-center">
                <div className="mb-4">{icon}</div>
                <h3 className="text-xl font-bold mb-2">{title}</h3>
                <p className="text-gray-400">{description}</p>
            </div>
        </div>
    );
};

interface StepCardProps {
    number: number;
    title: string;
    description: string;
}

const StepCard: React.FC<StepCardProps> = ({ number, title, description }) => {
    return (
        <div className="flex-1 flex flex-col items-center text-center">
            <div className="w-16 h-16 rounded-full bg-blue-900 flex items-center justify-center text-2xl font-bold text-white mb-4">
                {number}
            </div>
            <h3 className="text-xl font-bold mb-2">{title}</h3>
            <p className="text-gray-400">{description}</p>
        </div>
    );
};

interface TechCardProps {
    title: string;
    items: string[];
}

const TechCard: React.FC<TechCardProps> = ({ title, items }) => {
    return (
        <div className="bg-gray-800 bg-opacity-50 rounded-xl p-6">
            <h3 className="text-xl font-bold mb-4">{title}</h3>
            <ul className="space-y-2">
                {items.map((item, index) => (
                    <li key={index} className="flex items-center">
                        <svg className="h-5 w-5 text-blue-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="text-gray-300">{item}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default HomePage;