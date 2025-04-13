import React from 'react';
import { Link } from 'react-router-dom';
import Button from '../components/common/Button';

const AboutPage: React.FC = () => {
    return (
        <div className="max-w-4xl mx-auto">
            <h1 className="text-3xl font-bold mb-6">About DeepFake Detection Platform</h1>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
                <h2 className="text-2xl font-bold mb-4">Our Mission</h2>
                <p className="text-gray-300 mb-4">
                    Our mission is to provide reliable and accessible tools to help identify manipulated content in an era where
                    deepfakes and synthetic media are becoming increasingly sophisticated and prevalent.
                </p>
                <p className="text-gray-300">
                    By combining state-of-the-art AI technologies with intuitive visualizations, we aim to empower users to
                    make informed decisions about the content they encounter online.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h2 className="text-xl font-bold mb-4">Technology</h2>
                    <p className="text-gray-300 mb-4">
                        Our platform uses a combination of deep learning models and traditional analysis techniques
                        to detect inconsistencies and artifacts in manipulated media:
                    </p>
                    <ul className="list-disc pl-5 text-gray-300 space-y-1">
                        <li>Convolutional neural networks (CNNs) for image analysis</li>
                        <li>Specialized audio processing models</li>
                        <li>Temporal analysis for video content</li>
                        <li>Multi-modal fusion for comprehensive detection</li>
                    </ul>
                </div>

                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h2 className="text-xl font-bold mb-4">Capabilities</h2>
                    <p className="text-gray-300 mb-4">
                        Our platform can detect various types of manipulated content:
                    </p>
                    <ul className="list-disc pl-5 text-gray-300 space-y-1">
                        <li>Face swaps and GAN-generated faces</li>
                        <li>Synthetic voice and audio manipulation</li>
                        <li>Facial reenactment in videos</li>
                        <li>Inconsistent lip synchronization</li>
                        <li>Unusual artifacts and frequency patterns</li>
                    </ul>
                </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
                <h2 className="text-xl font-bold mb-4">Limitations</h2>
                <p className="text-gray-300 mb-4">
                    While our platform uses advanced technology to detect deepfakes, it's important to understand its limitations:
                </p>
                <ul className="list-disc pl-5 text-gray-300 space-y-2">
                    <li>
                        <strong>Not 100% accurate:</strong> No detection system is perfect. Results should be interpreted as probabilities rather than definitive judgments.
                    </li>
                    <li>
                        <strong>Evolving technology:</strong> As deepfake generation improves, detection becomes more challenging and requires ongoing updates.
                    </li>
                    <li>
                        <strong>Quality dependent:</strong> Low-quality media may lead to less reliable detection results.
                    </li>
                    <li>
                        <strong>Context matters:</strong> Technical analysis should be complemented by considering the source and context of the content.
                    </li>
                </ul>
            </div>

            <div className="text-center py-8">
                <h2 className="text-2xl font-bold mb-4">Ready to detect deepfakes?</h2>
                <Link to="/detect">
                    <Button variant="primary" size="lg">
                        Start Detection
                    </Button>
                </Link>
            </div>
        </div>
    );
};

export default AboutPage;