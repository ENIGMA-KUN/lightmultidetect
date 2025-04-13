# Multimodal Deepfake Detection Platform

A comprehensive platform for detecting deepfakes across multiple modalities: images, audio, and video.

## Features

- **Multimodal Detection**: Process and analyze images, audio, and video for deepfake content
- **AI-Powered Analysis**: Uses state-of-the-art deep learning models to detect manipulated media
- **Explainable Results**: Provides explanations and visualizations for detection decisions
- **Secure API**: Authentication and authorization with JWT tokens
- **Background Processing**: Processes media files asynchronously for better user experience
- **Modern UI**: Clean, responsive interface built with React

## System Architecture

The platform is built using a microservices architecture:

- **Frontend**: React-based web application
- **Backend API**: FastAPI for RESTful API endpoints
- **Workers**: Celery workers for background processing
- **Storage**: File-based storage for uploads and results
- **Cache/Queue**: Redis for task queue and caching

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for faster processing)
- At least 8GB RAM and 10GB disk space

## Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multimodal-deepfake-detection.git
   cd multimodal-deepfake-detection
   ```

2. Create and configure the environment file:
   ```
   cp .env.example .env
   # Edit .env file with your settings
   ```

3. Start the application using Docker Compose:
   ```
   docker-compose up -d
   ```

4. Access the web interface at http://localhost:3000

5. API documentation is available at http://localhost:8000/docs

## Project Structure

```
.
├── backend/                # Backend API code
│   ├── api/                # API routes and endpoints
│   ├── core/               # Core functionality and config
│   ├── ml/                 # Machine learning models and inference
│   │   ├── models/         # Model architecture definitions
│   │   ├── preprocessing/  # Data preprocessing modules
│   │   └── inference/      # Inference and prediction logic
│   ├── models/             # Pydantic data models
│   └── worker/             # Background task workers
├── frontend/               # React frontend application
├── uploads/                # Media file uploads (mounted volume)
├── results/                # Detection results (mounted volume)
├── visualizations/         # Explanation visualizations (mounted volume)
├── model_weights/          # Pre-trained model weights (mounted volume)
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # This file
```

## API Endpoints

- **POST /api/v1/detection/upload**: Upload media files for deepfake detection
- **GET /api/v1/detection/status/{task_id}**: Check detection task status
- **GET /api/v1/detection/result/{task_id}**: Get detection results
- **GET /api/v1/detection/visualization/{task_id}/{file_name}**: Get result visualization
- **POST /api/v1/users/token**: User authentication endpoint
- **GET /api/v1/health**: System health check

## Development

### Running Locally without Docker

1. Set up a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

2. Start Redis server locally

3. Run the backend:
   ```
   cd backend
   uvicorn main:app --reload
   ```

4. Start the Celery worker:
   ```
   python start_worker.py
   ```

5. Install frontend dependencies and start development server:
   ```
   cd frontend
   npm install
   npm start
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The deepfake detection models and techniques are based on recent research in this field
- Thanks to all open-source libraries and frameworks that made this project possible