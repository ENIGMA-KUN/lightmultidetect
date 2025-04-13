#!/bin/bash

# Setup and run script for DeepFake Detection Platform

# Display banner
echo "================================================================================"
echo "                       DeepFake Detection Platform Setup                       "
echo "================================================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and Docker Compose first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p uploads
mkdir -p visualizations
mkdir -p backend/app/models/weights

# Download model weights (optional, as they're quite large)
read -p "Do you want to download model weights? This might take some time. (y/n): " download_weights
if [[ $download_weights == "y" ]]; then
    echo "Downloading model weights..."
    python scripts/download_weights.py
else
    echo "Skipping model weights download. You'll need to provide them manually."
fi

# Build and start Docker containers
echo "Building and starting Docker containers..."
docker-compose up --build -d

echo "================================================================================"
echo "                        Setup completed successfully!                          "
echo "================================================================================"
echo ""
echo "The DeepFake Detection Platform is now running:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - API Documentation: http://localhost:8000/docs"
echo ""
echo "To stop the application, run: docker-compose down"
echo "================================================================================" 