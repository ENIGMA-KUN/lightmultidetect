#!/bin/bash

# Setup script for DeepFake Detection Platform

# Display banner
echo "================================================================================"
echo "               DeepFake Detection Platform - Project Setup Tool                 "
echo "================================================================================"

# Check for requirements
echo "Checking requirements..."

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "All requirements satisfied!"

# Create project directory structure
echo -e "\nCreating project structure..."

# Main directories
mkdir -p deepfake-detection
cd deepfake-detection

# Create directories
mkdir -p uploads
mkdir -p visualizations
mkdir -p scripts
mkdir -p logs

# Create backend structure
mkdir -p backend/app/api/endpoints
mkdir -p backend/app/core
mkdir -p backend/app/db
mkdir -p backend/app/models/weights
mkdir -p backend/app/preprocessing
mkdir -p backend/app/schemas
mkdir -p backend/app/services
mkdir -p backend/app/tasks
mkdir -p backend/app/utils
mkdir -p backend/tests

# Create frontend structure
mkdir -p frontend/public/assets
mkdir -p frontend/src/components/common
mkdir -p frontend/src/components/upload
mkdir -p frontend/src/components/analysis
mkdir -p frontend/src/components/results
mkdir -p frontend/src/contexts
mkdir -p frontend/src/hooks
mkdir -p frontend/src/pages
mkdir -p frontend/src/services
mkdir -p frontend/src/styles
mkdir -p frontend/src/types

echo "Project structure created successfully!"

# Copy configuration files
echo -e "\nSetting up configuration files..."

# Copy .env.example if it exists
if [ -f "../.env.example" ]; then
    cp "../.env.example" ".env.example"
    cp "../.env.example" ".env"
    echo "Created .env file from template"
fi

# Copy docker-compose.yml if it exists
if [ -f "../docker-compose.yml" ]; then
    cp "../docker-compose.yml" "docker-compose.yml"
    echo "Copied docker-compose.yml file"
fi

# Copy README.md if it exists
if [ -f "../README.md" ]; then
    cp "../README.md" "README.md"
    echo "Copied README.md file"
fi

# Setup backend files
if [ -d "../backend" ]; then
    echo -e "\nSetting up backend files..."
    cp -r "../backend/." "backend/"
    echo "Backend files copied successfully"
fi

# Setup frontend files
if [ -d "../frontend" ]; then
    echo -e "\nSetting up frontend files..."
    cp -r "../frontend/." "frontend/"
    echo "Frontend files copied successfully"
fi

# Setup utility scripts
if [ -d "../scripts" ]; then
    echo -e "\nSetting up utility scripts..."
    cp -r "../scripts/." "scripts/"
    echo "Utility scripts copied successfully"
fi

# Create Python virtual environment (optional)
echo -e "\nWould you like to create a Python virtual environment for development? (y/n)"
read create_venv

if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating Python virtual environment..."
    cd backend
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
    echo "Python virtual environment created and dependencies installed."
fi

# Download model weights (optional)
echo -e "\nWould you like to download model weights? This might take some time. (y/n)"
read download_weights

if [[ $download_weights == "y" || $download_weights == "Y" ]]; then
    echo "Setting up model weights..."
    mkdir -p backend/app/models/weights
    python scripts/download_weights.py
    echo "Model weights downloaded successfully."
else
    echo "Skipping model weights download. You'll need to download them later."
fi

# Build and start Docker services
echo -e "\nWould you like to build and start the Docker services now? (y/n)"
read start_docker

if [[ $start_docker == "y" || $start_docker == "Y" ]]; then
    echo "Building and starting Docker services..."
    docker-compose up -d
    echo "Docker services started successfully."
    
    echo -e "\nThe DeepFake Detection Platform is now running:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8000"
    echo "  - API Documentation: http://localhost:8000/docs"
else
    echo "Skipping Docker services startup."
    echo "You can start the services later with 'docker-compose up -d'"
fi

echo -e "\n================================================================================"
echo "                    Setup completed successfully!                          "
echo "================================================================================"