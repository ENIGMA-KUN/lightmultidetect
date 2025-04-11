#!/bin/bash

# Deployment script for LightMultiDetect

# Check for required tools
command -v docker >/dev/null 2>&1 || { echo >&2 "Docker is required but not installed. Aborting."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo >&2 "Docker Compose is required but not installed. Aborting."; exit 1; }

# Set environment variables
export COMPOSE_PROJECT_NAME=lightmultidetect

# Stop any running containers
echo "Stopping any running containers..."
docker-compose down

# Build and start containers
echo "Building and starting containers..."
docker-compose up -d --build

# Check container status
echo "Checking container status..."
docker-compose ps

echo "Deployment completed successfully!"
echo "Frontend available at: http://localhost:3000"
echo "Backend API available at: http://localhost:8000/api" 