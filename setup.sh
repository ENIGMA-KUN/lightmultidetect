#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install

# Start Redis (if not already running)
# On Windows, you need to install Redis separately
# On Linux/Mac: redis-server &

# Start Celery worker (in a new terminal)
cd ../backend
celery -A tasks.detection_tasks worker --loglevel=info

# Start backend server (in a new terminal)
cd ../backend
python -m uvicorn main:app --reload

# Start frontend development server (in a new terminal)
cd ../frontend
npm start 