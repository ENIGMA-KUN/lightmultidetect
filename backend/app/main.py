from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging

from app.api.router import api_router
from app.core.config import settings
from app.core.events import create_start_app_handler, create_stop_app_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for deepfake detection across images, audio, and video",
    debug=settings.DEBUG,
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Add event handlers
app.add_event_handler("startup", create_start_app_handler(app))
app.add_event_handler("shutdown", create_stop_app_handler(app))

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Mount static files for visualizations
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Import preprocessing modules to ensure they're registered
try:
    from app.preprocessing import image_preprocessing, audio_preprocessing, video_preprocessing
    logger.info("Preprocessing modules loaded successfully")
except ImportError as e:
    logger.warning(f"Error loading preprocessing modules: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the DeepFake Detection API",
        "documentation": "/docs",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server in {settings.ENVIRONMENT} mode")
    uvicorn.run(app, host="0.0.0.0", port=8000)