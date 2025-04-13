#!/usr/bin/env python
"""
Script to start the Celery worker for background tasks.

This script initializes and runs the Celery worker for processing
deepfake detection tasks in the background.
"""

import os
import argparse
import logging
from backend.worker import celery_app
from backend.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Start the Celery worker."""
    parser = argparse.ArgumentParser(description="Start the Celery worker for deepfake detection tasks")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--queues",
        type=str,
        default="celery",
        help="Comma-separated list of queues to consume from (default: celery)"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Ensure necessary directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    os.makedirs(settings.VISUALIZATIONS_DIR, exist_ok=True)
    os.makedirs(settings.MODEL_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.UPLOAD_DIR, "temp"), exist_ok=True)
    
    logger.info(f"Starting Celery worker with concurrency: {args.concurrency}")
    
    # Import tasks to ensure they're registered with Celery
    from backend.worker import tasks
    
    # Start the Celery worker
    celery_argv = [
        'worker',
        f'--concurrency={args.concurrency}',
        f'--queues={args.queues}',
        f'--loglevel={args.loglevel}',
        '--without-gossip',  # Disable gossip (not needed for a simple setup)
        '--without-mingle',  # Disable mingle (not needed for a simple setup)
    ]
    
    celery_app.worker_main(celery_argv)

if __name__ == "__main__":
    main() 