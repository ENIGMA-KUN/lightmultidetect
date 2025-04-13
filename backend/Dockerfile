FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY ./backend /app/backend

# Copy startup scripts
COPY ./main.py /app/
COPY ./start_worker.py /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_WEIGHTS_DIR=/app/backend/ml/models/weights
ENV UPLOAD_DIR=/app/uploads
ENV RESULTS_DIR=/app/results
ENV VISUALIZATIONS_DIR=/app/visualizations

# Create necessary directories
RUN mkdir -p $UPLOAD_DIR $RESULTS_DIR $VISUALIZATIONS_DIR $MODEL_WEIGHTS_DIR

# Create a non-root user and change ownership
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port for the FastAPI application
EXPOSE 8000

# Set the entrypoint command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]