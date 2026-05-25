# Production-grade Dockerfile for Federated Learning Simulation
# Standardizes environment for TU9 evaluation and ML engineering reproduction

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for OpenCV and development utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging specifications first for efficient layer caching
COPY requirements.txt pyproject.toml /app/

# Install exact version locked python requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source directories and verification modules
COPY data/ /app/data/
COPY federated/ /app/federated/
COPY models/ /app/models/
COPY utils/ /app/utils/
COPY experiments/ /app/experiments/
COPY scripts/ /app/scripts/

# Create results and logs structure
RUN mkdir -p /app/results /app/logs

# Set PYTHONPATH environment variable to allow module resolution from root
ENV PYTHONPATH=/app

# Command to execute verification by default
CMD ["python", "verify_env.py"]
