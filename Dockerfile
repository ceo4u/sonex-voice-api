# Use the official Python 3.11.9 slim image as the base
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (essential build tools and audio libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    python3-dev \
    gcc \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with explicit Gunicorn installation
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==21.2.0

# Copy application code
COPY . .

# Expose port (Render will use $PORT environment variable)
EXPOSE 5000

# Production server command with timeout and worker configuration
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "4", "--worker-class", "sync"]