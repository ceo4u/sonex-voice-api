FROM python:3.9-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p saved_models/default output

# Copy models first
COPY saved_models/default/*.pt saved_models/default/

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Start with gunicorn
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:5000", "--timeout", "600", "--workers", "1", "--preload"]
