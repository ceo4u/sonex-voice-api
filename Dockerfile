# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 \
    libopenblas-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create model directory structure
RUN mkdir -p saved_models/default

# Download models during build 
RUN wget -O saved_models/default/encoder.pt https://media.githubusercontent.com/media/ceo4u/sonex-voice-api/main/saved_models/default/encoder.pt
RUN wget -O saved_models/default/synthesizer.pt https://media.githubusercontent.com/media/ceo4u/sonex-voice-api/main/saved_models/default/synthesizer.pt
RUN wget -O saved_models/default/vocoder.pt https://media.githubusercontent.com/media/ceo4u/sonex-voice-api/main/saved_models/default/vocoder.pt

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn==21.2.0

# Copy application code
COPY . .

# Runtime configuration
EXPOSE 5000
CMD ["gunicorn", "api:app", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--workers", "2", \
     "--worker-class", "sync"]