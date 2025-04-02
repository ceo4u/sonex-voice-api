FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Pre-compile vocoder
RUN python precompile.py

EXPOSE 5000
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]