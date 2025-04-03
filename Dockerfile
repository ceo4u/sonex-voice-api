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

# Create a script to run the application
RUN echo '#!/bin/bash\n\
if [ -z "$PORT" ]; then\n\
  PORT=5000\n\
fi\n\
exec gunicorn api:app --config gunicorn.conf.py' > /app/start.sh \
    && chmod +x /app/start.sh

# Start with the script
CMD ["/app/start.sh"]
