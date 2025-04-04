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

# Copy the application code first (excluding models)
COPY . .

# Copy models last to ensure they override any existing models
COPY saved_models/default/encoder.pt saved_models/default/
COPY saved_models/default/synthesizer.pt saved_models/default/
COPY saved_models/default/vocoder.pt saved_models/default/

# Ensure models have correct permissions
RUN chmod 644 saved_models/default/*.pt

# List models to verify they exist
RUN ls -la saved_models/default/

# Expose port
EXPOSE 5000

# Create a script to run the application
RUN echo '#!/bin/bash\n\
if [ -z "$PORT" ]; then\n\
  PORT=5000\n\
fi\n\
\n\
# Verify models exist\n\
echo "Checking models..."\n\
ls -la saved_models/default/\n\
\n\
# Preload the application to ensure models are loaded\n\
echo "Preloading application..."\n\
python -c "import api; print(\'Models loaded: \', api.encoder is not None and api.synthesizer is not None and api.vocoder_model is not None)"\n\
\n\
# Start the server\n\
echo "Starting server..."\n\
exec gunicorn api:app --config gunicorn.conf.py' > /app/start.sh \
    && chmod +x /app/start.sh

# Start with the script
CMD ["/app/start.sh"]
