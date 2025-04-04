#!/bin/bash
if [ -z "$PORT" ]; then
  PORT=5000
fi

# Verify models exist
echo "Checking models..."
ls -la saved_models/default/

# Preload the application to ensure models are loaded
echo "Preloading application..."
python load_models.py

# Start the server with configuration file
echo "Starting server..."
exec gunicorn api:app --config gunicorn.conf.py
