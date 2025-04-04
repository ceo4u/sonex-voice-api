#!/bin/bash
if [ -z "$PORT" ]; then
  PORT=5000
fi

# Verify models exist
echo "Checking models..."
ls -la saved_models/default/

# Preload the application to ensure models are loaded
echo "Preloading application..."
python -c "import api; print('Models loaded:', api.encoder is not None and api.synthesizer is not None and api.vocoder_model is not None)"

# Start the server
echo "Starting server..."
exec gunicorn api:app --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --preload
