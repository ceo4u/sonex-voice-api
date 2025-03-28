# Use the official Python 3.11.9 slim image
FROM python:3.11.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && pip install --upgrade pip setuptools wheel \
    && pip install numpy==1.23.5 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y --purge build-essential gfortran \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL files (including config)
COPY . .

# Use Render's PORT variable
ENV PORT=5000
EXPOSE $PORT

# Start Gunicorn with config file
CMD ["gunicorn", "api:app", "--config", "gunicorn.conf.py", "--bind", "0.0.0.0:$PORT"]