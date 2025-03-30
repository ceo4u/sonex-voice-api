# Use the official Python 3.11.9 slim image as the base
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (build tools and ffmpeg for pydub)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel --root-user-action=ignore && \
    pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy the rest of the application code
COPY . .

# Expose port 5000 (Render will override this with its own PORT environment variable)
EXPOSE 5000

# Bind Gunicorn to the port provided by Render (default to 5000 if PORT is not set)
CMD ["sh", "-c", "gunicorn api:app --bind 0.0.0.0:${PORT:-5000}"]
