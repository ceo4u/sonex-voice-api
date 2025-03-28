# Use the official Python 3.11.9 slim image as the base
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for building numpy and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel, and force-install numpy first
RUN pip install --upgrade pip setuptools wheel --root-user-action=ignore && \
    pip install numpy==1.23.5 --root-user-action=ignore

# Now install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy the rest of your application code into the container
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Define the command to run your API using gunicorn
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:5000"]
