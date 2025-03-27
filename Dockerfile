# Use the official Python 3.11.9 slim image as the base
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install build dependencies (including gcc and other essential tools)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install dependencies from requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Define the command to run your API using gunicorn
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:5000"]
