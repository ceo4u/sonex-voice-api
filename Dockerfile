# Use the official Python image with the desired version (3.10.12 in this example)
FROM python:3.10.12-slim

# Set a working directory inside the container
WORKDIR /app

# Copy your requirements file first so that Docker can cache the pip install step
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port 5000 (adjust if your app uses a different port)
EXPOSE 5000

# Define the command to run your API
CMD ["python", "api.py"]
