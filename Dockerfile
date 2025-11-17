# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency list first (better caching)
COPY requirements-docker.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the service
CMD ["python", "scripts/serve.py"]
