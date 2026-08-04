# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install curl (required for the health check command below)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

HEALTHCHECK --interval=60s --timeout=10s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]