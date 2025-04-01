FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_ID=google/gemma-3-27b-it
ENV INFERENCE_SERVER_URL=http://vllm-server:5000/v1
ENV MAX_RETRIES=3
ENV RETRY_DELAY=2
ENV REQUEST_TIMEOUT=30

# Expose the application port
EXPOSE 8000

# Run the Chainlit application
CMD ["chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8000"]
