FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
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
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8000
ENV LOG_LEVEL=INFO

# Create necessary directories
RUN mkdir -p /app/logs

# Set permissions
RUN chmod -R 755 /app

# Expose the application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the Chainlit application with enhanced logging
CMD ["chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8000"]
