#!/bin/bash

# Check if podman is installed
if ! command -v podman &> /dev/null; then
    echo "Podman is not installed. Please install it first."
    exit 1
fi

# Check for NVIDIA GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA SMI not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Create necessary directories
echo "Creating required directories..."
mkdir -p models/FAISS_INGEST/vectorstore
mkdir -p logs

# Clean up existing containers and pod
echo "Cleaning up existing containers and pod..."
podman pod rm -f app-pod || true
podman rmi -f localhost/vllm-server localhost/chainlit-app || true

# Create pod
echo "Creating pod for the application..."
podman pod create --name app-pod -p 5000:5000 -p 8000:8000

# Build images
echo "Building vLLM server image..."
DOCKER_BUILDKIT=1 podman build -t localhost/vllm-server -f Containerfile.vllm .

echo "Building Chainlit app image..."
DOCKER_BUILDKIT=1 podman build -t localhost/chainlit-app -f Containerfile.chainlit .

# Start vLLM server
echo "Starting vLLM server..."
podman --runtime /home/${USER}/l/bin/crun run --device nvidia.com/gpu=4,5,6,7 -d \
    --pod app-pod \
    --name vllm-server \
    --security-opt=label=disable \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -v "${PWD}/models:/home/models:Z" \
    localhost/vllm-server

# Wait for vLLM server to start
echo "Waiting for vLLM server to initialize..."
sleep 10

# Start Chainlit app
echo "Starting Chainlit app..."
podman run -d \
    --pod app-pod \
    --name chainlit-app \
    --security-opt=label=disable \
    -v "${PWD}/models:/home/models:Z" \
    -e VECTOR_DB_PATH=/home/models/FAISS_INGEST/vectorstore/db_faiss \
    localhost/chainlit-app

echo "Application started!"
echo "vLLM server is accessible at: http://localhost:5000"
echo "Chainlit UI is accessible at: http://localhost:8000"
echo ""
echo "To check container status:"
echo "  podman ps"
echo ""
echo "To view logs:"
echo "  podman logs -f vllm-server"
echo "  podman logs -f chainlit-app"
echo ""
echo "To stop the application:"
echo "  podman pod rm -f app-pod"