import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-3-27b-it")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://vllm-server:5000/v1")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
CHAINLIT_HOST = os.getenv("CHAINLIT_HOST", "0.0.0.0")
CHAINLIT_PORT = int(os.getenv("CHAINLIT_PORT", "8000"))

# Log configuration
logger.info(f"Connecting to vLLM server at: {INFERENCE_SERVER_URL}")
logger.info(f"Using model: {MODEL_ID}")

# Model configuration
MAX_TOKENS = 500
TEMPERATURE = 0.7 