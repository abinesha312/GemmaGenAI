from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import base64
import time
from openai import OpenAI
import os

from config.settings import (
    MODEL_ID,
    INFERENCE_SERVER_URL,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT
)

logger = logging.getLogger(__name__)

# Initialize OpenAI client with vLLM API endpoint
client = OpenAI(
    api_key="EMPTY",  # vLLM doesn't require an actual API key
    base_url=INFERENCE_SERVER_URL,
    timeout=REQUEST_TIMEOUT
)

# Helper function to encode image to base64
def encode_image_to_base64(image_data):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_data).decode('utf-8')

class Agent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.required_inputs = []
        self.collected_inputs = {}
        self.waiting_for_input = False
        self.current_input_key = None
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for this agent"""
        pass
    
    def needs_additional_input(self) -> bool:
        """Check if the agent needs more information from the user"""
        if not self.required_inputs:
            return False
            
        for input_item in self.required_inputs:
            if input_item["key"] not in self.collected_inputs:
                return True
        return False
    
    def process_input(self, message: str) -> Dict[str, Any]:
        """Process the user input and decide what to do next"""
        result = {"type": "response", "content": None, "next_question": None}
        
        # If waiting for a specific input, collect it
        if self.waiting_for_input and self.current_input_key:
            self.collected_inputs[self.current_input_key] = message
            self.waiting_for_input = False
            self.current_input_key = None
        
        # Check if we need more inputs
        if self.needs_additional_input():
            # Find the next required input
            for input_item in self.required_inputs:
                if input_item["key"] not in self.collected_inputs:
                    self.waiting_for_input = True
                    self.current_input_key = input_item["key"]
                    result["type"] = "input_request"
                    result["next_question"] = input_item["question"]
                    break
        else:
            # All inputs collected, ready for final response
            result["type"] = "final_response"
            
        return result
    
    async def get_response(self, user_message: str, attachments=None) -> str:
        """Get a response from the LLM using this agent's specialized prompt"""
        system_content = self.get_system_prompt()
        
        # Format the messages for OpenAI API format
        messages = [
            {"role": "system", "content": system_content}
        ]
        
        # Process attachments if any (for multimodal input)
        if attachments and len(attachments) > 0:
            try:
                # Create a list to hold content items (for multimodal input)
                user_content = []
                
                # Add text message
                if user_message:
                    user_content.append({"type": "text", "text": user_message})
                
                # Process each attachment
                for attachment in attachments:
                    # Get the image bytes
                    image_data = await attachment.get_bytes()
                    logger.info(f"Received image attachment of size: {len(image_data)} bytes")
                    
                    # Encode the image to base64
                    base64_image = encode_image_to_base64(image_data)
                    
                    # Add the image to content
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                
                # Add the user message with text and images
                messages.append({"role": "user", "content": user_content})
            
            except Exception as img_err:
                logger.exception("Error processing image attachment:")
                messages.append({"role": "user", "content": f"{user_message}\n[Image attachment error: {str(img_err)}]"})
        else:
            # Text-only message
            messages.append({"role": "user", "content": user_message})
        
        for attempt in range(MAX_RETRIES):
            try:
                # Make the inference request using the OpenAI client
                response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                
                # Extract and log the response content
                reply = response.choices[0].message.content
                logger.info(f"Inference response from {self.name} agent")
                return reply
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    error_message = (
                        "I apologize, but I'm having trouble connecting to the AI service at the moment. "
                        "This could be due to:\n"
                        "1. The AI service is not running\n"
                        "2. Network connectivity issues\n"
                        "3. Service configuration problems\n\n"
                        "Please try again in a few moments or contact support if the issue persists."
                    )
                    return error_message

    def reset(self):
        """Reset the agent state for a new conversation"""
        self.collected_inputs = {}
        self.waiting_for_input = False
        self.current_input_key = None 