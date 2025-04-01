import chainlit as cl
import os
import logging
import base64
from openai import OpenAI
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import time

# Configure detailed logging output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables with defaults
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-3-27b-it")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://vllm-server:5000/v1")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

logger.info(f"Connecting to vLLM server at: {INFERENCE_SERVER_URL}")
logger.info(f"Using model: {MODEL_ID}")

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

# Base system prompt that all agents will build upon
BASE_PROMPT_TEMPLATE = (
    "You are an expert assistant for the University of North Texas (UNT), powered by advanced AI technology. "
    "Your goal is to provide clear, confident, and structured answers to help UNT students, faculty, staff, and visitors. "
    "When responding to queries:\n\n"
    "### Must Follow Guidelines:\n"
    "1. **Focus on Specificity**:\n"
    "- Ensure that your answer directly addresses the user's query with relevant information.\n"
    "- If the question is about a specific topic (e.g., Computer Science graduate programs), respond with detailed information specific to that program.\n"
    "- Use contextual information to improve relevance and avoid generic responses.\n\n"
    "2. **Structured Answers**:\n"
    "- Always respond using bullet points to make your answer easy to read and understand.\n"
    "- Ensure each bullet point provides actionable, useful advice.\n\n"
    "3. **Do Not Be Vague**:\n"
    "- Avoid vague phrases like 'check the website for more information.' Instead, provide exact information and actionable next steps based on the user's intent.\n\n"
    "4. **Relevant URL Links**:\n"
    "- You MUST PROVIDE https URLs that are directly relevant to the user's query verbatim from the context. For example, if they ask about Computer Science, include links to Computer Science-related resources.\n\n"
    "5. **Anticipate User Intent**:\n"
    "- Assume that users are seeking specific, helpful information. Anticipate what they would likely ask next (e.g., application deadlines, course requirements, admission contact information) and provide guidance to help them complete their task.\n\n"
    "6. **If the question is related to time period or specific dates**:\n"
    "- Give the response corresponding to current year which is 2025 or if it talks about a specific year give response tailored to it.\n\n"
    "7. **Tone**:\n"
    "- Always maintain a confident, helpful, and direct tone. Users should feel that their questions have been fully addressed without confusion.\n"
    "- Avoid expressions of uncertainty such as 'I do not know,' 'it is unclear,' or anything similar.\n\n"
    "8. **Generic Questions**:\n"
    "- If a question is generic (e.g., 'Tell me about UNT'), provide detailed information about UNT's academic programs, student life, campus facilities, research opportunities, and any other relevant aspects."
)

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


class EmailComposeAgent(Agent):
    """Agent specialized in helping with email composition"""
    
    def __init__(self):
        super().__init__(
            name="Email Composer",
            description="Helps compose professional emails for academic settings"
        )
        self.required_inputs = [
            {"key": "recipient_type", "question": "Who are you writing to? (e.g., professor, advisor, administrator)"},
            {"key": "purpose", "question": "What is the main purpose of your email?"},
            {"key": "details", "question": "Any specific details to include in the email?"}
        ]
    
    def get_system_prompt(self) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE
        specialized_instructions = (
            "\n\nYou are now functioning as an Email Composition Assistant. "
            "Help the user draft professional emails for academic settings. "
            "Create emails that are concise, respectful, and clearly communicate the student's needs. "
            "Include all necessary components: greeting, introduction, body, request, gratitude, and signature. "
            "Ensure the tone is appropriate for academic correspondence."
        )
        
        # Add collected inputs to the prompt
        if self.collected_inputs:
            specialized_instructions += "\n\nUser has provided the following details:\n"
            for key, value in self.collected_inputs.items():
                specialized_instructions += f"- {key}: {value}\n"
            
            specialized_instructions += "\nBased on this information, draft a complete, professional email."
        
        return base_prompt + specialized_instructions


class ResearchPaperAgent(Agent):
    """Agent specialized in helping with research papers"""
    
    def __init__(self):
        super().__init__(
            name="Research Paper Assistant",
            description="Helps structure and develop academic research papers"
        )
        self.required_inputs = [
            {"key": "paper_topic", "question": "What is the main topic of your research paper?"},
            {"key": "academic_level", "question": "What is your academic level? (undergraduate, graduate, doctoral)"},
            {"key": "paper_length", "question": "What is the approximate length requirement for your paper?"}
        ]
    
    def get_system_prompt(self) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE
        specialized_instructions = (
            "\n\nYou are now functioning as a Research Paper Assistant. "
            "Help students plan, structure, and develop their academic research papers. "
            "Provide guidance on creating clear thesis statements, organizing sections, integrating sources, "
            "and developing strong arguments. Focus on academic writing conventions and research methodologies."
        )
        
        # Add collected inputs to the prompt
        if self.collected_inputs:
            specialized_instructions += "\n\nUser has provided the following details:\n"
            for key, value in self.collected_inputs.items():
                specialized_instructions += f"- {key}: {value}\n"
            
            specialized_instructions += "\nBased on this information, provide a structured outline and guidance."
        
        return base_prompt + specialized_instructions


class AcademicConceptsAgent(Agent):
    """Agent specialized in explaining academic concepts"""
    
    def __init__(self):
        super().__init__(
            name="Academic Concepts Guide",
            description="Explains complex academic concepts, theories, and works"
        )
        self.required_inputs = [
            {"key": "subject_area", "question": "What subject area are you interested in? (e.g., physics, literature, history)"},
            {"key": "concept", "question": "What specific concept, theory, book, or author would you like explained?"}
        ]
    
    def get_system_prompt(self) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE
        specialized_instructions = (
            "\n\nYou are now functioning as an Academic Concepts Guide. "
            "Provide clear, in-depth explanations of academic concepts, theories, books, and authors. "
            "Focus on explaining complex ideas in an accessible way while maintaining academic accuracy. "
            "Include key points, historical context, significance in the field, and connections to related concepts."
        )
        
        # Add collected inputs to the prompt
        if self.collected_inputs:
            specialized_instructions += "\n\nUser has provided the following details:\n"
            for key, value in self.collected_inputs.items():
                specialized_instructions += f"- {key}: {value}\n"
            
            specialized_instructions += "\nBased on this information, provide a comprehensive explanation."
        
        return base_prompt + specialized_instructions


class RedirectAgent(Agent):
    """Agent specialized in redirecting users to university resources"""
    
    def __init__(self):
        super().__init__(
            name="Resource Redirector",
            description="Guides users to appropriate University resources and websites"
        )
        self.required_inputs = [
            {"key": "resource_type", "question": "What type of resource are you looking for? (e.g., admissions, financial aid, department website)"},
            {"key": "specific_need", "question": "What specific information do you need from this resource?"}
        ]
    
    def get_system_prompt(self) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE
        specialized_instructions = (
            "\n\nYou are now functioning as a University Resource Guide. "
            "Your primary role is to direct users to the most relevant UNT resources and websites. "
            "Provide exact URLs to official UNT pages that address the user's specific needs. "
            "Explain what information they will find at each resource and why it's relevant to their query."
        )
        
        # Add collected inputs to the prompt
        if self.collected_inputs:
            specialized_instructions += "\n\nUser has provided the following details:\n"
            for key, value in self.collected_inputs.items():
                specialized_instructions += f"- {key}: {value}\n"
            
            specialized_instructions += "\nBased on this information, provide relevant UNT resources with URLs."
        
        return base_prompt + specialized_instructions


class GeneralAgent(Agent):
    """General purpose UNT assistant for queries that don't fit specialized categories"""
    
    def __init__(self):
        super().__init__(
            name="General UNT Assistant",
            description="Provides general information about University of North Texas"
        )
        # No required inputs for general agent
        self.required_inputs = []
    
    def get_system_prompt(self) -> str:
        # General agent just uses the base prompt
        return BASE_PROMPT_TEMPLATE


# Agent registry to store and manage our agents
agents = {
    "email": EmailComposeAgent(),
    "research": ResearchPaperAgent(),
    "academic": AcademicConceptsAgent(),
    "redirect": RedirectAgent(),
    "general": GeneralAgent()
}

# Session context to store the active agent for each user session
@cl.cache
def get_session_context():
    return {
        "active_agent": "general",  # Default to general agent
        "conversation_history": []
    }

def determine_agent_type(message: str) -> str:
    """Determine which agent type to use based on the message content"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["email", "compose", "write", "draft", "professor"]):
        return "email"
    elif any(word in message_lower for word in ["research", "paper", "thesis", "dissertation", "study", "outline"]):
        return "research"
    elif any(word in message_lower for word in ["concept", "explain", "theory", "book", "author", "academic"]):
        return "academic"
    elif any(word in message_lower for word in ["redirect", "website", "link", "resource", "url", "find"]):
        return "redirect"
    else:
        return "general"


@cl.set_starters
async def set_starters():
    """Define university-related starter suggestions for the welcome screen."""
    return [
        cl.Starter(
            label="Email to Professor",
            message="Help me compose a professional email to my professor requesting an extension for my term paper due to health issues.",
            icon="/public/icons/email.svg",
        ),
        cl.Starter(
            label="Research Paper Assistant",
            message="I need help structuring my research paper on climate change impacts. Can you provide a outline with sections I should include?",
            icon="/public/icons/research.svg",
        ),
        cl.Starter(
            label="Academic related concepts",
            message="Explain the concept of quantum mechanics and its fundamental principles.",
            icon="/public/icons/academic.svg",
        ),
        cl.Starter(
            label="Redirect me to ?",
            message="Where can I find information about graduate admissions requirements for the Computer Science department?",
            icon="/public/icons/url.svg",
        )
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize the session and display welcome message"""
    pass


@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handle incoming user messages by:
    1. Determining the appropriate agent
    2. Collecting necessary inputs
    3. Generating responses
    """
    user_input = message.content.strip()
    logger.info(f"Received user input: {user_input}")
    
    # Get the session context and current active agent
    context = get_session_context()
    current_agent_type = context["active_agent"]
    current_agent = agents[current_agent_type]
    
    # Check if this is a new conversation direction
    # Only switch agent if we're not in the middle of collecting inputs
    if not current_agent.waiting_for_input:
        detected_agent_type = determine_agent_type(user_input)
        if detected_agent_type != current_agent_type:
            # Reset the previous agent before switching
            current_agent.reset()
            
            # Switch to the newly detected agent
            context["active_agent"] = detected_agent_type
            current_agent_type = detected_agent_type
            current_agent = agents[current_agent_type]
            logger.info(f"Switched to {current_agent.name} agent")
    
    # Add message to conversation history
    context["conversation_history"].append({"role": "user", "content": user_input})
    
    # Check for attachments
    attachments = getattr(message, "attachments", None)
    
    # Process the input with the current agent
    result = current_agent.process_input(user_input)
    
    if result["type"] == "input_request":
        # Agent needs more information, ask a follow-up question
        await cl.Message(content=result["next_question"]).send()
        # Add this system question to conversation history
        context["conversation_history"].append({"role": "system", "content": result["next_question"]})
    else:
        # Agent has all it needs, generate a response
        # Show thinking indicator
        msg = cl.Message(content="")
        await msg.send()
        
        # Generate response
        response = await current_agent.get_response(user_input, attachments)
        msg.content = response
        # Update the message with the response
        await msg.update()
        
        # Add response to conversation history
        context["conversation_history"].append({"role": "assistant", "content": response})


if __name__ == "__main__":
    cl.run()
