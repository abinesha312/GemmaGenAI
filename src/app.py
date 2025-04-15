import chainlit as cl
import logging
from config.settings import (
    CHAINLIT_HOST,
    CHAINLIT_PORT
)
from agents.registry import agents, determine_agent_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Session context to store the active agent for each user session
@cl.cache
def get_session_context():
    return {
        "active_agent": "general",  # Default to general agent
        "conversation_history": []
    }

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
    logger.info("New chat session started")
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
    logger.info("Starting Chainlit application")
    cl.run(host=CHAINLIT_HOST, port=CHAINLIT_PORT)