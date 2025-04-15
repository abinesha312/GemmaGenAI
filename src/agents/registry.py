from .specialized_agents import (
    EmailComposeAgent,
    ResearchPaperAgent,
    AcademicConceptsAgent,
    RedirectAgent,
    GeneralAgent
)
from models.classification import PromptClassifier, AgentType

# Initialize the classifier
classifier = PromptClassifier()

# Initialize all agents
agents = {
    AgentType.EMAIL: EmailComposeAgent(),
    AgentType.RESEARCH: ResearchPaperAgent(),
    AgentType.ACADEMIC: AcademicConceptsAgent(),
    AgentType.REDIRECT: RedirectAgent(),
    AgentType.GENERAL: GeneralAgent()
}

def determine_agent_type(message: str) -> str:
    """
    Determine which agent should handle the message using the classification system.
    
    Args:
        message: The user's message to classify
        
    Returns:
        The type of agent that should handle the message
    """
    # Get classification result
    result = classifier.classify_message(message)
    
    # Log classification details
    print(f"Classified as: {result.agent_type}")
    print(f"Confidence score: {result.confidence_score}")
    print(f"Matched keywords: {result.matched_keywords}")
    print(f"Alternative agents: {result.alternative_agents}")
    
    # Return the agent type
    return result.agent_type 