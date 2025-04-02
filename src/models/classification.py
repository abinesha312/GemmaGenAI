from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AgentType(str, Enum):
    EMAIL = "email"
    RESEARCH = "research"
    ACADEMIC = "academic"
    REDIRECT = "redirect"
    GENERAL = "general"

class AgentProfile(BaseModel):
    """Profile defining an agent's characteristics and capabilities"""
    type: AgentType = Field(..., description="Type of agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of agent's capabilities")
    keywords: List[str] = Field(default=[], description="Keywords associated with this agent")
    prompt_template: str = Field(..., description="Template for agent's system prompt")
    capabilities: List[str] = Field(default=[], description="List of agent's capabilities")
    priority: int = Field(default=0, description="Priority level for classification (higher = more specific)")

class ClassificationResult(BaseModel):
    """Result of the classification process"""
    agent_type: AgentType = Field(..., description="Classified agent type")
    confidence_score: float = Field(..., description="Confidence score of classification")
    matched_keywords: List[str] = Field(default=[], description="Keywords that matched")
    alternative_agents: List[Dict[str, float]] = Field(default=[], description="Alternative agent matches with scores")

class PromptClassifier:
    """Classifier for determining the appropriate agent based on message content"""
    
    def __init__(self):
        self.agents = self._initialize_agents()
        self.vectorizer = TfidfVectorizer()
        self._initialize_vectorizer()
    
    def _initialize_agents(self) -> List[AgentProfile]:
        """Initialize agent profiles with their characteristics"""
        return [
            AgentProfile(
                type=AgentType.EMAIL,
                name="Email Composition Assistant",
                description="Helps compose professional emails for academic settings",
                keywords=[
                    "email", "compose", "draft", "write", "send", "correspondence",
                    "professor", "advisor", "administrator", "request", "inquiry"
                ],
                prompt_template="You are an email composition assistant...",
                capabilities=[
                    "Draft professional emails",
                    "Format academic correspondence",
                    "Handle various email scenarios",
                    "Maintain appropriate tone"
                ],
                priority=3
            ),
            AgentProfile(
                type=AgentType.RESEARCH,
                name="Research Paper Assistant",
                description="Helps with research paper composition and analysis",
                keywords=[
                    "research", "paper", "thesis", "dissertation", "academic writing",
                    "literature review", "methodology", "analysis", "citation"
                ],
                prompt_template="You are a research paper assistant...",
                capabilities=[
                    "Structure research papers",
                    "Guide methodology",
                    "Help with citations",
                    "Provide writing guidance"
                ],
                priority=3
            ),
            AgentProfile(
                type=AgentType.ACADEMIC,
                name="Academic Concepts Guide",
                description="Explains academic concepts and theories",
                keywords=[
                    "explain", "concept", "theory", "principle", "define",
                    "understand", "learn", "study", "academic", "subject"
                ],
                prompt_template="You are an academic concepts guide...",
                capabilities=[
                    "Explain complex concepts",
                    "Provide examples",
                    "Connect related ideas",
                    "Adjust difficulty level"
                ],
                priority=2
            ),
            AgentProfile(
                type=AgentType.REDIRECT,
                name="Resource Guide",
                description="Directs users to appropriate UNT resources",
                keywords=[
                    "where", "find", "resource", "website", "link",
                    "information", "help", "support", "service", "office"
                ],
                prompt_template="You are a resource guide...",
                capabilities=[
                    "Find relevant resources",
                    "Provide direct links",
                    "Explain resource usage",
                    "Guide to services"
                ],
                priority=2
            ),
            AgentProfile(
                type=AgentType.GENERAL,
                name="General UNT Assistant",
                description="Provides general information about UNT",
                keywords=[],
                prompt_template="You are a general UNT assistant...",
                capabilities=["Handle general queries"],
                priority=1
            )
        ]
    
    def _initialize_vectorizer(self):
        """Initialize the TF-IDF vectorizer with agent keywords"""
        all_keywords = []
        for agent in self.agents:
            all_keywords.extend(agent.keywords)
        self.vectorizer.fit([" ".join(all_keywords)])
    
    def classify_message(self, message: str) -> ClassificationResult:
        """
        Classify a message to determine the appropriate agent using cosine similarity.
        
        Args:
            message: The user's message to classify
            
        Returns:
            ClassificationResult with the best matching agent and confidence score
        """
        # Convert message to TF-IDF vector
        message_vector = self.vectorizer.transform([message])
        
        # Calculate similarities with each agent's keywords
        similarities = []
        for agent in self.agents:
            if not agent.keywords:
                continue
                
            agent_keywords = " ".join(agent.keywords)
            agent_vector = self.vectorizer.transform([agent_keywords])
            similarity = cosine_similarity(message_vector, agent_vector)[0][0]
            
            # Adjust similarity based on agent priority
            adjusted_similarity = similarity * (1 + (agent.priority * 0.1))
            
            similarities.append({
                "agent_type": agent.type,
                "score": adjusted_similarity,
                "keywords": [k for k in agent.keywords if k in message.lower()]
            })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        # Get the best match
        best_match = similarities[0]
        
        # Get alternative matches
        alternatives = [
            {"agent_type": s["agent_type"], "score": s["score"]}
            for s in similarities[1:3]  # Get next 2 best matches
        ]
        
        return ClassificationResult(
            agent_type=best_match["agent_type"],
            confidence_score=best_match["score"],
            matched_keywords=best_match["keywords"],
            alternative_agents=alternatives
        ) 