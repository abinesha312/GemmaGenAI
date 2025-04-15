from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AgentType(str, Enum):
    """Enum for different types of agents"""
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

class AlternativeAgent(BaseModel):
    """Model for alternative agent suggestions"""
    agent_type: AgentType
    confidence_score: float

class ClassificationResult(BaseModel):
    """Model for classification results"""
    agent_type: AgentType
    confidence_score: float
    matched_keywords: List[str] = Field(default_factory=list)
    alternative_agents: List[AlternativeAgent] = Field(default_factory=list)

class PromptClassifier:
    """Classifier for determining which agent should handle a prompt"""
    
    def __init__(self):
        # Initialize keyword dictionaries for each agent type
        self.keywords = {
            AgentType.EMAIL: [
                "email", "compose", "write", "draft", "send", "message",
                "professor", "instructor", "faculty", "reply", "respond",
                "extension", "request", "meeting", "appointment"
            ],
            AgentType.RESEARCH: [
                "research", "paper", "thesis", "dissertation", "study",
                "methodology", "analysis", "literature", "review", "citation",
                "reference", "bibliography", "data", "results", "findings"
            ],
            AgentType.ACADEMIC: [
                "explain", "concept", "theory", "definition", "understand",
                "learn", "topic", "subject", "course", "material", "example",
                "homework", "assignment", "problem", "solution"
            ],
            AgentType.REDIRECT: [
                "where", "find", "location", "website", "link", "resource",
                "information", "contact", "office", "department", "building",
                "service", "help", "support", "assistance"
            ],
            AgentType.GENERAL: [
                "unt", "university", "campus", "student", "program",
                "admission", "enrollment", "registration", "general",
                "information", "question", "help"
            ]
        }
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000
        )
        
        # Create corpus for training vectorizer
        corpus = []
        for keywords in self.keywords.values():
            corpus.append(" ".join(keywords))
        
        # Fit vectorizer on keyword corpus
        self.vectorizer.fit(corpus)
        
        # Create keyword vectors
        self.keyword_vectors = {}
        for agent_type, keywords in self.keywords.items():
            keyword_text = " ".join(keywords)
            vector = self.vectorizer.transform([keyword_text])
            self.keyword_vectors[agent_type] = vector
    
    def classify_message(self, message: str) -> ClassificationResult:
        """
        Classify a message to determine which agent should handle it.
        
        Args:
            message: The user's message to classify
            
        Returns:
            ClassificationResult containing the best matching agent and alternatives
        """
        # Transform message
        message_vector = self.vectorizer.transform([message])
        
        # Calculate similarities with each agent's keywords
        similarities = {}
        for agent_type, keyword_vector in self.keyword_vectors.items():
            similarity = cosine_similarity(message_vector, keyword_vector)[0][0]
            similarities[agent_type] = similarity
        
        # Sort by similarity score
        sorted_agents = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get best match and alternatives
        best_match = sorted_agents[0]
        alternatives = sorted_agents[1:3]  # Get next 2 best matches
        
        # Find matched keywords
        message_words = set(message.lower().split())
        matched_keywords = []
        for word in message_words:
            for keywords in self.keywords.values():
                if word in keywords:
                    matched_keywords.append(word)
        
        # Create alternative agent list
        alternative_agents = [
            AlternativeAgent(
                agent_type=agent_type,
                confidence_score=float(score)
            )
            for agent_type, score in alternatives
        ]
        
        return ClassificationResult(
            agent_type=best_match[0],
            confidence_score=float(best_match[1]),
            matched_keywords=matched_keywords,
            alternative_agents=alternative_agents
        ) 