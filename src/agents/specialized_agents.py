from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from ..models.query_models import EmailQuery, QueryResponse, ResearchQuery, AcademicQuery, RedirectQuery
from ..config.prompts import EMAIL_AGENT_PROMPT, RESEARCH_AGENT_PROMPT, ACADEMIC_AGENT_PROMPT, REDIRECT_AGENT_PROMPT
from config.prompts import BASE_PROMPT_TEMPLATE

class EmailComposeAgent(BaseAgent):
    """Agent specialized in composing professional academic emails."""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = EMAIL_AGENT_PROMPT
    
    async def process_query(self, query: EmailQuery) -> QueryResponse:
        """
        Process an email composition query with step-by-step reasoning.
        
        Args:
            query: EmailQuery object containing query details
            
        Returns:
            QueryResponse object with the composed email and metadata
        """
        try:
            # Step 1: Analyze query components
            context = f"""
            Recipient Type: {query.recipient_type}
            Purpose: {query.purpose}
            Details: {query.details}
            Tone: {query.tone}
            """
            
            # Step 2: Generate email structure
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Please compose a professional email with the following context:\n{context}"}
            ]
            
            # Step 3: Get initial response
            response = await self.get_response(messages)
            
            # Step 4: Validate and refine the response
            if not response or len(response) < 50:  # Basic validation
                return QueryResponse(
                    success=False,
                    content="I apologize, but I couldn't generate a complete email. Please try again with more specific details.",
                    metadata={"error": "Incomplete response"},
                    error="Incomplete email generation"
                )
            
            # Step 5: Format the response
            formatted_response = self._format_email_response(response)
            
            return QueryResponse(
                success=True,
                content=formatted_response,
                metadata={
                    "recipient_type": query.recipient_type,
                    "purpose": query.purpose,
                    "tone": query.tone
                }
            )
            
        except Exception as e:
            return QueryResponse(
                success=False,
                content="I encountered an error while composing your email. Please try again.",
                metadata={"error": str(e)},
                error=str(e)
            )
    
    def _format_email_response(self, response: str) -> str:
        """Format the email response with proper structure and styling."""
        # Add markdown formatting for better readability
        formatted = f"""
### Composed Email

{response}

---
*Note: This is a template email. Please review and modify it according to your specific needs before sending.*
"""
        return formatted


class ResearchPaperAgent(BaseAgent):
    """Agent specialized in helping with research paper composition and analysis."""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = RESEARCH_AGENT_PROMPT
    
    async def process_query(self, query: ResearchQuery) -> QueryResponse:
        """
        Process a research paper query with step-by-step reasoning.
        
        Args:
            query: ResearchQuery object containing query details
            
        Returns:
            QueryResponse object with the research paper guidance and metadata
        """
        try:
            # Step 1: Analyze query components
            context = f"""
            Paper Topic: {query.paper_topic}
            Academic Level: {query.academic_level}
            Paper Length: {query.paper_length}
            Requirements: {query.requirements}
            """
            
            # Step 2: Generate research paper structure
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Please provide guidance for writing a research paper with the following context:\n{context}"}
            ]
            
            # Step 3: Get initial response
            response = await self.get_response(messages)
            
            # Step 4: Validate and refine the response
            if not response or len(response) < 100:  # Basic validation
                return QueryResponse(
                    success=False,
                    content="I apologize, but I couldn't generate complete research paper guidance. Please try again with more specific details.",
                    metadata={"error": "Incomplete response"},
                    error="Incomplete research paper guidance"
                )
            
            # Step 5: Format the response
            formatted_response = self._format_research_response(response)
            
            return QueryResponse(
                success=True,
                content=formatted_response,
                metadata={
                    "paper_topic": query.paper_topic,
                    "academic_level": query.academic_level,
                    "paper_length": query.paper_length
                }
            )
            
        except Exception as e:
            return QueryResponse(
                success=False,
                content="I encountered an error while providing research paper guidance. Please try again.",
                metadata={"error": str(e)},
                error=str(e)
            )
    
    def _format_research_response(self, response: str) -> str:
        """Format the research paper response with proper structure and styling."""
        formatted = f"""
### Research Paper Guidance

{response}

---
*Note: This guidance is based on standard academic requirements. Please verify specific requirements with your instructor or department guidelines.*
"""
        return formatted


class AcademicConceptsAgent(BaseAgent):
    """Agent specialized in explaining academic concepts and theories."""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = ACADEMIC_AGENT_PROMPT
    
    async def process_query(self, query: AcademicQuery) -> QueryResponse:
        """
        Process an academic concept query with step-by-step reasoning.
        
        Args:
            query: AcademicQuery object containing query details
            
        Returns:
            QueryResponse object with the concept explanation and metadata
        """
        try:
            # Step 1: Analyze query components
            context = f"""
            Subject Area: {query.subject_area}
            Concept: {query.concept}
            Difficulty Level: {query.difficulty_level}
            Prerequisites: {query.prerequisites}
            """
            
            # Step 2: Generate concept explanation
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Please explain the following academic concept with the given context:\n{context}"}
            ]
            
            # Step 3: Get initial response
            response = await self.get_response(messages)
            
            # Step 4: Validate and refine the response
            if not response or len(response) < 50:  # Basic validation
                return QueryResponse(
                    success=False,
                    content="I apologize, but I couldn't generate a complete explanation. Please try again with more specific details.",
                    metadata={"error": "Incomplete response"},
                    error="Incomplete concept explanation"
                )
            
            # Step 5: Format the response
            formatted_response = self._format_concept_response(response)
            
            return QueryResponse(
                success=True,
                content=formatted_response,
                metadata={
                    "subject_area": query.subject_area,
                    "concept": query.concept,
                    "difficulty_level": query.difficulty_level
                }
            )
            
        except Exception as e:
            return QueryResponse(
                success=False,
                content="I encountered an error while explaining the concept. Please try again.",
                metadata={"error": str(e)},
                error=str(e)
            )
    
    def _format_concept_response(self, response: str) -> str:
        """Format the concept explanation with proper structure and styling."""
        formatted = f"""
### Concept Explanation

{response}

---
*Note: This explanation is tailored to your specified difficulty level. If you need more or less detail, please let me know.*
"""
        return formatted


class RedirectAgent(BaseAgent):
    """Agent specialized in redirecting users to appropriate UNT resources."""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = REDIRECT_AGENT_PROMPT
    
    async def process_query(self, query: RedirectQuery) -> QueryResponse:
        """
        Process a resource redirection query with step-by-step reasoning.
        
        Args:
            query: RedirectQuery object containing query details
            
        Returns:
            QueryResponse object with resource information and metadata
        """
        try:
            # Step 1: Analyze query components
            context = f"""
            Resource Type: {query.resource_type}
            Specific Need: {query.specific_need}
            Department: {query.department}
            Context: {query.context}
            """
            
            # Step 2: Generate resource information
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Please provide information about UNT resources with the following context:\n{context}"}
            ]
            
            # Step 3: Get initial response
            response = await self.get_response(messages)
            
            # Step 4: Validate and refine the response
            if not response or len(response) < 50:  # Basic validation
                return QueryResponse(
                    success=False,
                    content="I apologize, but I couldn't generate complete resource information. Please try again with more specific details.",
                    metadata={"error": "Incomplete response"},
                    error="Incomplete resource information"
                )
            
            # Step 5: Format the response
            formatted_response = self._format_redirect_response(response)
            
            return QueryResponse(
                success=True,
                content=formatted_response,
                metadata={
                    "resource_type": query.resource_type,
                    "department": query.department,
                    "specific_need": query.specific_need
                }
            )
            
        except Exception as e:
            return QueryResponse(
                success=False,
                content="I encountered an error while providing resource information. Please try again.",
                metadata={"error": str(e)},
                error=str(e)
            )
    
    def _format_redirect_response(self, response: str) -> str:
        """Format the resource information with proper structure and styling."""
        formatted = f"""
### UNT Resource Information

{response}

---
*Note: Please verify the availability and specific requirements of these resources by visiting the provided links or contacting the respective departments.*
"""
        return formatted


class GeneralAgent(Agent):
    """General purpose UNT assistant for queries that don't fit specialized categories"""
    
    def __init__(self):
        super().__init__(
            name="General UNT Assistant",
            description="Provides general information about University of North Texas"
        )
        self.required_inputs = []
    
    def get_system_prompt(self) -> str:
        return BASE_PROMPT_TEMPLATE 