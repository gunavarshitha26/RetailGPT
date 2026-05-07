from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import settings
import os

def get_llm():
    """Returns the configured LLM. Gracefully degrades if no API key is provided."""
    if settings.is_groq_configured:
        return ChatGroq(temperature=0.2, model_name=settings.LLM_MODEL, groq_api_key=settings.GROQ_API_KEY)
    
    # Can extend with Azure OpenAI logic here in the future
    return None
