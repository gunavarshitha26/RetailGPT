"""
============================================
RetailGPT — Centralized Configuration
============================================
Loads all settings from .env file with sensible defaults.
Structured for easy swap from local (Groq) to Azure OpenAI.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application-wide settings loaded from environment variables."""

    # --- Database ---
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "retail.db")

    # --- LLM Provider ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

    # --- Azure OpenAI (for production) ---
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    # --- Data Paths ---
    RAW_DATA_DIR: str = "pipeline/raw"
    STAGED_DATA_DIR: str = "pipeline/staged"
    CURATED_DATA_DIR: str = "curated"
    DATA_STORE_DIR: str = "data_store"
    POWERBI_EXPORT_DIR: str = "powerbi_export"

    # --- PDF for RAG ---
    PLAYBOOK_PATH: str = "Superstore_Operations_Playbook.pdf"

    @property
    def is_groq_configured(self) -> bool:
        """Check if Groq API key is set and non-empty."""
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY != "your_groq_api_key_here")

    @property
    def is_azure_configured(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return bool(self.AZURE_OPENAI_API_KEY and self.AZURE_OPENAI_ENDPOINT)


# Singleton instance used across the application
settings = Settings()
