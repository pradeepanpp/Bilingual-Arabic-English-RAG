import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()


class Config:
    """Configuration class for RAG system"""

    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    
    LLM_MODEL = "openai:gpt-4o-mini"

    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50


    DATA_DIR = "data"
    DEFAULT_SOURCE_FILES = [
        os.path.join(DATA_DIR, "url.txt"),  
        DATA_DIR,  
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)