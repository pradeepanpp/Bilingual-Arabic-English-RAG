"""Settings, read from the environment (.env) where it makes sense."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "docs"        # PDFs indexed at startup
UPLOAD_DIR = DATA_DIR / "uploads"   # PDFs uploaded in the app
URL_FILE = DATA_DIR / "url.txt"     # web pages, one URL per line


def get_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain.chat_models import init_chat_model
    return init_chat_model(f"openai:{LLM_MODEL}", temperature=0)
