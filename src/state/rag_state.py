from langchain_core.documents import Document
from pydantic import BaseModel, Field


class RAGState(BaseModel):
    question: str
    answer_language: str = "auto"  # "auto" (same as the question), "en" or "ar"
    retrieved_docs: list[Document] = Field(default_factory=list)
    answer: str = ""
