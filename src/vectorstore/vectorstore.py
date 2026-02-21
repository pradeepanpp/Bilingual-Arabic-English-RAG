"""Vector store module for document embedding and retrieval"""

from __future__ import annotations
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class E5Embeddings(HuggingFaceEmbeddings):
    """
    Multilingual E5 expects:
      - docs:   'passage: ...'
      - query:  'query: ...'
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"query: {text}")


class VectorStore:
    """Manages vector store operations"""

    def __init__(self):
        """Initialize vector store with Multilingual E5 embeddings (local)"""
        self.embedding = E5Embeddings(
            model_name="intfloat/multilingual-e5-small",  
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents

        Args:
            documents: List of documents to embed
        """
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_retriever(self):
        """
        Get the retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)