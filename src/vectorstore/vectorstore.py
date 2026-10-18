"""FAISS vector store with multilingual E5 embeddings."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 6


def e5_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    E5 is trained with "query: " and "passage: " prefixes; without them
    retrieval quality drops. Vectors are normalised, so FAISS L2 distance
    ranks the same way as cosine similarity.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    class E5Embeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts):
            return super().embed_documents([f"passage: {t}" for t in texts])

        def embed_query(self, text):
            return super().embed_query(f"query: {text}")

    return E5Embeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


class VectorStore:
    def __init__(self, embedding=None):
        self.embedding = embedding or e5_embeddings()
        self.store = None

    def add_documents(self, docs: list[Document]):
        if not docs:
            return
        if self.store is None:
            self.store = FAISS.from_documents(docs, self.embedding)
        else:
            self.store.add_documents(docs)

    def get_retriever(self, k: int = TOP_K):
        if self.store is None:
            raise ValueError("The vector store is empty. Add documents first.")
        return self.store.as_retriever(search_kwargs={"k": k})
