"""LangGraph nodes for RAG workflow + Agentic tool-use inside generate_answer"""

from __future__ import annotations
from typing import List, Optional

from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent


class RAGNodes:
    """Contains node functions for RAG workflow (Agentic Mode)"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  

        
        self.system_prompt = (
            "You are a helpful RAG agent. "
            "Use the 'retriever' tool to fetch passages from the indexed corpus. "
            "Answer concisely and ground your response in retrieved passages."
        )

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node (still used by the graph)"""
        docs = self.retriever.invoke(state.question)
        return RAGState(question=state.question, retrieved_docs=docs)

    def _build_tools(self) -> List[Tool]:
        """Build retriever tool only (Wikipedia removed)"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:6], start=1):
                meta = getattr(d, "metadata", {}) or {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                page = meta.get("page")
                suffix = f" (p.{page})" if page is not None else ""
                merged.append(f"[{i}] {title}{suffix}\n{d.page_content}")
            return "\n\n".join(merged)

        return [
            Tool(
                name="retriever",
                description="Fetch passages from the indexed corpus.",
                func=retriever_tool_fn,
            )
        ]

    def _build_agent(self):
        """Build agent once"""
        tools = self._build_tools()
      
        self._agent = create_agent(self.llm, tools=tools)

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using agentic tool use"""
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke(
            {
                "messages": [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=state.question),
                ]
            }
        )

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer = getattr(messages[-1], "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer.",
        )