"""
Agent mode: the model decides what to search for (it may search several
times with its own queries). The sources shown to the user are the
passages the agent actually retrieved.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.nodes.nodes import LANGUAGE_RULES, NOT_FOUND, format_sources
from src.state.rag_state import RAGState

SYSTEM_PROMPT = f"""You answer questions about the user's documents.
- Use the search_documents tool, rephrasing or translating the query if the first search is not enough.
- Answer only from what the tool returns and cite the passages you use, like [1] or [2].
- If the documents do not contain the answer, reply only: "{NOT_FOUND['en']}" (in Arabic: "{NOT_FOUND['ar']}")."""


class AgentNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def answer(self, state: RAGState) -> dict:
        found = []

        @tool
        def search_documents(query: str) -> str:
            """Search the indexed documents (Arabic and English) and return numbered passages."""
            docs = self.retriever.invoke(query)
            new = [d for d in docs if d not in found]
            if not new:
                return "No new passages found."
            start = len(found) + 1  # keep numbering across searches so citations stay unique
            found.extend(new)
            return format_sources(new, start)

        agent = create_agent(self.llm, tools=[search_documents], system_prompt=SYSTEM_PROMPT)
        rule = LANGUAGE_RULES.get(state.answer_language, LANGUAGE_RULES["auto"])
        result = agent.invoke({"messages": [HumanMessage(content=f"{state.question}\n\n({rule})")]})
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else ""
        return {"answer": answer or "Could not generate an answer.", "retrieved_docs": found}
