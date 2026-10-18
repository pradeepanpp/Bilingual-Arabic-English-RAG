"""LangGraph workflow: retrieve -> answer, or a single agent node in agent mode."""

from langgraph.graph import END, StateGraph

from src.nodes.nodes import RAGNodes
from src.nodes.reactnode import AgentNodes
from src.state.rag_state import RAGState


class GraphBuilder:
    def __init__(self, retriever, llm, use_agent: bool = False, translate_query: bool = True):
        builder = StateGraph(RAGState)
        if use_agent:
            builder.add_node("agent", AgentNodes(retriever, llm).answer)
            builder.set_entry_point("agent")
            builder.add_edge("agent", END)
        else:
            nodes = RAGNodes(retriever, llm, translate_query=translate_query)
            builder.add_node("retriever", nodes.retrieve_docs)
            builder.add_node("responder", nodes.generate_answer)
            builder.set_entry_point("retriever")
            builder.add_edge("retriever", "responder")
            builder.add_edge("responder", END)
        self.graph = builder.compile()

    def run(self, question: str, answer_language: str = "auto") -> dict:
        return self.graph.invoke(RAGState(question=question, answer_language=answer_language))
