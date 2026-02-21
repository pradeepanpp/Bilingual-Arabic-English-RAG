"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState


class GraphBuilder:
    """Builds and manages the LangGraph workflow"""

    def __init__(self, retriever, llm, use_agent: bool = False, enable_wikipedia: bool = False):
        """
        Initialize graph builder

        Args:
            retriever: Document retriever instance
            llm: Language model instance
            use_agent: If True, use Agentic RAG node (reactnode.py)
            enable_wikipedia: If True, include Wikipedia tool in agent mode
        """
        self.retriever = retriever
        self.llm = llm
        self.use_agent = use_agent
        self.enable_wikipedia = enable_wikipedia

       
        if self.use_agent:
            from src.nodes.reactnode import RAGNodes  
            self.nodes = RAGNodes(retriever=self.retriever, llm=self.llm)
        else:
            from src.nodes.nodes import RAGNodes 
            self.nodes = RAGNodes(retriever=self.retriever, llm=self.llm)

        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph

        Returns:
            Compiled graph instance
        """
        builder = StateGraph(RAGState)

      
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

    
        builder.set_entry_point("retriever")

   
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

  
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        """
        Run the RAG workflow

        Args:
            question: User question

        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question, retrieved_docs=[], answer=None)
        return self.graph.invoke(initial_state)
