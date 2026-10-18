from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.graph_builder.graph_builder import GraphBuilder
from src.nodes.nodes import NOT_FOUND, build_prompt
from tests.conftest import FakeRetriever, fake_llm


def test_prompt_numbers_sources_with_pages(docs):
    prompt = build_prompt("How long is maternity leave?", docs, "ar")
    assert "[1] law.pdf, page 33\nMaternity leave is 60 days." in prompt
    assert "[2] attention.pdf, page 3" in prompt
    assert NOT_FOUND["en"] in prompt and NOT_FOUND["ar"] in prompt
    assert "Answer in Arabic." in prompt


def test_arabic_answer_mode_does_not_change_the_search(docs):
    """The old app put an Arabic instruction in front of the question before searching."""
    retriever = FakeRetriever(docs)
    llm = fake_llm("مدة إجازة الوضع 60 يوماً [1]")
    result = GraphBuilder(retriever, llm).run("How long is maternity leave?", answer_language="ar")
    assert retriever.queries == ["How long is maternity leave?"]
    assert result["answer"] == "مدة إجازة الوضع 60 يوماً [1]"
    assert result["retrieved_docs"] == docs


def test_agent_shows_the_passages_it_used():
    first = [Document(page_content="A", metadata={"source": "a.pdf", "page": 0})]
    second = [Document(page_content="B", metadata={"source": "b.pdf", "page": 4})]

    class TwoSearches(FakeRetriever):
        def invoke(self, query):
            self.queries.append(query)
            return first if len(self.queries) == 1 else first + second

    retriever = TwoSearches([])
    llm = fake_llm(
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "إجازة الوضع"}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "maternity"}, "id": "2"}]),
        "Answer from [1] and [2].",
    )
    result = GraphBuilder(retriever, llm, use_agent=True).run("How long is maternity leave?")
    assert retriever.queries == ["إجازة الوضع", "maternity"]
    assert result["retrieved_docs"] == first + second  # each passage once, in the order found
    assert result["answer"] == "Answer from [1] and [2]."
