from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.graph_builder.graph_builder import GraphBuilder
from src.nodes.nodes import NOT_FOUND, RAGNodes, build_prompt, detect_language, fuse
from tests.conftest import FakeRetriever, fake_llm


def doc(text, page=0, source="a.pdf"):
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_prompt_numbers_sources_with_pages(docs):
    prompt = build_prompt("How long is maternity leave?", docs)
    assert "[1] law.pdf, page 33\nMaternity leave is 60 days." in prompt
    assert "[2] attention.pdf, page 3" in prompt


def test_prompt_uses_one_language_chosen_in_code(docs):
    """The model used to drift into Arabic when told 'answer in the language of the question'."""
    english = build_prompt("How long is maternity leave?", docs)
    assert "Answer in English." in english and NOT_FOUND["en"] in english and NOT_FOUND["ar"] not in english
    arabic = build_prompt("ما مدة إجازة الوضع؟", docs)
    assert "Answer in Arabic." in arabic and NOT_FOUND["ar"] in arabic and NOT_FOUND["en"] not in arabic
    assert "Answer in Arabic." in build_prompt("How long is maternity leave?", docs, answer_language="ar")


def test_detect_language():
    assert detect_language("How many layers does BERT have?") == "en"
    assert detect_language("كم عدد طبقات نموذج BERT؟") == "ar"


def test_fusion_ranks_chunks_found_by_both_searches_first():
    a, b, c = doc("a"), doc("b"), doc("c")
    assert fuse([[a, b], [c, b]], k=3) == [b, a, c]
    assert fuse([[a, b], [c, b]], k=1) == [b]


def test_search_uses_the_question_and_its_translation():
    retriever = FakeRetriever([doc("x")])
    nodes = RAGNodes(retriever, fake_llm("ما مدة إجازة الوضع؟"), translate_query=True)
    assert nodes.search("How long is maternity leave?") == [doc("x")]
    assert retriever.queries == ["How long is maternity leave?", "ما مدة إجازة الوضع؟"]


def test_answer_language_does_not_change_the_search(docs):
    """The old app put an Arabic instruction in front of the question before searching."""
    retriever = FakeRetriever(docs)
    llm = fake_llm("مدة إجازة الوضع 60 يوماً [1]")
    result = GraphBuilder(retriever, llm, translate_query=False).run(
        "How long is maternity leave?", answer_language="ar")
    assert retriever.queries == ["How long is maternity leave?"]
    assert result["answer"] == "مدة إجازة الوضع 60 يوماً [1]"
    assert result["retrieved_docs"] == docs


def test_agent_shows_the_passages_it_used():
    first = [doc("A", 0, "a.pdf")]
    second = [doc("B", 4, "b.pdf")]

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
