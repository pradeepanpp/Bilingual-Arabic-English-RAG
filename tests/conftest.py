"""Shared helpers. Everything runs offline: fake LLM, fake retriever, no model downloads."""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeLLM(GenericFakeChatModel):
    """Replies with the given messages in order; bind_tools is a no-op so agents accept it."""

    def bind_tools(self, tools, **kwargs):
        return self


def fake_llm(*replies):
    return FakeLLM(messages=iter([r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]))


class FakeRetriever:
    """Records every query and returns the same documents."""

    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.docs


def make_pdf(path, pages):
    """A PDF with one text page per string; an empty string makes a blank (image-like) page."""
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    doc.save(path)
    return path


@pytest.fixture
def docs():
    return [
        Document(page_content="Maternity leave is 60 days.", metadata={"source": "law.pdf", "page": 32}),
        Document(page_content="The encoder has 6 layers.", metadata={"source": "attention.pdf", "page": 2}),
    ]
