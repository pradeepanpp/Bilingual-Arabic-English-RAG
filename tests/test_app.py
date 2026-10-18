from pathlib import Path

import pytest

pytest.importorskip("streamlit")

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture(autouse=True)
def fresh_cache():
    """The app caches the index and the LLM; tests must not share them."""
    import streamlit as st
    st.cache_resource.clear()


def test_app_reports_missing_api_key(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("OPENAI_API_KEY", "")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert "OPENAI_API_KEY is not set" in app.error[0].value


def test_text_from_documents_is_escaped(monkeypatch):
    import streamlit_app

    shown = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda html, **kw: shown.append(html))
    streamlit_app.show_text('<img src=x onerror="alert(1)">\nline 2', rtl=True)
    assert "<img" not in shown[0] and "&lt;img" in shown[0]
    assert "dir='rtl'" in shown[0] and "<br>line 2" in shown[0]


def test_ask_a_question_end_to_end(tmp_path, monkeypatch):
    """The real app with a fake LLM and fake embeddings: answer and sources with page numbers are shown."""
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from streamlit.testing.v1 import AppTest

    from src.config import config
    from src.vectorstore import vectorstore
    from tests.conftest import fake_llm, make_pdf

    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "leave.pdf", ["Maternity leave is sixty days: 45 days on full pay and 15 days on half pay. " * 2])
    monkeypatch.setattr(config, "DOCS_DIR", docs)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "URL_FILE", tmp_path / "url.txt")
    monkeypatch.setattr(config, "get_llm", lambda: fake_llm("ما مدة إجازة الوضع؟", "Maternity leave is 60 days [1]."))
    monkeypatch.setattr(vectorstore, "e5_embeddings", lambda: DeterministicFakeEmbedding(size=16))

    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    app.text_input[0].input("How long is maternity leave?")
    app.button[0].click().run()

    assert not app.exception
    page = " ".join(m.value for m in app.markdown)
    assert "Maternity leave is 60 days [1]." in page
    assert "[1] leave.pdf, page 1" in page
