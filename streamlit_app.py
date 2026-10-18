"""
Streamlit app: ask questions in Arabic or English about the indexed documents.

Run:  streamlit run streamlit_app.py
"""

import html
import time

import streamlit as st

from src.config import config
from src.document_ingestion.document_processor import DocumentProcessor
from src.graph_builder.graph_builder import GraphBuilder
from src.vectorstore.vectorstore import VectorStore

st.set_page_config(page_title="Bilingual RAG", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner=False)
def build_index(pdf_files: tuple, include_urls: bool):
    """Index the PDFs (and web pages). Cached until the file list changes."""
    processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    sources = list(pdf_files) + ([str(config.URL_FILE)] if include_urls else [])
    pages = processor.load_sources(sources)
    chunks = processor.split(pages)
    store = VectorStore()
    store.add_documents(chunks)
    ocr_pages = sum(1 for p in pages if p.metadata.get("method") == "ocr")
    return store.get_retriever(), len(chunks), ocr_pages


@st.cache_resource(show_spinner=False)
def get_llm():
    return config.get_llm()


def save_uploads(files) -> None:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        (config.UPLOAD_DIR / f.name).write_bytes(f.getvalue())


def pdf_files() -> tuple:
    files = sorted(config.DOCS_DIR.glob("*.pdf")) + sorted(config.UPLOAD_DIR.glob("*.pdf"))
    return tuple(str(f) for f in files)


def show_text(text: str, rtl: bool):
    """Text from documents and the model is escaped, so it can never inject HTML."""
    direction = "rtl" if rtl else "ltr"
    body = html.escape(text or "").replace("\n", "<br>")
    st.markdown(f"<div dir='{direction}' style='text-align:{'right' if rtl else 'left'};"
                f"line-height:1.6'>{body}</div>", unsafe_allow_html=True)


def source_header(i: int, doc) -> str:
    meta = doc.metadata or {}
    header = f"[{i}] {meta.get('source', 'unknown')}"
    if meta.get("page") is not None:
        header += f", page {meta['page'] + 1}"
    if meta.get("method") == "ocr":
        header += " (OCR)"
    return header


def main():
    st.session_state.setdefault("history", [])

    with st.sidebar:
        st.header("Settings")
        use_agent = st.checkbox("Agent mode", help="The model chooses its own searches")
        arabic = st.checkbox("Answer in Arabic (RTL)")
        both_languages = st.checkbox("Search in both languages", value=True,
                                     help="Also searches with the question translated into the other language")
        include_urls = st.checkbox("Include web pages from data/url.txt", value=True)
        st.divider()
        uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
        if uploads:
            save_uploads(uploads)
        st.caption("Scanned or badly encoded pages are read with OCR the first time (slow on a CPU), "
                   "then cached.")

    st.title("Bilingual RAG: Arabic–English")
    st.write("Ask in Arabic or English. Answers cite the passages they come from.")

    try:
        llm = get_llm()
        with st.spinner("Indexing documents..."):
            retriever, n_chunks, n_ocr = build_index(pdf_files(), include_urls)
    except Exception as e:
        st.error(f"Could not start: {e}")
        return
    st.caption(f"{n_chunks} chunks indexed, {n_ocr} pages read with OCR")

    with st.form("ask"):
        question = st.text_input("Question", placeholder="اكتب سؤالك هنا... / Type your question...")
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        start = time.time()
        with st.spinner("Searching..."):
            result = GraphBuilder(retriever, llm, use_agent=use_agent, translate_query=both_languages).run(
                question.strip(), answer_language="ar" if arabic else "auto")
        elapsed = time.time() - start
        answer = result.get("answer", "")
        st.session_state.history.append({"question": question, "answer": answer, "time": elapsed})

        st.subheader("Answer")
        show_text(answer, rtl=arabic)
        docs = result.get("retrieved_docs") or []
        with st.expander(f"Sources ({len(docs)})"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**{html.escape(source_header(i, doc))}**")
                show_text(doc.page_content[:500], rtl=arabic)
        st.caption(f"{elapsed:.1f} s")

    if st.session_state.history:
        st.divider()
        st.subheader("Recent questions")
        for item in reversed(st.session_state.history[-3:]):
            show_text(f"Q: {item['question']}", rtl=arabic)
            show_text(item["answer"][:200], rtl=arabic)
            st.caption(f"{item['time']:.1f} s")


if __name__ == "__main__":
    main()
