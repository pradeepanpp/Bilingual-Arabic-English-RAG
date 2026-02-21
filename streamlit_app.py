import streamlit as st
from pathlib import Path
import sys
import time


sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


from src.evaluation.ragas_eval import run_ragas_eval, append_scores_json


st.set_page_config(
    page_title="Bilingual RAG: Arabic–English Intelligence",
    page_icon="⚖️",
    layout="centered"
)

st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def render_text(text: str, rtl: bool):
    """Render text with RTL/LTR direction using HTML."""
    direction = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"
    safe_text = (text or "").replace("\n", "<br>")
    st.markdown(
        f"<div dir='{direction}' style='text-align:{align}; line-height:1.6'>{safe_text}</div>",
        unsafe_allow_html=True
    )


def init_session_state():
    """Initialize session state variables"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "rag_mode_key" not in st.session_state:
        st.session_state.rag_mode_key = None


@st.cache_resource
def build_retriever_and_llm():
    """Build core components once (cached): llm + retriever"""
    llm = Config.get_llm()

    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    vector_store = VectorStore()

    sources = Config.DEFAULT_SOURCE_FILES

    documents = doc_processor.process_sources(sources)
    vector_store.create_vectorstore(documents)

    return llm, vector_store.get_retriever(), len(documents)


def main():
    """Main application"""
    init_session_state()


    st.sidebar.header("Mode")
    use_agent = st.sidebar.checkbox("🤖 Agent Mode (Agentic RAG)", value=False)
    rtl_mode = st.sidebar.checkbox("Arabic Layout (RTL)", value=False)

    st.sidebar.markdown("---")
    enable_eval = st.sidebar.checkbox("📊 Evaluation (RAGAS)", value=False)
    st.sidebar.caption("When enabled, logs metrics to scores.json")

    st.sidebar.markdown("---")
    st.sidebar.caption("Sources: data/url.txt + PDFs inside data/")


    st.title("⚖️ Bilingual RAG: Arabic–English Intelligence")
    st.markdown("Arabic–English retrieval with source attribution (URLs + PDFs)")

    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            try:
                llm, retriever, num_chunks = build_retriever_and_llm()
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
            except Exception as e:
                st.error(f"Failed to initialize: {str(e)}")
                return
    else:
        llm, retriever, num_chunks = build_retriever_and_llm()


    mode_key = (use_agent,)
    if (st.session_state.rag_system is None) or (st.session_state.rag_mode_key != mode_key):
        graph_builder = GraphBuilder(
            retriever=retriever,
            llm=llm,
            use_agent=use_agent,
        )
        graph_builder.build()
        st.session_state.rag_system = graph_builder
        st.session_state.rag_mode_key = mode_key

    st.markdown("---")

    placeholder = "اكتب سؤالك هنا..." if rtl_mode else "Type your question here..."
    with st.form("search_form"):
        question = st.text_input(
            "Ask a question:",
            placeholder=placeholder
        )
        submit = st.form_submit_button("🔍 Search")

    
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()

            
                question_for_system = (
                    f"أجب باللغة العربية فقط.\n\nالسؤال: {question}"
                    if rtl_mode else question
                )

                result = st.session_state.rag_system.run(question_for_system)

                elapsed_time = time.time() - start_time

            
                st.session_state.history.append({
                    "question": question,
                    "answer": result.get("answer"),
                    "time": elapsed_time
                })

            
                st.markdown("### 💡 Answer")
                answer = result.get("answer", "No answer produced.")
                render_text(answer, rtl=rtl_mode)

            
                retrieved = result.get("retrieved_docs") or []
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(retrieved, 1):
                        meta = getattr(doc, "metadata", {}) or {}
                        source = meta.get("source") or meta.get("url") or meta.get("title") or "unknown"
                        page = meta.get("page")
                        ocr_used = bool(meta.get("ocr_used", False))

                        
                        page_str = f"{page + 1}" if isinstance(page, int) else (str(page) if page is not None else None)

                        header = f"Document {i} | Source: {source}"
                        if page_str is not None:
                            header += f" | Page: {page_str}"
                        header += " | OCR: ✅" if ocr_used else " | OCR: —"

                        st.markdown(f"**{header}**")

                        chunk_text = (doc.page_content[:500] + "...") if doc.page_content else ""
                        render_text(chunk_text, rtl=rtl_mode)
                        st.markdown("---")

                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

                
                if enable_eval:
                    with st.spinner("Running evaluation (RAGAS)..."):
                        contexts = [
                            d.page_content for d in retrieved
                            if getattr(d, "page_content", None)
                        ]

                        
                        sources_meta = []
                        for d in retrieved[:6]:
                            meta = getattr(d, "metadata", {}) or {}
                            sources_meta.append({
                                "source": meta.get("source") or meta.get("url") or meta.get("title"),
                                "page": meta.get("page"),
                                "ocr_used": bool(meta.get("ocr_used", False)),
                            })

                        metrics = run_ragas_eval(
                            question=question,              
                            answer=answer,                  
                            contexts=contexts,
                            ground_truth=None,              
                        )

                        append_scores_json(
                            out_path="scores.json",
                            question=question,
                            answer=answer,
                            docs_metadata=sources_meta,
                            metrics=metrics,
                        )

                        st.markdown("### 📊 Evaluation (RAGAS)")
                        st.json(metrics)
                        st.caption("Saved to scores.json")

    
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")

        for item in reversed(st.session_state.history[-3:]):  
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                preview = (str(item["answer"])[:200] + "...") if item.get("answer") else ""
                render_text(preview, rtl=rtl_mode)
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")


if __name__ == "__main__":
    main()