"""Retrieve-then-answer nodes and the answer prompt."""

from langchain_core.documents import Document

from src.state.rag_state import RAGState

NOT_FOUND = {
    "en": "The documents do not contain this information.",
    "ar": "لا تحتوي المستندات على هذه المعلومة.",
}

LANGUAGE_RULES = {
    "en": "Answer in English.",
    "ar": "Answer in Arabic.",
    "auto": "Answer in the same language as the question.",
}

PROMPT = """You answer questions using only the numbered sources below.
- Cite the sources you use, like [1] or [2].
- If the sources do not contain the answer, reply only: "{not_found_en}" \
(in Arabic: "{not_found_ar}").
- {language_rule}

Sources:
{sources}

Question: {question}
Answer:"""


def source_label(doc: Document) -> str:
    meta = doc.metadata or {}
    label = meta.get("source", "unknown")
    if meta.get("page") is not None:
        label += f", page {meta['page'] + 1}"
    return label


def format_sources(docs: list[Document], start: int = 1) -> str:
    return "\n\n".join(f"[{i}] {source_label(d)}\n{d.page_content}" for i, d in enumerate(docs, start))


def build_prompt(question: str, docs: list[Document], answer_language: str = "auto") -> str:
    return PROMPT.format(
        not_found_en=NOT_FOUND["en"],
        not_found_ar=NOT_FOUND["ar"],
        language_rule=LANGUAGE_RULES.get(answer_language, LANGUAGE_RULES["auto"]),
        sources=format_sources(docs),
        question=question,
    )


class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> dict:
        # Only the question is searched; the answer language never changes the search
        return {"retrieved_docs": self.retriever.invoke(state.question)}

    def generate_answer(self, state: RAGState) -> dict:
        prompt = build_prompt(state.question, state.retrieved_docs, state.answer_language)
        return {"answer": self.llm.invoke(prompt).content}
