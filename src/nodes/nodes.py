"""Retrieve-then-answer nodes: search (optionally in both languages) and the answer prompt."""

import re

from langchain_core.documents import Document

from src.state.rag_state import RAGState

NOT_FOUND = {
    "en": "The documents do not contain this information.",
    "ar": "لا تحتوي المستندات على هذه المعلومة.",
}
LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}
ARABIC_LETTER = re.compile(r"[ء-ي]")
RRF_K = 60  # standard constant for reciprocal rank fusion

PROMPT = """You answer questions using only the numbered sources below.
- Cite the sources you use, like [1] or [2].
- Use only facts the sources state about exactly what is asked. If they only cover a related
  topic, or do not contain the answer, reply only: "{not_found}"
- Answer in {language}.

Sources:
{sources}

Question: {question}
Answer:"""

TRANSLATE_PROMPT = """Translate this search question into {language}. Keep numbers and technical terms.
Return only the translation.

{question}"""


def detect_language(text: str) -> str:
    """'ar' if most letters are Arabic, otherwise 'en'."""
    letters = [c for c in text if c.isalpha()]
    arabic = sum(bool(ARABIC_LETTER.match(c)) for c in letters)
    return "ar" if letters and arabic / len(letters) > 0.5 else "en"


def resolve_language(question: str, answer_language: str = "auto") -> str:
    return answer_language if answer_language in LANGUAGE_NAMES else detect_language(question)


def source_label(doc: Document) -> str:
    meta = doc.metadata or {}
    label = meta.get("source", "unknown")
    if meta.get("page") is not None:
        label += f", page {meta['page'] + 1}"
    return label


def format_sources(docs: list[Document], start: int = 1) -> str:
    return "\n\n".join(f"[{i}] {source_label(d)}\n{d.page_content}" for i, d in enumerate(docs, start))


def build_prompt(question: str, docs: list[Document], answer_language: str = "auto") -> str:
    language = resolve_language(question, answer_language)
    return PROMPT.format(not_found=NOT_FOUND[language], language=LANGUAGE_NAMES[language],
                         sources=format_sources(docs), question=question)


def fuse(result_lists: list[list[Document]], k: int) -> list[Document]:
    """Reciprocal rank fusion: a chunk ranked high in either list ends up high in the merged list."""
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results, 1):
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank)
            docs[key] = doc
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[key] for key in best]


class RAGNodes:
    def __init__(self, retriever, llm, translate_query: bool = True, k: int = 6):
        self.retriever = retriever
        self.llm = llm
        self.translate_query = translate_query
        self.k = k

    def translate(self, question: str) -> str:
        target = "en" if detect_language(question) == "ar" else "ar"
        prompt = TRANSLATE_PROMPT.format(language=LANGUAGE_NAMES[target], question=question)
        return self.llm.invoke(prompt).content.strip()

    def search(self, question: str) -> list[Document]:
        """
        Search with the question and, if enabled, with its translation into the other language,
        because the embedding model matches same-language text much better than cross-language.
        The answer language never changes the search.
        """
        results = [self.retriever.invoke(question)]
        if self.translate_query:
            results.append(self.retriever.invoke(self.translate(question)))
        return fuse(results, self.k)

    def retrieve_docs(self, state: RAGState) -> dict:
        return {"retrieved_docs": self.search(state.question)}

    def generate_answer(self, state: RAGState) -> dict:
        prompt = build_prompt(state.question, state.retrieved_docs, state.answer_language)
        return {"answer": self.llm.invoke(prompt).content}
