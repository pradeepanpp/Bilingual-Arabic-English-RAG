import json
from collections import Counter
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.document_ingestion.document_processor import DocumentProcessor
from src.evaluation.metrics import (
    answer_language, direction, has_citation, is_refusal, relevant_rank, retrieval_summary, wilson_interval,
)

QUESTIONS = json.loads(Path("data/eval/questions.json").read_text(encoding="utf-8"))
DOCS = Path("data/docs")


def test_question_set_shape():
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS) == 48
    answerable = [q for q in QUESTIONS if q["source"]]
    assert len(answerable) == 42
    assert Counter(direction(q) for q in answerable) == {"en->ar": 14, "ar->ar": 10, "en->en": 10, "ar->en": 8}
    for q in answerable:
        assert q["pages"] and q["evidence"] and q["answer"]


@pytest.mark.parametrize("source", ["uae_labour_law_ar.pdf", "attention.pdf"])
def test_gold_pages_contain_the_evidence(source):
    """Every answer page must really contain the facts (checked on the text the app indexes)."""
    if not (DOCS / source).exists():
        pytest.skip(f"{source} not downloaded (python scripts/download_data.py)")

    class NoOCR:
        def readtext(self, *args, **kwargs):
            raise RuntimeError("only cached OCR is used in tests")

    processor = DocumentProcessor()
    processor.ocr_reader = NoOCR()
    pages = {p.metadata["page"] + 1: "".join(p.page_content.split()).lower() for p in processor.load_pdf(DOCS / source)}
    for q in [q for q in QUESTIONS if q["source"] == source]:
        for evidence in q["evidence"]:
            needle = "".join(evidence.split()).lower()
            assert any(needle in pages.get(p, "") for p in q["pages"]), (q["id"], evidence)


def test_relevant_rank():
    item = {"source": "law.pdf", "pages": [33]}
    docs = [Document(page_content="", metadata={"source": "law.pdf", "page": 10}),
            Document(page_content="", metadata={"source": "law.pdf", "page": 32})]
    assert relevant_rank(docs, item) == 2
    assert relevant_rank(docs[:1], item) is None


def test_retrieval_summary():
    s = retrieval_summary([1, 2, None, 4])
    assert s["n"] == 4 and s["hit_at_1"] == 0.25 and s["recall_at_k"] == 0.75
    assert s["mrr"] == round((1 + 0.5 + 0.25) / 4, 3)
    assert wilson_interval(0, 0) == [0.0, 0.0]


def test_answer_checks():
    assert is_refusal("The documents do not contain this information.")
    assert is_refusal("لا تحتوي المستندات على هذه المعلومة.")
    assert not is_refusal("Maternity leave is 60 days [1].")
    assert has_citation("60 days [1].") and not has_citation("60 days.")
    assert answer_language("مدة الإجازة 60 يوماً") == "ar"
    assert answer_language("It is 60 days, see القانون") == "en"
