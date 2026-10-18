"""
Evaluate the pipeline on data/eval/questions.json (48 questions about the
two PDFs in data/docs: 42 answerable, 6 not answerable from the documents).

Retrieval: for each configuration, index both PDFs and check the top-k chunks.
    page recall - a chunk from the correct page is in the top k
    fact recall - a chunk in the top k contains the fact itself (stricter)
Configurations, each adding one change to the one before:
    pdfplumber           the original extraction (Arabic comes out reversed)
    pymupdf              PyMuPDF text layer, cleaned, no OCR
    pymupdf+ocr          + OCR for pages that fail the quality check
    +translation         + search with the question translated into the other language
    +1000-char chunks    + bigger chunks
The translation configurations need OPENAI_API_KEY (one cheap call per question).

Answers (--answers): every question is answered with the app's settings and
graded by the LLM against the reference answer; unanswerable questions must be refused.

    python scripts/evaluate.py --answers
    python scripts/evaluate.py --model intfloat/multilingual-e5-base   # compare embedding models

Writes data/results/eval_report[_<model>].json and eval_summary[_<model>].md.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.append(".")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_core.documents import Document

from src.config import config
from src.document_ingestion.document_processor import DocumentProcessor
from src.evaluation.metrics import (
    answer_language, direction, has_citation, is_refusal, relevant_rank, retrieval_summary,
)
from src.nodes.nodes import RAGNodes
from src.vectorstore.vectorstore import EMBEDDING_MODEL, TOP_K, VectorStore, e5_embeddings

QUESTIONS = Path("data/eval/questions.json")
PDFS = [config.DOCS_DIR / "uae_labour_law_ar.pdf", config.DOCS_DIR / "attention.pdf"]
RESULTS_DIR = Path("data/results")
GROUPS = ["all", "en->ar", "ar->ar", "en->en", "ar->en", "cross-lingual"]

JUDGE_PROMPT = """You grade answers from a question-answering system.
The answer may be in a different language from the reference; that is fine.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Reply CORRECT only if the system answer states every key fact of the reference
(numbers, durations, percentages, conditions) and contradicts none of them.
If the reference gives alternatives, one of them is enough.
Extra correct detail is fine. Otherwise reply WRONG. Reply with one word."""


@dataclass
class Setup:
    name: str
    extraction: str
    translate: bool = False
    chunk_size: int = config.CHUNK_SIZE


SETUPS = [
    Setup("pdfplumber", "pdfplumber"),
    Setup("pymupdf", "pymupdf"),
    Setup("pymupdf+ocr", "pymupdf+ocr"),
    Setup("+translation", "pymupdf+ocr", translate=True),
    Setup("+1000-char chunks", "pymupdf+ocr", translate=True, chunk_size=1000),
]
APP_SETUP = SETUPS[3]  # the app's settings: OCR, translation, 500-character chunks


class CachedTranslationNodes(RAGNodes):
    """Translate each question once, however many setups use it."""
    translations = {}

    def translate(self, question: str) -> str:
        if question not in self.translations:
            self.translations[question] = super().translate(question)
        return self.translations[question]


def pdfplumber_pages(path: Path) -> list[Document]:
    """The original project's extraction: pdfplumber text, no cleaning (its OCR never ran)."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [Document(page_content=text, metadata={"source": path.name, "page": i, "method": "text"})
                for i, page in enumerate(pdf.pages) if (text := (page.extract_text() or "").strip())]


def load_pages(extraction: str) -> list[Document]:
    if extraction == "pdfplumber":
        return [p for pdf in PDFS for p in pdfplumber_pages(pdf)]
    processor = DocumentProcessor(use_ocr=(extraction == "pymupdf+ocr"))
    return [p for pdf in PDFS for p in processor.load_pdf(pdf)]


def build_retriever(setup: Setup, embedding):
    processor = DocumentProcessor(setup.chunk_size, max(50, setup.chunk_size // 10))
    chunks = processor.split(load_pages(setup.extraction))
    store = VectorStore(embedding)
    store.add_documents(chunks)
    return store.get_retriever(TOP_K), len(chunks)


def evaluate_retrieval(setup: Setup, questions: list, embedding, llm) -> tuple:
    retriever, n_chunks = build_retriever(setup, embedding)
    nodes = CachedTranslationNodes(retriever, llm, translate_query=setup.translate, k=TOP_K)
    rows, start = [], time.perf_counter()
    for item in [q for q in questions if q["source"]]:
        docs = nodes.search(item["question"])
        rows.append({"id": item["id"], "direction": direction(item),
                     "page_rank": relevant_rank(docs, item), "fact_rank": relevant_rank(docs, item, fact=True)})
    ms_per_query = (time.perf_counter() - start) * 1000 / len(rows)

    summary = {"chunks": n_chunks, "ms_per_query": round(ms_per_query, 1)}
    for level in ["page", "fact"]:
        ranks = {g: [r[f"{level}_rank"] for r in rows if g == "all" or r["direction"] == g] for g in GROUPS[:-1]}
        ranks["cross-lingual"] = ranks["en->ar"] + ranks["ar->en"]
        summary[level] = {g: retrieval_summary(ranks[g]) for g in GROUPS}
    return summary, rows, retriever


def evaluate_answers(questions: list, retriever, llm, use_agent: bool) -> tuple:
    from src.graph_builder.graph_builder import GraphBuilder

    graph = GraphBuilder(retriever, llm, use_agent=use_agent, translate_query=APP_SETUP.translate)
    rows = []
    for item in questions:
        answer = graph.run(item["question"]).get("answer", "")
        row = {"id": item["id"], "answer": answer, "refused": is_refusal(answer),
               "language_ok": answer_language(answer) == item["language"]}
        if item["source"] is None:
            row["correct"] = row["refused"]
        elif row["refused"]:
            row["correct"], row["cited"] = False, False
        else:
            verdict = llm.invoke(JUDGE_PROMPT.format(
                question=item["question"], reference=item["answer"], answer=answer)).content
            row["correct"] = verdict.strip().upper().startswith("CORRECT")
            row["cited"] = has_citation(answer)
        rows.append(row)
        print(f"  {item['id']}: {'correct' if row['correct'] else 'WRONG'}", flush=True)

    answerable = [r for r in rows if not r["id"].startswith("none")]
    unanswerable = [r for r in rows if r["id"].startswith("none")]
    summary = {
        "mode": "agent" if use_agent else "retrieve-then-answer",
        "answer_accuracy": round(sum(r["correct"] for r in answerable) / len(answerable), 3),
        "answerable_n": len(answerable),
        "false_refusals": sum(r["refused"] for r in answerable),
        "citation_rate": round(sum(r["cited"] for r in answerable) / len(answerable), 3),
        "unanswerable_refused": f"{sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}",
        "language_match": round(sum(r["language_ok"] for r in rows) / len(rows), 3),
    }
    return summary, rows


def markdown_summary(report: dict) -> str:
    lines = ["# Evaluation", "",
             f"{report['questions']} questions, embedding model `{report['model']}`, top-{TOP_K} chunks.", "",
             "## Retrieval: fact recall@6 (page recall@6)", "",
             "| Setup | All | EN→AR | AR→AR | EN→EN | AR→EN | Cross-lingual | MRR (fact) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in report["retrieval"].items():
        cells = [f"{s['fact'][g]['recall_at_k']:.0%} ({s['page'][g]['recall_at_k']:.0%})" for g in GROUPS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {s['fact']['all']['mrr']:.2f} |")
    if "answers" in report:
        lines += ["", f"## Answers ({APP_SETUP.name} settings)", ""]
        lines += [f"- {key}: {value}" for key, value in report["answers"].items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also grade LLM answers")
    parser.add_argument("--agent", action="store_true", help="grade agent mode instead of retrieve-then-answer")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="embedding model to evaluate")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    missing = [str(p) for p in PDFS if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing}. Run: python scripts/download_data.py")
    llm = config.get_llm() if os.getenv("OPENAI_API_KEY") else None
    if llm is None:
        print("No OPENAI_API_KEY: skipping the translation setups and the answers.")

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    embedding = e5_embeddings(args.model)
    report = {"questions": len(questions), "model": args.model, "retrieval": {}, "per_question": {}}
    app_retriever = None
    for setup in SETUPS:
        if setup.translate and llm is None:
            continue
        print(f"Retrieval: {setup.name}...", flush=True)
        summary, rows, retriever = evaluate_retrieval(setup, questions, embedding, llm)
        report["retrieval"][setup.name] = summary
        report["per_question"][setup.name] = rows
        if setup == APP_SETUP:
            app_retriever = retriever

    if args.answers and app_retriever is not None:
        print("Answering and grading...", flush=True)
        report["answers"], report["per_question"]["answers"] = evaluate_answers(
            questions, app_retriever, llm, args.agent)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_agent" if args.agent else "") + ("" if args.model == EMBEDDING_MODEL else "_" + args.model.split("/")[-1])
    (RESULTS_DIR / f"eval_report{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = markdown_summary(report)
    (RESULTS_DIR / f"eval_summary{suffix}.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
