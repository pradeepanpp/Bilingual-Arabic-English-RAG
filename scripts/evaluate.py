"""
Evaluate the pipeline on data/eval/questions.json (48 questions about the
two PDFs in data/docs: 42 answerable, 6 not answerable from the documents).

Retrieval (no API key needed): for each text-extraction method, index both
PDFs and check whether a chunk from the correct page is in the top 4.
    pdfplumber  - the original extraction (Arabic comes out reversed)
    pymupdf     - PyMuPDF text layer, cleaned, no OCR
    pymupdf+ocr - the app's method: PyMuPDF, OCR for pages that fail the check

Answers (--answers, needs OPENAI_API_KEY, costs a few cents): every
question is answered with the app's method and graded by the LLM against
the reference answer; questions without an answer must be refused.

    python scripts/evaluate.py
    python scripts/evaluate.py --answers [--agent]

Writes data/results/eval_report.json and eval_summary.md (with _agent for --agent).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.append(".")

from langchain_core.documents import Document

from src.config import config
from src.document_ingestion.document_processor import DocumentProcessor
from src.evaluation.metrics import (
    answer_language, direction, has_citation, is_refusal, relevant_rank, retrieval_summary,
)
from src.vectorstore.vectorstore import TOP_K, VectorStore, e5_embeddings

QUESTIONS = Path("data/eval/questions.json")
PDFS = [config.DOCS_DIR / "uae_labour_law_ar.pdf", config.DOCS_DIR / "attention.pdf"]
RESULTS_DIR = Path("data/results")
METHODS = ["pdfplumber", "pymupdf", "pymupdf+ocr"]

JUDGE_PROMPT = """You grade answers from a question-answering system.
The reference answer contains the facts that matter. The answer may be in
a different language from the reference; that is fine.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Is the system answer correct, i.e. does it state the key facts of the
reference without contradicting them? Reply with one word: CORRECT or WRONG."""


def pdfplumber_pages(path: Path) -> list[Document]:
    """The original project's extraction: pdfplumber text, no cleaning (its OCR never ran)."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [Document(page_content=text, metadata={"source": path.name, "page": i, "method": "text"})
                for i, page in enumerate(pdf.pages) if (text := (page.extract_text() or "").strip())]


def load_pages(method: str) -> list[Document]:
    if method == "pdfplumber":
        return [p for pdf in PDFS for p in pdfplumber_pages(pdf)]
    processor = DocumentProcessor(use_ocr=(method == "pymupdf+ocr"))
    return [p for pdf in PDFS for p in processor.load_pdf(pdf)]


def evaluate_retrieval(method: str, questions: list, embedding) -> tuple:
    processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    pages = load_pages(method)
    chunks = processor.split(pages)
    store = VectorStore(embedding)
    store.add_documents(chunks)
    retriever = store.get_retriever(TOP_K)

    rows, start = [], time.perf_counter()
    for item in questions:
        if item["source"] is None:
            continue
        ranks = relevant_rank(retriever.invoke(item["question"]), item)
        rows.append({"id": item["id"], "direction": direction(item), "rank": ranks})
    ms_per_query = (time.perf_counter() - start) * 1000 / max(len(rows), 1)

    groups = {"all": [r["rank"] for r in rows]}
    for name in ["en->ar", "ar->ar", "en->en", "ar->en"]:
        groups[name] = [r["rank"] for r in rows if r["direction"] == name]
    groups["cross-lingual"] = groups["en->ar"] + groups["ar->en"]
    summary = {name: retrieval_summary(ranks) for name, ranks in groups.items()}
    summary["chunks"] = len(chunks)
    summary["ocr_pages"] = sum(p.metadata.get("method") == "ocr" for p in pages)
    summary["ms_per_query"] = round(ms_per_query, 1)
    return summary, rows, retriever


def evaluate_answers(questions: list, retriever, use_agent: bool) -> tuple:
    from src.graph_builder.graph_builder import GraphBuilder

    llm = config.get_llm()
    graph = GraphBuilder(retriever, llm, use_agent=use_agent)
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
    lines = ["# Evaluation", "", f"{report['questions']} questions, top-{TOP_K} retrieval.", "",
             "## Retrieval: recall@4 (MRR)", "",
             "| Method | All | EN→AR | AR→AR | EN→EN | AR→EN | Cross-lingual |",
             "|---|---|---|---|---|---|---|"]
    for method, s in report["retrieval"].items():
        cells = [f"{s[g]['recall_at_k']:.0%} ({s[g]['mrr']:.2f})"
                 for g in ["all", "en->ar", "ar->ar", "en->en", "ar->en", "cross-lingual"]]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    if "answers" in report:
        lines += ["", "## Answers", ""]
        for key, value in report["answers"].items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also grade LLM answers (needs OPENAI_API_KEY)")
    parser.add_argument("--agent", action="store_true", help="grade agent mode instead of retrieve-then-answer")
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    missing = [str(p) for p in PDFS if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing}. Run: python scripts/download_data.py")

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    embedding = e5_embeddings()
    report = {"questions": len(questions), "retrieval": {}, "per_question": {}}
    retriever = None
    for method in args.methods:
        print(f"Retrieval with {method}...", flush=True)
        summary, rows, retriever = evaluate_retrieval(method, questions, embedding)
        report["retrieval"][method] = summary
        report["per_question"][method] = rows

    if args.answers:
        if args.methods[-1] != "pymupdf+ocr":
            sys.exit("--answers uses the app's method; include pymupdf+ocr last in --methods")
        print("Answering and grading...", flush=True)
        report["answers"], report["per_question"]["answers"] = evaluate_answers(questions, retriever, args.agent)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_agent" if args.agent else ""
    (RESULTS_DIR / f"eval_report{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = markdown_summary(report)
    (RESULTS_DIR / f"eval_summary{suffix}.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
