<div align="center">

# Bilingual RAG: Arabic–English Document Q&A

Ask questions in Arabic or English and get answers from Arabic and English documents with the source page.

[![CI](https://github.com/pradeepanpp/Bilingual-Arabic-English-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/pradeepanpp/Bilingual-Arabic-English-RAG/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-red.svg)](https://www.langchain.com/langgraph)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io/)

</div>

## Overview

Many organisations in the Gulf keep documents in both Arabic and English. Their staff often read only one of the two languages. This project answers a question in either language from documents in either language. Every answer cites its source page. If the documents do not contain the answer the system says so.

It was tested on 48 questions about the UAE labour law (100 pages in Arabic) and the paper "Attention Is All You Need" (15 pages in English).

| Result | Score |
|---|---|
| Correct answers | 81% (34 of 42) |
| Questions with no answer in the documents that were refused | 6 of 6 |
| Answers in the language of the question | 100% |

**Stack:** Python · LangGraph · GPT-4o-mini · multilingual-E5 · FAISS · PyMuPDF · EasyOCR · Streamlit

## How it works

<p align="center">
  <img src="docs/images/how-it-works.png" alt="How a question is answered" width="100%">
</p>

1. **Read:** PyMuPDF extracts the text of each page. Pages with broken Arabic text are read again with EasyOCR.
2. **Index:** The text is split into 500-character chunks and stored in FAISS with multilingual-E5 embeddings.
3. **Search:** The question is also translated into the other language. Both versions are searched and the results are merged.
4. **Answer:** GPT-4o-mini answers only from the retrieved chunks and cites the page.

## Results

Share of questions where the answer is found in the top 6 chunks:

| Version | Arabic questions on the Arabic law | Questions in the other language | All questions |
|---|---|---|---|
| First version (pdfplumber) | 10% | 14% | 26% |
| PyMuPDF with OCR for broken pages | 100% | 59% | 69% |
| Plus query translation (final) | 100% | 82% | 81% |

- **OCR fixed Arabic search.** The first version read "جدول المحتويات" as "تايوتحلما لودج" because the PDF text layer is broken.
- **Query translation fixed cross-language search.** English questions now find Arabic passages and the other way round.
- **A larger embedding model did not help.** `multilingual-e5-base` scored 79% against 81% so the smaller model is used.

## Getting started

```bash
git clone https://github.com/pradeepanpp/Bilingual-Arabic-English-RAG.git
cd Bilingual-Arabic-English-RAG
uv sync                                # or: pip install -r requirements.txt
cp .env.example .env                   # add your OPENAI_API_KEY
python scripts/download_data.py        # downloads the English paper
uv run streamlit run streamlit_app.py
```

Run `python scripts/evaluate.py --answers` to repeat the evaluation and `pytest -q` to run the tests.

## Limitations and next steps

- The test set is small (42 answerable questions). Scores changed by up to 3 questions between two runs.
- GPT-4o-mini grades its own answers.
- Next steps are a reranker and a larger separate test set.