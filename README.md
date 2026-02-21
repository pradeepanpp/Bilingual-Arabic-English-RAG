<div align="center">

# Bilingual RAG: Arabic–English Document Q&A

Arabic–English retrieval-augmented generation (RAG) for PDFs/URLs with OCR fallback, citations, and optional RAGAS evaluation.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Tooling](https://img.shields.io/badge/Package_Manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Orchestration](https://img.shields.io/badge/Orchestration-LangGraph-red.svg)](https://www.langchain.com/langgraph)
[![Evaluation](https://img.shields.io/badge/Evaluation-RAGAS-orange.svg)](https://ragas.io/)

</div>




<img width="1631" height="846" alt="Screenshot 2026-02-21 143913" src="https://github.com/user-attachments/assets/f923780d-5e36-44c9-9a9f-a3e327ff020d" />
) <img width="1629" height="846" alt="Screenshot 2026-02-21 143625" src="https://github.com/user-attachments/assets/f1b69286-e220-4be3-9a17-53cfed0454da" />

<p>
  <img src="assets/rag_project.gif" width="900" />
</p>

## Overview
This project implements a bilingual (Arabic–English) RAG pipeline that answers questions from user-provided documents while showing source citations. It is designed to work with mixed-quality PDFs, including scanned documents via OCR fallback.

**Key features**
- **Bilingual retrieval** using multilingual embeddings (Arabic ↔ English)
- **PDF ingestion with OCR fallback** (triggered when text extraction is insufficient)
- **FAISS-based vector search** with chunk metadata (source/page)
- **LangGraph orchestration** for a clear retrieval → generation workflow
- **Streamlit UI** with optional RTL layout for Arabic
- **Optional evaluation** using RAGAS; results can be saved to `scores.json`

## Architecture 
1. Ingest documents (PDF/URL) → extract text (OCR fallback if needed)
2. Chunk text → embed chunks → store in FAISS
3. For each query: embed query → retrieve top-k chunks → generate answer grounded in retrieved context
4. Display answer + citations (source/page); optionally run evaluation

## Evaluation (optional)
This repo includes an optional evaluation toggle using RAGAS metrics (e.g., faithfulness).  
Run evaluation on your own test set and inspect results in `scores.json`.

> Note: scores depend on the dataset, chunking settings, and model configuration.

## Getting started (uv)
```bash
# 1) Install dependencies
uv sync

# 2) Configure environment
cp .env.example .env
# set OPENAI_API_KEY 

# 3) Run the app
uv run streamlit run streamlit_app.py
