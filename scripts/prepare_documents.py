"""
Read every PDF in data/docs and fill the OCR cache, so the app starts fast.
Pages that need OCR take tens of seconds each on a CPU; cached pages are instant.

Run:  python scripts/prepare_documents.py
"""

import logging
import sys
from collections import Counter

sys.path.append(".")

from src.config import config
from src.document_ingestion.document_processor import DocumentProcessor


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    processor = DocumentProcessor()
    for pdf in sorted(config.DOCS_DIR.glob("*.pdf")):
        pages = processor.load_pdf(pdf)
        methods = Counter(p.metadata["method"] for p in pages)
        print(f"{pdf.name}: {len(pages)} pages ({methods['text']} text layer, {methods['ocr']} OCR)")


if __name__ == "__main__":
    main()
