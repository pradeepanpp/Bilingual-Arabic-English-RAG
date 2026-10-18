# Evaluation

48 questions, embedding model `intfloat/multilingual-e5-base`, top-6 chunks.

## Retrieval: fact recall@6 (page recall@6)

| Setup | All | EN→AR | AR→AR | EN→EN | AR→EN | Cross-lingual | MRR (fact) |
|---|---|---|---|---|---|---|---|
| pdfplumber | 29% (38%) | 0% (0%) | 0% (10%) | 70% (90%) | 62% (75%) | 23% (27%) | 0.22 |
| pymupdf | 38% (79%) | 21% (79%) | 0% (70%) | 70% (90%) | 75% (75%) | 41% (77%) | 0.30 |
| pymupdf+ocr | 69% (79%) | 43% (57%) | 100% (100%) | 70% (90%) | 75% (75%) | 55% (64%) | 0.48 |
| +translation | 79% (90%) | 71% (86%) | 90% (100%) | 70% (90%) | 88% (88%) | 77% (86%) | 0.49 |
| +1000-char chunks | 83% (88%) | 79% (79%) | 100% (100%) | 70% (90%) | 88% (88%) | 82% (82%) | 0.49 |
