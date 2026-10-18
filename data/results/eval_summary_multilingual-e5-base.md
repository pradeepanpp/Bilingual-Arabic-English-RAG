# Evaluation

48 questions, embedding model `intfloat/multilingual-e5-base`, top-6 chunks.

## Retrieval: fact recall@6 (page recall@6)

| Setup | All | EN→AR | AR→AR | EN→EN | AR→EN | Cross-lingual | MRR (fact) |
|---|---|---|---|---|---|---|---|
| pdfplumber | 29% (38%) | 0% (0%) | 0% (10%) | 80% (90%) | 50% (75%) | 18% (27%) | 0.24 |
| pymupdf | 40% (76%) | 7% (71%) | 10% (70%) | 90% (90%) | 75% (75%) | 32% (73%) | 0.25 |
| pymupdf+ocr | 69% (79%) | 29% (57%) | 100% (100%) | 90% (90%) | 75% (75%) | 46% (64%) | 0.50 |
| +translation | 86% (93%) | 79% (100%) | 100% (100%) | 90% (90%) | 75% (75%) | 77% (91%) | 0.54 |
| +1000-char chunks | 88% (95%) | 86% (100%) | 100% (100%) | 90% (90%) | 75% (88%) | 82% (96%) | 0.62 |
