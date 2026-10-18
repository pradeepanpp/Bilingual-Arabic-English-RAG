# Evaluation

48 questions, embedding model `intfloat/multilingual-e5-small`, top-6 chunks.

## Retrieval: fact recall@6 (page recall@6)

| Setup | All | EN→AR | AR→AR | EN→EN | AR→EN | Cross-lingual | MRR (fact) |
|---|---|---|---|---|---|---|---|
| pdfplumber | 26% (36%) | 0% (0%) | 10% (20%) | 70% (90%) | 38% (50%) | 14% (18%) | 0.18 |
| pymupdf | 31% (69%) | 14% (50%) | 0% (90%) | 60% (80%) | 62% (62%) | 32% (55%) | 0.22 |
| pymupdf+ocr | 69% (74%) | 50% (50%) | 100% (100%) | 60% (80%) | 75% (75%) | 59% (59%) | 0.51 |
| +translation | 81% (86%) | 79% (79%) | 100% (100%) | 60% (80%) | 88% (88%) | 82% (82%) | 0.58 |
| +1000-char chunks | 86% (93%) | 93% (93%) | 100% (100%) | 60% (90%) | 88% (88%) | 91% (91%) | 0.62 |

## Answers (+translation settings)

- mode: retrieve-then-answer
- answer_accuracy: 0.81
- answerable_n: 42
- false_refusals: 5
- citation_rate: 0.881
- unanswerable_refused: 6/6
- language_match: 1.0
