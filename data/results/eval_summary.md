# Evaluation

48 questions, embedding model `intfloat/multilingual-e5-small`, top-6 chunks.

## Retrieval: fact recall@6 (page recall@6)

| Setup | All | EN→AR | AR→AR | EN→EN | AR→EN | Cross-lingual | MRR (fact) |
|---|---|---|---|---|---|---|---|
| pdfplumber | 26% (33%) | 0% (0%) | 10% (20%) | 80% (90%) | 25% (38%) | 9% (14%) | 0.15 |
| pymupdf | 33% (71%) | 7% (50%) | 10% (90%) | 80% (90%) | 50% (62%) | 23% (55%) | 0.25 |
| pymupdf+ocr | 64% (74%) | 36% (50%) | 100% (100%) | 80% (90%) | 50% (62%) | 41% (55%) | 0.50 |
| +translation | 88% (90%) | 93% (93%) | 100% (100%) | 80% (90%) | 75% (75%) | 86% (86%) | 0.59 |
| +1000-char chunks | 88% (93%) | 93% (100%) | 100% (100%) | 90% (90%) | 62% (75%) | 82% (91%) | 0.63 |

## Answers (+translation settings)

- mode: retrieve-then-answer
- answer_accuracy: 0.786
- answerable_n: 42
- false_refusals: 5
- citation_rate: 0.881
- unanswerable_refused: 6/6
- language_match: 1.0
