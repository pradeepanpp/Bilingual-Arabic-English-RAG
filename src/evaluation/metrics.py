"""Metrics for the evaluation set (data/eval/questions.json)."""

import math
import re

from src.nodes.nodes import NOT_FOUND

DOC_LANGUAGE = {"uae_labour_law_ar.pdf": "ar", "attention.pdf": "en"}
ARABIC_LETTER = re.compile(r"[ء-ي]")


def direction(item: dict) -> str:
    """Question language -> document language, e.g. "en->ar" for an English question about the Arabic law."""
    return f"{item['language']}->{DOC_LANGUAGE[item['source']]}"


def squash(text: str) -> str:
    return "".join((text or "").split()).lower()


def relevant_rank(docs: list, item: dict, fact: bool = False):
    """
    1-based rank of the first relevant chunk, or None.
    Page level: the chunk comes from a gold page. Fact level (stricter): it also
    contains every evidence string of the question, i.e. the fact itself.
    """
    for rank, doc in enumerate(docs, 1):
        meta = doc.metadata
        if meta.get("source") != item["source"] or meta.get("page", -1) + 1 not in item["pages"]:
            continue
        if not fact or all(squash(e) in squash(doc.page_content) for e in item["evidence"]):
            return rank
    return None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return [round(max(0.0, (centre - margin) / denom), 3), round(min(1.0, (centre + margin) / denom), 3)]


def retrieval_summary(ranks: list) -> dict:
    """ranks: first relevant rank per question (None = not found in the top k)."""
    n = len(ranks)
    hits = sum(r is not None for r in ranks)
    return {
        "n": n,
        "hit_at_1": round(sum(r == 1 for r in ranks) / n, 3) if n else 0.0,
        "recall_at_k": round(hits / n, 3) if n else 0.0,
        "recall_at_k_ci95": wilson_interval(hits, n),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3) if n else 0.0,
    }


def is_refusal(answer: str) -> bool:
    text = (answer or "").strip().rstrip(".").lower()
    return any(phrase.rstrip(".").lower() in text for phrase in NOT_FOUND.values())


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer or ""))


def answer_language(answer: str) -> str:
    letters = [c for c in (answer or "") if c.isalpha()]
    if not letters:
        return "none"
    return "ar" if sum(bool(ARABIC_LETTER.match(c)) for c in letters) / len(letters) > 0.5 else "en"
