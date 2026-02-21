from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
)


@dataclass
class RAGASResult:
    timestamp: str
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, float]


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def run_ragas_eval(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, float]:
    
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

 
    if ground_truth:
        data["ground_truth"] = [ground_truth]

    ds = Dataset.from_dict(data)

    result = evaluate(
        ds,
        metrics=[
            faithfulness,
        

        ],
    )

   
    scores = result.to_pandas().iloc[0].to_dict()

   
    cleaned = {}
    for k, v in scores.items():
        if k in {
            "faithfulness",
        }:
            cleaned[k] = _safe_float(v) if v is not None else None

    return cleaned


def append_scores_json(
    out_path: str | Path,
    question: str,
    answer: str,
    docs_metadata: List[Dict[str, Any]],
    metrics: Dict[str, float],
) -> None:
    """
    Append a single evaluation record to scores.json (list of records).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = RAGASResult(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        question=question,
        answer=answer,
        sources=docs_metadata,
        metrics=metrics,
    )

    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []

    existing.append(asdict(record))
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")