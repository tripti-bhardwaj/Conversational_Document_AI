from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class ConfidenceResult:
    retrieval_confidence: float
    answer_confidence: float
    combined_confidence: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_similarity(score: float) -> float:
    if score < 0:
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    return max(0.0, min(1.0, score))


def retrieval_confidence(source_chunks: List[Dict[str, Any]]) -> float:
    if not source_chunks:
        return 0.0
    confidences = [float(c.get("retrieval_confidence", normalize_similarity(float(c.get("score", 0.0))))) for c in source_chunks]
    return max(confidences)


def heuristic_answer_confidence(answer: str, fallback: bool = False) -> float:
    answer = (answer or "").strip()
    if fallback or not answer:
        return 0.1
    weak_markers = ["could not find", "not in the context", "i don't know", "not enough information"]
    if any(marker in answer.lower() for marker in weak_markers):
        return 0.2
    if len(answer.split()) < 3:
        return 0.45
    return 0.85


def combined_confidence(source_chunks: List[Dict[str, Any]], answer: str, fallback: bool = False, retrieval_weight: float = 0.65) -> ConfidenceResult:
    retrieval = retrieval_confidence(source_chunks)
    answer_score = heuristic_answer_confidence(answer, fallback=fallback)
    combined = retrieval_weight * retrieval + (1.0 - retrieval_weight) * answer_score
    if fallback or combined < 0.45:
        label = "low"
    elif combined < 0.7:
        label = "medium"
    else:
        label = "high"
    return ConfidenceResult(round(retrieval, 4), round(answer_score, 4), round(combined, 4), label)
