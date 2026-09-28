"""Correctness and per-run derived metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def contains_match(pred: str, gold: str) -> bool:
    """BABILong's official scoring: the target word appears in the output."""
    g = normalize(gold)
    return bool(g) and re.search(rf"\b{re.escape(g)}\b", normalize(pred)) is not None


def is_correct(pred: str, gold: str, dataset: str) -> bool:
    if dataset == "babilong":
        return contains_match(pred, gold)
    # BrowseComp answers are short; the official grader is an LLM judge
    # (eval/judge.py). EM-or-contains is the cheap, deterministic proxy.
    return exact_match(pred, gold) or contains_match(pred, gold)


def over_read_ratio(total_tokens: int, gold_seen_at: Optional[int]) -> Optional[float]:
    """Share of the run's tokens spent after all required evidence had been read."""
    if not total_tokens or gold_seen_at is None:
        return None
    return max(0.0, (total_tokens - gold_seen_at) / total_tokens)
