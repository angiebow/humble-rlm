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
    # BrowseComp's official grader is an LLM judge; EM-or-contains is the
    # cheap, deterministic proxy used here. See semantic_similarity below
    # for a softer alternative when pred is a long, unfinished reasoning
    # chain rather than a clean short answer (contains_match still catches
    # a verbatim-but-buried gold string; it can't catch a paraphrase).
    return exact_match(pred, gold) or contains_match(pred, gold)


_EMBEDDER_CACHE: dict = {}


def _get_embedder(model_name: str):
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def semantic_similarity(
    pred: str, gold: str, model_name: str = "BAAI/bge-small-en-v1.5"
) -> float:
    """Cosine similarity between sentence-embeddings of ``pred`` and ``gold``
    (same embedder relevance.py already uses, so no new dependency).

    exact_match/contains_match need the gold string to appear verbatim; a
    model that answers correctly but paraphrases (e.g. "the ISBN is
    978-0-19-754970-4" with different hyphenation, or a name spelled with a
    diacritic the gold string doesn't have) scores 0 on both even though the
    content is right. Semantic similarity credits that at the cost of being
    a continuous score, not a crisp right/wrong -- see semantic_correct for
    turning it back into one via a threshold, which needs to be chosen
    empirically for this task, not assumed.
    """
    embedder = _get_embedder(model_name)
    embs = embedder.encode(
        [pred or "", gold or ""], normalize_embeddings=True, convert_to_numpy=True
    )
    return float(embs[0] @ embs[1])


def semantic_correct(
    pred: str,
    gold: str,
    threshold: float = 0.75,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> bool:
    return semantic_similarity(pred, gold, model_name) >= threshold


def over_read_ratio(total_tokens: int, gold_seen_at: Optional[int]) -> Optional[float]:
    """Share of the run's tokens spent after all required evidence had been read."""
    if not total_tokens or gold_seen_at is None:
        return None
    return max(0.0, (total_tokens - gold_seen_at) / total_tokens)
