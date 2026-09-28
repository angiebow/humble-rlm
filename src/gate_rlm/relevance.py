"""Stage 1 relevance scoring: cosine prefilter, then a cross-encoder reranker.

Cosine similarity is cheap but only measures topical similarity; distractor chunks
can be similar and still useless. The cross-encoder reads (query, passage) jointly
and is the relevance score we gate on. Long chunks are split into passages; the
chunk score is the max over its best passages.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence


@dataclass
class RelevanceScore:
    cosine: float        # max cosine(query, passage) over the chunk
    rerank: float        # max sigmoid(cross-encoder logit) over top-m passages
    n_passages: int


class Scorer(Protocol):
    def score(self, query: str, chunk: str) -> RelevanceScore: ...


def split_passages(text: str, passage_chars: int) -> List[str]:
    """Split on paragraph/sentence boundaries into ~passage_chars pieces."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= passage_chars:
        return [text]
    passages, buf = [], ""
    for piece in text.replace("\n\n", "\n").split("\n"):
        for sent in piece.split(". "):
            sent = sent.strip()
            if not sent:
                continue
            if len(buf) + len(sent) + 2 > passage_chars and buf:
                passages.append(buf)
                buf = ""
            buf = f"{buf} {sent}." if buf else f"{sent}."
            while len(buf) > passage_chars:  # very long sentence
                passages.append(buf[:passage_chars])
                buf = buf[passage_chars:]
    if buf:
        passages.append(buf)
    return passages


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderScorer:
    """bge-small embeddings for the prefilter, bge-reranker for relevance."""

    def __init__(
        self,
        embedder: str = "BAAI/bge-small-en-v1.5",
        reranker: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        passage_chars: int = 1000,
        top_m_passages: int = 8,
    ) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedder = SentenceTransformer(embedder, device=device)
        self.reranker = CrossEncoder(reranker, device=device)
        self.passage_chars = passage_chars
        self.top_m = top_m_passages
        self._lock = threading.Lock()  # sub-calls run concurrently in threads

    def score(self, query: str, chunk: str) -> RelevanceScore:
        passages = split_passages(chunk, self.passage_chars)
        if not passages:
            return RelevanceScore(0.0, 0.0, 0)
        with self._lock:
            embs = self.embedder.encode(
                [query] + passages, normalize_embeddings=True, convert_to_numpy=True
            )
            cos = embs[1:] @ embs[0]
            order = cos.argsort()[::-1][: self.top_m]
            logits = self.reranker.predict([(query, passages[i]) for i in order])
        rerank = max(_sigmoid(float(x)) for x in logits)
        return RelevanceScore(float(cos.max()), rerank, len(passages))


class KeywordScorer:
    """Dependency-free stand-in for tests and dry runs (token overlap)."""

    def __init__(self, passage_chars: int = 1000) -> None:
        self.passage_chars = passage_chars

    def score(self, query: str, chunk: str) -> RelevanceScore:
        q = {w.strip("?.,!").lower() for w in query.split() if len(w) > 3}
        passages = split_passages(chunk, self.passage_chars) or [""]
        best = 0.0
        for p in passages:
            words = {w.strip("?.,!").lower() for w in p.split()}
            best = max(best, len(q & words) / max(1, len(q)))
        return RelevanceScore(best, best, len(passages))


def build_scorer(cfg: dict, dry_run: bool = False) -> Optional[Scorer]:
    rel = cfg.get("relevance", {})
    if not rel.get("enabled", True) and cfg.get("mode") != "observe":
        return None
    if dry_run:
        return KeywordScorer(rel.get("passage_chars", 1000))
    return CrossEncoderScorer(
        embedder=rel.get("embedder", "BAAI/bge-small-en-v1.5"),
        reranker=rel.get("reranker", "BAAI/bge-reranker-base"),
        device=rel.get("device", "cpu"),
        passage_chars=rel.get("passage_chars", 1000),
        top_m_passages=rel.get("top_m_passages", 8),
    )


def contains_any(text: str, fingerprints: Sequence[str]) -> bool:
    """Logging-only check: does ``text`` contain any gold-evidence fingerprint?"""
    return any(fp and fp in text for fp in fingerprints)
