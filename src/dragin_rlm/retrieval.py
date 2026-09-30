"""Stand-in for DRAGIN's retrieval module (paper: BM25 over a Wikipedia index).

DRAGIN-RLM has no external corpus to query. BrowseComp-Plus already assembles,
per question, a single long document (gold + evidence + sampled hard-negative
passages, shuffled) as the RLM's context -- that document IS the corpus for this
question. So "retrieval" here means: given the QFS query, rank passages of that
already-provided document by BM25 term overlap and hand the best-matching slice
to a worker LLM call, instead of hitting an external index. This is the same
"read a slice of the stored context" move RLM's own worker sub-calls already
make; QFS just picks the slice instead of the root's own Python code.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

from gate_rlm.relevance import split_passages

_WORD = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


def bm25_rank(
    query: str, passages: List[str], k1: float = 1.5, b: float = 0.75
) -> List[Tuple[int, float]]:
    """Standard Okapi BM25 over a small, in-memory passage set (Robertson et al.,
    2009) -- no external index needed since ``passages`` is already everything
    there is to search."""
    q_terms = _tokenize(query)
    if not q_terms or not passages:
        return []
    doc_tokens = [_tokenize(p) for p in passages]
    doc_lens = [len(d) for d in doc_tokens]
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    n_docs = len(passages)
    df: Counter = Counter()
    for d in doc_tokens:
        df.update(set(d))
    scores = [0.0] * n_docs
    for pi, d in enumerate(doc_tokens):
        tf = Counter(d)
        for term in q_terms:
            n_q = df.get(term, 0)
            freq = tf.get(term, 0)
            if n_q == 0 or freq == 0:
                continue
            idf = math.log(1 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            denom = freq + k1 * (1 - b + b * doc_lens[pi] / (avgdl or 1))
            scores[pi] += idf * (freq * (k1 + 1)) / denom
    ranked = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)
    return [(i, scores[i]) for i in ranked]


def slice_context(
    document: str, query: str, top_k: int = 3, passage_chars: int = 1000
) -> str:
    """Best-matching passages for ``query``, joined back in their original
    left-to-right order (so the worker sub-call sees them the way it would have
    encountered them in the document)."""
    passages = split_passages(document, passage_chars)
    if not passages:
        return ""
    ranked = bm25_rank(query, passages)
    keep_idx = [i for i, score in ranked[:top_k] if score > 0]
    if not keep_idx:
        return ""
    keep_idx.sort()
    return "\n\n".join(passages[i] for i in keep_idx)
