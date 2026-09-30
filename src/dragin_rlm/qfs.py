"""QFS: Query Formulation based on Self-attention (DRAGIN, Su et al. 2024, §3.2).

Given the token t_i that triggered RIND, rank the tokens that precede it,
{t_{i-1}, ..., t_1}, by how much t_i attended to them (its own generation-time
attention row -- RindState.qfs_row), take the top n, and put them back in their
original left-to-right order to form the query (steps 1-4 of §3.2).

Simplification versus the paper: §3.2 step 3 maps selected subword tokens back to
their whole containing words before building the query ("find the words
corresponding to these tokens"). This module skips that remapping and uses the
selected tokens' own decoded text directly, so a query built from token
boundaries that split a word may contain a bare subword fragment rather than the
whole word. This trades a small amount of query readability for not needing the
tokenizer's offset mapping; it doesn't change which underlying content gets
retrieved; see harness.py for how the query is actually used (a BM25-lite ranking
over the RLM's own context.py, not a lookup against an external index).
"""

from __future__ import annotations

from typing import List, Sequence


def select_query_tokens(
    prior_tokens: Sequence[str], attn_row: Sequence[float], top_n: int
) -> List[str]:
    """``prior_tokens[j]`` is the token at position j; ``attn_row[j]`` is the
    triggering token's attention to it. Returns the top-n tokens' text, restored
    to original left-to-right order (§3.2 step 3)."""
    n = min(len(prior_tokens), len(attn_row))
    if n == 0 or top_n <= 0:
        return []
    ranked = sorted(range(n), key=lambda j: attn_row[j], reverse=True)[:top_n]
    ranked.sort()
    return [prior_tokens[j] for j in ranked]


def format_query(prior_tokens: Sequence[str], attn_row: Sequence[float], top_n: int) -> str:
    """The selected tokens joined into a single query string (§3.2 step 4)."""
    tokens = select_query_tokens(prior_tokens, attn_row, top_n)
    return " ".join(t.strip() for t in tokens if t.strip())
