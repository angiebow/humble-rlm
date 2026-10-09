"""QFS: Query Formulation based on Self-attention (DRAGIN, Su et al. 2024, §3.2).

Given the token t_i that triggered RIND, rank the tokens that precede it,
{t_{i-1}, ..., t_1}, by how much t_i attended to them (its own generation-time
attention row -- RindState.qfs_row), take the top n, and put them back in their
original left-to-right order to form the query (steps 1-4 of §3.2).

Simplification versus the paper: §3.2 step 3 maps selected subword tokens back to
their whole containing words. This module does that by using the leading-space
marker on each token (a token that starts with whitespace begins a new word):
each selected token is expanded to the full word that contains it. If no token
carries a leading space (a tokenizer without that convention), the selected
tokens are used as-is.

"""

from __future__ import annotations

import re
from typing import List, Sequence

_THINK_TAG = re.compile(r"</?think>")


def _selected_indices(attn_row: Sequence[float], n: int, top_n: int) -> List[int]:
    ranked = sorted(range(n), key=lambda j: attn_row[j], reverse=True)[:top_n]
    return sorted(ranked)


def select_query_tokens(
    prior_tokens: Sequence[str], attn_row: Sequence[float], top_n: int
) -> List[str]:
    """``prior_tokens[j]`` is the token at position j; ``attn_row[j]`` is the
    triggering token's attention to it. Returns the top-n tokens' text, restored
    to original left-to-right order (§3.2 step 3)."""
    n = min(len(prior_tokens), len(attn_row))
    if n == 0 or top_n <= 0:
        return []
    return [prior_tokens[j] for j in _selected_indices(attn_row, n, top_n)]


def _word_span(prior_tokens: Sequence[str], j: int) -> tuple:
    """Token range [start, end] of the whole word containing token j."""
    def starts_word(k: int) -> bool:
        return prior_tokens[k][:1].isspace()

    start = j
    while start > 0 and not starts_word(start):
        start -= 1
    end = j
    while end + 1 < len(prior_tokens) and not starts_word(end + 1):
        end += 1
    return start, end


def format_query(prior_tokens: Sequence[str], attn_row: Sequence[float], top_n: int) -> str:
    """The selected words joined into a single query string (§3.2 steps 3-4)."""
    n = min(len(prior_tokens), len(attn_row))
    if n == 0 or top_n <= 0:
        return ""
    selected = _selected_indices(attn_row, n, top_n)
    if any(t[:1].isspace() for t in prior_tokens):
        spans = sorted({_word_span(prior_tokens, j) for j in selected})
        words = ["".join(prior_tokens[s:e + 1]) for s, e in spans]
    else:
        words = [prior_tokens[j] for j in selected]
    text = " ".join(w.strip() for w in words if w.strip())
    # Thinking tags are markup, not content; drop them from the query.
    return " ".join(_THINK_TAG.sub(" ", text).split())
