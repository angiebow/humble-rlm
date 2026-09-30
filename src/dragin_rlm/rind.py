"""RIND: Real-time Information Needs Detection (DRAGIN, Su et al. 2024, §3.1).

    S_RIND(t_i) = H_i . a_max(i) . s_i                                    (Eq. 5)
      H_i       token entropy over the full vocabulary at position i      (Eq. 1)
      a_max(i)  max attention any LATER token pays back to t_i, last
                full-attention transformer layer                         (Eq. 2-3)
      s_i       1 unless t_i is a stopword                                (Eq. 4)

Retrieval triggers the first time any already-generated token's score exceeds a
threshold theta.

a_max(i) is retrospective: Eq. 3 defines it as max over j>i of the attention FROM
t_j back TO t_i, which by construction can't be known until at least one token
after t_i exists. The paper (an offline, dataset-benchmark setting) doesn't need
to say when exactly to check a position online; for streaming generation we do,
so this module makes it a real policy: a position becomes "checkable" the first
time it receives an attention update from the very next token, and is checked
then, using whatever a_max it has at that moment (which can only be a lower bound
on the true retrospective max — later tokens beyond that one could in principle
push it higher, but a single-step lookahead is what makes online triggering
possible at all, and matches the paper's own Figure 1, which shows RIND firing on
the most recently completed token as generation proceeds).

RIND itself operates on raw generated subword tokens (whatever attention_probe.py
yields), not whitespace-split words -- see stopwords.py for how the semantic mask
handles that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .stopwords import is_stopword


def entropy_from_distribution(logits_or_logprobs: Sequence[float]) -> float:
    """H_i = -sum_v p(v) log p(v) (Eq. 1), computed stably via log-softmax.

    Accepts either raw logits or an already-normalized log-probability vector
    (e.g. mlx_lm.generate_step's ``logprobs``): log-softmax is idempotent, so
    log-softmax(log_softmax(x)) recovers the same distribution either way.
    """
    if len(logits_or_logprobs) == 0:
        return 0.0
    m = max(logits_or_logprobs)
    shifted = [x - m for x in logits_or_logprobs]
    log_z = m + math.log(sum(math.exp(s) for s in shifted))
    log_p = [x - log_z for x in logits_or_logprobs]
    return -sum(math.exp(lp) * lp for lp in log_p)


def semantic_score(token_text: str) -> int:
    """s_i in Eq. 4: 0 for stopwords (or punctuation-only tokens), 1 otherwise."""
    return 0 if is_stopword(token_text) else 1


def rind_score(entropy: float, max_attn: float, semantic: int) -> float:
    """S_RIND(t_i) = H_i . a_max(i) . s_i  (Eq. 5)."""
    return entropy * max_attn * semantic


@dataclass
class RindState:
    """Online accumulator for RIND over one generation stream.

    Also retains each token's own generation-time attention row
    (``own_attn_rows``), which RIND itself doesn't need but QFS does (§3.2: QFS
    ranks the SAME row -- t_i's attention back over {t_{i-1},...,t_1} at the
    moment t_i was generated -- not the retrospective max RIND tracks).
    """

    tokens: List[str] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    semantic: List[int] = field(default_factory=list)
    max_attn: List[float] = field(default_factory=list)
    own_attn_rows: List[List[float]] = field(default_factory=list)

    def update(
        self, token_text: str, entropy: float, attn_row: Sequence[float]
    ) -> Optional[int]:
        """Record the newly generated token's own signals, fold its attention row
        into every earlier position's running max, and return the index that just
        became checkable (n-1), or None for the very first token (nothing
        precedes it, so it has no a_max(i) yet).

        ``attn_row[j]`` is this token's attention back to position j, for
        j in [0, n) -- i.e. exactly what attention_probe.ProbedToken.attn_row is.
        """
        self.tokens.append(token_text)
        self.entropies.append(entropy)
        self.semantic.append(semantic_score(token_text))
        self.max_attn.append(0.0)
        self.own_attn_rows.append(list(attn_row))
        n = len(self.tokens) - 1
        for i, a in enumerate(attn_row):
            if i < n and a > self.max_attn[i]:
                self.max_attn[i] = a
        return n - 1 if n >= 1 else None

    def score_at(self, i: int) -> float:
        return rind_score(self.entropies[i], self.max_attn[i], self.semantic[i])

    def qfs_row(self, i: int) -> List[float]:
        """t_i's own attention to {t_0, ..., t_{i-1}} -- what QFS ranks (§3.2)."""
        return self.own_attn_rows[i]

    def reset_from(self, keep: int) -> None:
        """Drop everything from position ``keep`` onward. Used after a RIND
        trigger: DRAGIN truncates the output at the triggering position (Eq. 6)
        and regenerates from there once retrieved context is injected, so
        positions before it are never revisited."""
        self.tokens = self.tokens[:keep]
        self.entropies = self.entropies[:keep]
        self.semantic = self.semantic[:keep]
        self.max_attn = self.max_attn[:keep]
        self.own_attn_rows = self.own_attn_rows[:keep]
