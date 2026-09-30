"""Stopword filter for RIND's semantic indicator s_i (paper Eq. 4).

The paper uses spaCy's ``en_core_web_sm`` stopword list. We don't add spaCy as a
dependency for ~200 words, so this is a static, documented substitute: the
standard NLTK/scikit-learn "core" English function-word list, lowercased. It will
disagree with spaCy at the margins (a handful of words either list treats
differently), which is worth knowing if RIND trigger counts are ever compared
directly against the paper's own numbers -- they're close, not identical.

RIND operates on raw generated subword tokens (see rind.py / attention_probe.py),
not whitespace-split words, so ``is_stopword`` normalizes punctuation/case/leading
subword markers before checking membership.
"""

from __future__ import annotations

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself let's me more most mustn't my
    myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some
    such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under
    until up very was wasn't we we'd we'll we're we've were weren't what
    what's when when's where where's which while who who's whom why why's
    with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves
    """.split()
)

_STRIP_CHARS = " \t\n\r.,!?;:\"'()[]{}<>"


def is_stopword(token_text: str) -> bool:
    """True if ``token_text`` (a generated token, possibly a bare subword piece
    with a leading '▁'/'Ġ' marker) is a stopword or has no letters at all
    (punctuation-only tokens carry no semantic content either, s_i = 0)."""
    cleaned = token_text.strip(_STRIP_CHARS).lstrip("▁Ġ").lower()
    if not cleaned:
        return True
    if not any(c.isalpha() for c in cleaned):
        return True
    return cleaned in STOPWORDS
