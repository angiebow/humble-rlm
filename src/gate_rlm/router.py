"""Stage 0 router: answer directly, or recurse?

P(IK)-inspired (Kadavath et al. 2022). P(IK) itself asks whether a model knows an
answer from its own training; here the answer is in the document either way, so
the question becomes: can ONE direct read of this context answer this query
reliably? Signals, cheapest first:
  1. does the context fit comfortably in the root model's window?
  2. is the query simple (single fact) or complex (aggregation / multi-hop)?
  3. optional: the root model's self-assessed probability, from a context preview.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

COMPLEX_PATTERNS = [
    r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\blist (all|every)\b",
    r"\ball (the )?\w+ (that|who|which)\b", r"\bcompare\b", r"\bdifference\b",
    r"\bbefore\b", r"\bafter\b", r"\bboth\b", r"\bmost\b", r"\bleast\b",
    r"\baverage\b", r"\btotal\b", r"\beach\b", r"\bwhich of\b",
]


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def is_complex_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in COMPLEX_PATTERNS)


@dataclass
class RouteDecision:
    route: str               # "direct" | "rlm"
    reason: str
    context_tokens: int
    complex_query: bool
    p_direct: Optional[float] = None
    aux_usage: Optional[dict] = None


SELF_ASSESS_PROMPT = """You will be asked a question about a document of about {n_tokens} tokens.
Below are the question and the first part of the document.

Question: {query}

Document preview:
{preview}

If you read the ENTIRE document once, what is the probability (0.0-1.0) that you
would answer this question correctly? Reply with only the number."""


def route(query: str, context: str, cfg: dict) -> RouteDecision:
    rcfg = cfg.get("router", {})
    n_tokens = estimate_tokens(context)
    complex_q = is_complex_query(query)
    if not rcfg.get("enabled", True):
        return RouteDecision("rlm", "router disabled", n_tokens, complex_q)
    if n_tokens > rcfg.get("direct_max_tokens", 100_000):
        return RouteDecision("rlm", "context too long", n_tokens, complex_q)
    if not rcfg.get("use_self_assessment", False):
        if complex_q:
            return RouteDecision("rlm", "complex query", n_tokens, complex_q)
        return RouteDecision("direct", "fits and simple", n_tokens, complex_q)

    import litellm

    prompt = SELF_ASSESS_PROMPT.format(
        n_tokens=n_tokens, query=query, preview=context[: rcfg.get("preview_chars", 4000)]
    )
    resp = litellm.completion(
        model=cfg["models"]["root"], messages=[{"role": "user", "content": prompt}]
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\d*\.?\d+", text)
    p = min(1.0, max(0.0, float(m.group(0)))) if m else 0.0
    usage = dict(getattr(resp, "usage", {}) or {})
    usage["cost_usd"] = _safe_cost(resp)
    if p >= rcfg.get("tau_route", 0.7) and not complex_q:
        return RouteDecision("direct", "self-assessed", n_tokens, complex_q, p, usage)
    return RouteDecision("rlm", "self-assessed low", n_tokens, complex_q, p, usage)


def _safe_cost(resp) -> Optional[float]:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return None
