"""Stage 0 router: answer directly, or recurse?

P(IK)-inspired (Kadavath et al. 2022). P(IK) itself asks whether a model knows an
answer from its own training; here the answer is in the document either way, so
the question becomes: can ONE direct read of this context answer this query
reliably? Three interchangeable estimators, selected by router.method:
  rule        cheapest first: does the context fit? is the query simple or
              complex (regex)? (default; no extra LLM call)
  verbalized  the root model self-rates P(direct succeeds) in words, from a
              context preview (router.use_self_assessment)
  logprob     a short candidate answer is generated on a context preview with
              logprobs=True; P(IK) = exp(mean log P(token)) over its tokens
              (Kadavath et al. 2022, Eq. logit-based probing). If P(IK) >= tau,
              that candidate is reused as the final answer -- no second call.
"""

from __future__ import annotations

import math
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
    direct_answer: Optional[str] = None  # reuse the probe's own generation, skip a 2nd call


SELF_ASSESS_PROMPT = """You will be asked a question about a document of about {n_tokens} tokens.
Below are the question and the first part of the document.

Question: {query}

Document preview:
{preview}

If you read the ENTIRE document once, what is the probability (0.0-1.0) that you
would answer this question correctly? Reply with only the number."""

LOGPROB_PROBE_PROMPT = """Question: {query}

Document preview (of about {n_tokens} tokens total):
{preview}

Answer the question as best you can from this preview alone. Reply with the
short answer only, no explanation."""


def _logprob_pik(query: str, context: str, n_tokens: int, cfg: dict) -> RouteDecision:
    """Kadavath et al. 2022, logit-based P(IK): generate a candidate answer on a
    context preview with logprobs=True, P(IK) = exp(mean log P(token)). If
    P(IK) >= tau_route, that candidate is reused as the final direct answer so
    the route decision costs no extra generation beyond this one probe call.
    """
    import litellm

    rcfg = cfg["router"]
    complex_q = is_complex_query(query)
    prompt = LOGPROB_PROBE_PROMPT.format(
        query=query, n_tokens=n_tokens, preview=context[: rcfg.get("preview_chars", 4000)]
    )
    resp = litellm.completion(
        model=cfg["models"]["root"],
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        max_tokens=rcfg.get("logprob_probe_max_tokens", 64),
        temperature=0,
    )
    usage = dict(getattr(resp, "usage", {}) or {})
    usage["cost_usd"] = _safe_cost(resp)
    answer = (resp.choices[0].message.content or "").strip()
    try:
        tokens = resp.choices[0].logprobs["content"]
        logprobs = [t["logprob"] for t in tokens if t.get("token", "").strip()]
        p_ik = math.exp(sum(logprobs) / len(logprobs)) if logprobs else 0.0
    except (AttributeError, KeyError, TypeError, IndexError, ZeroDivisionError):
        p_ik = 0.0

    tau = rcfg.get("tau_route", 0.7)
    if p_ik >= tau and not complex_q and answer:
        return RouteDecision("direct", "logprob P(IK) high", n_tokens, complex_q,
                              p_ik, usage, direct_answer=answer)
    return RouteDecision("rlm", "logprob P(IK) low", n_tokens, complex_q, p_ik, usage)


def route(query: str, context: str, cfg: dict) -> RouteDecision:
    rcfg = cfg.get("router", {})
    n_tokens = estimate_tokens(context)
    complex_q = is_complex_query(query)
    if not rcfg.get("enabled", True):
        return RouteDecision("rlm", "router disabled", n_tokens, complex_q)
    if n_tokens > rcfg.get("direct_max_tokens", 100_000):
        return RouteDecision("rlm", "context too long", n_tokens, complex_q)

    method = rcfg.get("method", "rule")
    if method == "logprob":
        return _logprob_pik(query, context, n_tokens, cfg)
    if method != "verbalized" and not rcfg.get("use_self_assessment", False):
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
