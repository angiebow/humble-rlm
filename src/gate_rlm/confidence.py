"""Worker confidence signals.

GPT-5 / GPT-5-mini are reasoning models and generally do not return token
log-probs, so the default signal is *verbalized* confidence in a structured
worker reply (P(True)-style, Kadavath et al. 2022). A log-prob signal is
available when a separate non-reasoning scorer model is configured.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Optional

WORKER_INSTRUCTION = """

Reply with ONLY a JSON object, no prose:
{"found": true|false, "answer": "<short answer, empty if not found>",
 "confidence": <0.0-1.0 probability your answer is correct>,
 "evidence": "<verbatim quote from the context that supports the answer, empty if none>"}
Set found=false if the context does not contain the information needed."""


@dataclass
class WorkerReply:
    found: bool
    answer: str
    confidence: float
    evidence: str
    parsed: bool          # False if the model ignored the JSON format
    raw: str


def parse_worker_reply(text: str) -> WorkerReply:
    """Parse the worker's JSON; degrade gracefully if the format is broken."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            conf = float(obj.get("confidence", 0.0))
            return WorkerReply(
                found=bool(obj.get("found", False)),
                answer=str(obj.get("answer", "") or "").strip(),
                confidence=min(1.0, max(0.0, conf)),
                evidence=str(obj.get("evidence", "") or "").strip(),
                parsed=True,
                raw=text,
            )
        except (ValueError, TypeError):
            pass
    stripped = (text or "").strip()
    return WorkerReply(
        found=bool(stripped),
        answer=stripped[:200],
        confidence=0.0,  # unknown confidence never counts as a vote
        evidence="",
        parsed=False,
        raw=text,
    )


def logprob_confidence(
    scorer_model: str, query: str, context: str, answer: str
) -> tuple[Optional[float], dict]:
    """Min token probability of a short re-answer from a log-prob-capable model.

    Returns (confidence, usage). This costs one extra call, which is added to the
    run's reported cost as auxiliary usage.
    """
    import litellm

    resp = litellm.completion(
        model=scorer_model,
        messages=[
            {"role": "system", "content": "Answer in as few words as possible."},
            {"role": "user", "content": f"Context:\n{context[:12000]}\n\nQuestion: {query}"},
        ],
        logprobs=True,
        max_tokens=16,
        temperature=0,
    )
    usage = dict(getattr(resp, "usage", {}) or {})
    try:
        tokens = resp.choices[0].logprobs["content"]
        probs = [math.exp(t["logprob"]) for t in tokens if t.get("token", "").strip()]
        return (min(probs) if probs else None), usage
    except (AttributeError, KeyError, TypeError, IndexError):
        return None, usage
