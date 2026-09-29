"""Run one example under one configuration and return a result record.

Every condition (direct, vanilla, observe, gate, ablations) writes the same
record schema (docs/RESULT_SCHEMA.md) so evaluation code never branches on mode.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, Optional

from .router import RouteDecision, estimate_tokens, route

DIRECT_SYSTEM = (
    "Answer the question using only the document. Reply with the short answer only."
)


def _cost(resp) -> Optional[float]:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return None


def run_direct(query: str, context: str, cfg: dict) -> Dict[str, Any]:
    import litellm

    t0 = time.perf_counter()
    resp = litellm.completion(
        model=cfg["models"]["root"],
        messages=[
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": f"Document:\n{context}\n\nQuestion: {query}"},
        ],
    )
    usage = getattr(resp, "usage", None)
    return {
        "answer": (resp.choices[0].message.content or "").strip(),
        "latency_s": time.perf_counter() - t0,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cost_usd": _cost(resp),
        "llm_calls": 1,
        "leaf_calls": 0,
        "root_iterations": 1,
    }


def run_rlm(example: dict, cfg: dict, scorer) -> Dict[str, Any]:
    from .gate import GateRLM

    r = cfg["rlm"]
    kwargs = {k: v for k, v in r.items() if v is not None}
    rlm = GateRLM(
        cfg=cfg,
        user_query=example["query"],
        scorer=scorer if cfg["mode"] in ("gate", "observe") else None,
        gold=example.get("gold_fingerprints"),
        gold_mode=example.get("gold_mode", "any"),
        evidence=example.get("evidence_fingerprints"),
        model=cfg["models"]["root"],
        recursive_model=cfg["models"]["worker"],
        capture_trajectory_content=cfg["logging"].get("capture_trajectory_content", True),
        **kwargs,
    )
    t0 = time.perf_counter()
    run = rlm.try_complete_result(query=example["query"], context=example["context"])
    latency = time.perf_counter() - t0
    stats = run.stats
    aux_cost = sum((u.get("cost_usd") or 0.0) for u in rlm.aux_usage)
    return {
        "answer": (run.answer or "") if run.succeeded else "",
        "error": None if run.succeeded else f"{run.error_type}: {run.error}",
        "latency_s": latency,
        "prompt_tokens": stats.get("prompt_tokens", 0),
        "completion_tokens": stats.get("completion_tokens", 0),
        "cost_usd": (stats.get("estimated_cost_usd") or 0.0) + aux_cost,
        "priced_calls": stats.get("priced_calls"),
        "llm_calls": stats.get("llm_calls", 0),
        "leaf_calls": stats.get("leaf_calls", 0),
        "root_iterations": stats.get("total_iterations", 0),
        "by_model": stats.get("by_model", {}),
        **rlm.gate_report(),
    }


def run_example(example: dict, cfg: dict, scorer, seed: int) -> Dict[str, Any]:
    """Route (if enabled), execute, and package one result record."""
    record: Dict[str, Any] = {
        "config": cfg["name"],
        "mode": cfg["mode"],
        "seed": seed,
        "qid": example["qid"],
        "dataset": example["dataset"],
        "split": example.get("split"),
        "length_bucket": example.get("length_bucket"),
        "complexity": example.get("complexity"),
        "query": example["query"],
        "gold_answer": example["gold_answer"],
        "context_tokens": example.get("context_tokens") or estimate_tokens(example["context"]),
        "models": cfg["models"],
    }
    try:
        if cfg["mode"] == "direct":
            decision = RouteDecision("direct", "baseline", record["context_tokens"], False)
        elif cfg["mode"] == "gate":
            decision = route(example["query"], example["context"], cfg)
        else:
            decision = RouteDecision("rlm", "baseline", record["context_tokens"], False)
        record["route"] = decision.route
        record["route_reason"] = decision.reason
        record["p_direct"] = decision.p_direct

        if decision.route == "direct" and decision.direct_answer is not None:
            # logprob P(IK) probe already generated the answer; no 2nd call.
            usage = decision.aux_usage or {}
            out = {
                "answer": decision.direct_answer,
                "latency_s": 0.0,
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "cost_usd": usage.get("cost_usd"),
                "llm_calls": 1,
                "leaf_calls": 0,
                "root_iterations": 1,
            }
            decision.aux_usage = None  # already folded into out, don't double-count below
        elif decision.route == "direct":
            out = run_direct(example["query"], example["context"], cfg)
        else:
            out = run_rlm(example, cfg, scorer)
        if decision.aux_usage:  # router self-assessment is part of the method's cost
            out["cost_usd"] = (out.get("cost_usd") or 0.0) + (decision.aux_usage.get("cost_usd") or 0.0)
            out["router_tokens"] = int(decision.aux_usage.get("total_tokens", 0) or 0)
        record.update(out)
    except Exception as exc:  # keep going; failed runs are data too
        record["answer"] = ""
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=3)
    record["total_tokens"] = int(record.get("prompt_tokens", 0) or 0) + int(
        record.get("completion_tokens", 0) or 0
    ) + int(record.get("router_tokens", 0) or 0)
    return record
