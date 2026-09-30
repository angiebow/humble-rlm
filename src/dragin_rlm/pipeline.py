"""Run one example under DRAGIN-RLM and return a gate_rlm-compatible result
record, so eval/aggregate.py works unmodified across both construction tracks.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

from gate_rlm.router import estimate_tokens

from .harness import DraginConfig, run_dragin


def dragin_config_from_cfg(cfg: dict) -> DraginConfig:
    d = cfg.get("dragin", {})
    return DraginConfig(
        model_path=d["model_path"],
        worker_model=cfg["models"]["worker"],
        theta=d.get("theta", 1.0),
        top_n=d.get("top_n", 25),
        generate_length=d.get("generate_length", 256),
        max_retrieval_seconds=d.get("max_retrieval_seconds", 600.0),
        retrieval_top_k=d.get("retrieval_top_k", 3),
        passage_chars=d.get("passage_chars", 1000),
        temperature=d.get("temperature", 0.0),
    )


def run_example(
    example: dict, cfg: dict, seed: int, model: Any = None, tokenizer: Any = None
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "config": cfg["name"],
        "mode": "dragin",
        "seed": seed,
        "qid": example["qid"],
        "dataset": example["dataset"],
        "split": example.get("split"),
        "length_bucket": example.get("length_bucket"),
        "complexity": example.get("complexity"),
        "query": example["query"],
        "gold_answer": example["gold_answer"],
        "context_tokens": example.get("context_tokens") or estimate_tokens(example["context"]),
        "models": {"root": cfg.get("dragin", {}).get("model_path"), **cfg["models"]},
    }
    try:
        dcfg = dragin_config_from_cfg(cfg)
        out = run_dragin(
            example["query"], example["context"], dcfg, model=model, tokenizer=tokenizer
        )
        record.update(out)
    except Exception as exc:  # keep going; failed runs are data too
        record["answer"] = ""
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=3)
    record["total_tokens"] = int(record.get("completion_tokens", 0) or 0)
    return record
