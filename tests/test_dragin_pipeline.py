"""Pipeline plumbing for DRAGIN-RLM, with harness.run_dragin mocked out -- no mlx
or model weights needed. attention_probe.py itself can only be exercised on real
Apple Silicon with mlx-lm + the model (see its module docstring); this checks
everything around it: config extraction and the result-record schema.
"""

import pytest

from dragin_rlm import pipeline
from gate_rlm.config import load_config


def example():
    return {
        "qid": "t1",
        "dataset": "browsecomp",
        "split": "test",
        "query": "How many people does the arena seat?",
        "context": "Some document text about an arena and its capacity.",
        "gold_answer": "3,677",
    }


def cfg_for_dragin():
    cfg = load_config("configs/experiments/dragin_rlm.yaml")
    return cfg


def test_dragin_config_from_cfg_reads_the_dragin_block():
    cfg = cfg_for_dragin()
    dcfg = pipeline.dragin_config_from_cfg(cfg)
    assert dcfg.model_path == "mlx-community/Qwen3.5-35B-A3B-4bit"
    assert dcfg.worker_model == "openai/qwen35-worker"
    assert dcfg.theta == 0.0001
    assert dcfg.top_n == 25


def test_run_example_success_path(monkeypatch):
    def fake_run_dragin(query, context, dcfg, model=None, tokenizer=None):
        return {
            "answer": "3,677",
            "raw_generation": "... So the answer is: 3,677",
            "latency_s": 1.23,
            "completion_tokens": 42,
            "llm_calls": 2,
            "leaf_calls": 1,
            "root_iterations": 2,
            "n_retrievals": 1,
            "rind_triggers": [{"index": 5, "query": "seating capacity", "score": 1.4}],
            "theta": 1.0,
            "top_n": 25,
        }

    monkeypatch.setattr(pipeline, "run_dragin", fake_run_dragin)
    rec = pipeline.run_example(example(), cfg_for_dragin(), seed=0)
    assert rec["error"] is None if "error" in rec else True
    assert rec["answer"] == "3,677"
    assert rec["config"] == "dragin_rlm"
    assert rec["mode"] == "dragin"
    assert rec["qid"] == "t1"
    assert rec["n_retrievals"] == 1
    assert rec["total_tokens"] == 42
    assert rec["context_tokens"] > 0


def test_run_example_failure_path_is_captured_not_raised(monkeypatch):
    def failing_run_dragin(query, context, dcfg, model=None, tokenizer=None):
        raise RuntimeError("mlx not available on this box")

    monkeypatch.setattr(pipeline, "run_dragin", failing_run_dragin)
    rec = pipeline.run_example(example(), cfg_for_dragin(), seed=0)
    assert rec["answer"] == ""
    assert "mlx not available" in rec["error"]
    assert rec["qid"] == "t1"  # record is still well-formed despite the failure
