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
    assert dcfg.model_path == "mlx-community/Qwen3.5-4B-4bit"
    assert dcfg.worker_model == "openai/qwen35-worker"
    assert dcfg.theta == 0.001
    assert dcfg.top_n == 25
    assert dcfg.max_retrieval_seconds == 600.0
    assert dcfg.max_triggers == 20


def test_run_example_success_path(monkeypatch):
    def fake_run_dragin(query, context, dcfg, model=None, tokenizer=None, on_checkpoint=None):
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
    def failing_run_dragin(query, context, dcfg, model=None, tokenizer=None, on_checkpoint=None):
        raise RuntimeError("mlx not available on this box")

    monkeypatch.setattr(pipeline, "run_dragin", failing_run_dragin)
    rec = pipeline.run_example(example(), cfg_for_dragin(), seed=0)
    assert rec["answer"] == ""
    assert "mlx not available" in rec["error"]
    assert rec["qid"] == "t1"  # record is still well-formed despite the failure


def test_checkpoint_written_per_retrieval_and_merged_with_record(tmp_path):
    ckpt_path = str(tmp_path / "run.checkpoint.json")

    def run_dragin_with_checkpoints(query, context, dcfg, model=None, tokenizer=None, on_checkpoint=None):
        # Simulate two retrievals, each checkpointing before the question finishes --
        # this is what a kill between them should leave recoverable on disk. Shaped
        # like a real harness.py partial (same keys _build_result always includes),
        # since pipeline.py's on_checkpoint prints a progress line off these fields.
        on_checkpoint({"answer": "", "n_retrievals": 1, "completion_tokens": 10, "latency_s": 5.0, "partial": True})
        on_checkpoint({"answer": "", "n_retrievals": 2, "completion_tokens": 20, "latency_s": 9.0, "partial": True})
        return {"answer": "3,677", "n_retrievals": 2, "completion_tokens": 25, "latency_s": 12.0, "partial": False}

    import json
    from unittest import mock

    with mock.patch("dragin_rlm.pipeline.run_dragin", run_dragin_with_checkpoints):
        rec = pipeline.run_example(example(), cfg_for_dragin(), seed=0, checkpoint_path=ckpt_path)

    # The final record is unaffected by checkpointing.
    assert rec["answer"] == "3,677"
    # But the checkpoint file was written, carries the question's own identity
    # (qid/gold_answer from `record`, not just harness fields), and reflects
    # the LAST checkpoint call, not the first.
    ckpt = json.loads(open(ckpt_path).read())
    assert ckpt["qid"] == "t1"
    assert ckpt["gold_answer"] == "3,677"
    assert ckpt["n_retrievals"] == 2
    assert ckpt["partial"] is True
