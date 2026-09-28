"""End-to-end GateRLM run against a scripted fake model (no API key, no cost).

The fake root reads the document in 4 chunks through llm_query_batched, then
answers. The worker finds the answer only in chunk 2. We check that:
  * gate mode stops once confident, relevant evidence arrives and skips later calls,
  * gold evidence is logged (for the over-read ratio) but never drives decisions,
  * vanilla mode logs checkpoints without scoring or gating.
"""

import json
from types import SimpleNamespace

import pytest

rlm = pytest.importorskip("rlm")
import litellm  # noqa: E402

from gate_rlm.config import load_config  # noqa: E402
from gate_rlm.pipeline import run_example  # noqa: E402
from gate_rlm.relevance import KeywordScorer  # noqa: E402

NOISE = "The river was quiet and the hills were green. " * 8
CHUNKS = [NOISE, NOISE + "Mary travelled to the office. " + NOISE, NOISE, NOISE]
CONTEXT = "|||".join(CHUNKS)
ROOT_CODE = (
    "for c in context.split('|||'):\n"
    "    print(llm_query(query, c))"
)


def fake_response(text):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                            prompt_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=usage, _hidden_params={})


async def fake_acompletion(model, messages, **kwargs):
    system = messages[0]["content"]
    if system.startswith("Answer the subproblem"):  # worker
        body = messages[-1]["content"]
        if "Mary travelled to the office" in body:
            return fake_response(json.dumps(
                {"found": True, "answer": "office", "confidence": 0.95,
                 "evidence": "Mary travelled to the office."}))
        return fake_response('{"found": false, "answer": "", "confidence": 0.9, "evidence": ""}')
    # root: first turn runs code; afterwards finalize
    if sum(m["role"] == "assistant" for m in messages) == 0:
        return fake_response(ROOT_CODE)
    return fake_response('FINAL("office")')


@pytest.fixture(autouse=True)
def patch_litellm(monkeypatch):
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)


def example():
    return {"qid": "t1", "dataset": "babilong", "query": "Where did Mary travel?",
            "context": CONTEXT, "gold_answer": "office",
            "gold_fingerprints": [["Mary travelled to the office."]], "gold_mode": "all",
            "evidence_fingerprints": [["Mary travelled to the office."]]}


def cfg_for(mode):
    cfg = load_config()
    cfg.update({"name": mode, "mode": mode})
    cfg["router"]["enabled"] = False
    cfg["models"] = {"root": "fake-root", "worker": "fake-worker"}
    cfg["relevance"]["tau_cos"] = 0.0
    cfg["relevance"]["tau_rel"] = 0.3
    cfg["stopping"]["tau_hi"] = 0.3
    return cfg


def test_gate_mode_stops_and_skips_later_calls():
    rec = run_example(example(), cfg_for("gate"), KeywordScorer(), seed=0)
    assert rec.get("error") is None, rec.get("traceback")
    assert "office" in rec["answer"]
    assert rec["stop_reason"] == "sufficient"
    assert rec["worker_calls_skipped"] >= 1        # chunks after the stop cost nothing
    assert rec["gold_seen_at_tokens"] is not None
    assert any(c["contains_gold"] for c in rec["checkpoints"])


def test_observe_mode_logs_signals_without_acting():
    rec = run_example(example(), cfg_for("observe"), KeywordScorer(), seed=0)
    assert rec.get("error") is None, rec.get("traceback")
    assert rec["stop_reason"] is None and rec["worker_calls_skipped"] == 0
    assert len(rec["checkpoints"]) == 4
    assert all(c["rerank"] is not None for c in rec["checkpoints"])


def test_vanilla_mode_logs_evidence_only():
    rec = run_example(example(), cfg_for("vanilla"), None, seed=0)
    assert rec.get("error") is None, rec.get("traceback")
    assert len(rec["checkpoints"]) == 4
    assert all(c["rerank"] is None for c in rec["checkpoints"])
