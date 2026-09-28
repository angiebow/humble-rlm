from eval.metrics import contains_match, exact_match, f1, is_correct, over_read_ratio
from eval.stats import mcnemar_exact, paired_bootstrap
from gate_rlm.confidence import parse_worker_reply
from gate_rlm.data import align_facts, assign_split, babi_facts, babilong_gold_facts
from gate_rlm.relevance import KeywordScorer, split_passages
from gate_rlm.router import is_complex_query, route

FACTS_0K = ("Mary moved to the bathroom. John went to the hallway. "
            "Mary travelled to the office. Daniel picked up the football. "
            "Daniel went to the garden.")


def test_babilong_qa1_gold_is_last_mention():
    facts = babi_facts(FACTS_0K)
    idx, ok = babilong_gold_facts("qa1", "Where is Mary?", facts, "office")
    assert ok and facts[idx[0]] == "Mary travelled to the office."


def test_babilong_qa2_gold_is_pickup_plus_location():
    facts = babi_facts(FACTS_0K)
    idx, ok = babilong_gold_facts("qa2", "Where is the football?", facts, "garden")
    assert ok and [facts[i] for i in idx] == [
        "Daniel picked up the football.", "Daniel went to the garden."]


def test_align_facts_in_noisy_context():
    facts = babi_facts(FACTS_0K)
    noisy = "Once upon a time. " + " Some noise here. ".join(facts) + " The end."
    pos = align_facts(noisy, facts)
    assert pos is not None and all(noisy[p:p + len(f)] == f for p, f in zip(pos, facts))
    assert align_facts("unrelated text", facts) is None


def test_split_is_deterministic():
    assert assign_split("babilong-qa1-7") == assign_split("babilong-qa1-7")


def test_metrics():
    assert exact_match("The Office.", "office")
    assert contains_match("Mary is in the office now", "office")
    assert not contains_match("officers", "office")
    assert abs(f1("in the office", "office") - 2 / 3) < 1e-9
    assert is_correct("office", "office", "babilong")
    assert over_read_ratio(1000, 250) == 0.75
    assert over_read_ratio(1000, None) is None


def test_stats():
    assert mcnemar_exact([1, 1, 1, 0], [1, 1, 1, 0])["p_value"] == 1.0
    ci = paired_bootstrap([1, 2, 3, 4], [1, 2, 3, 4])["ci95"]
    assert ci == [0.0, 0.0]


def test_worker_reply_parsing():
    r = parse_worker_reply('Sure: {"found": true, "answer": "office", "confidence": 0.9, "evidence": "x"}')
    assert r.parsed and r.found and r.answer == "office" and r.confidence == 0.9
    bad = parse_worker_reply("The office, I think.")
    assert not bad.parsed and bad.confidence == 0.0


def test_router_rules():
    cfg = {"router": {"enabled": True, "direct_max_tokens": 1000}, "models": {"root": "x"}}
    assert route("Where is Mary?", "short doc", cfg).route == "direct"
    assert route("How many times did Mary move?", "short doc", cfg).route == "rlm"
    assert route("Where is Mary?", "word " * 5000, cfg).route == "rlm"
    assert is_complex_query("Compare the two reports")


def test_keyword_scorer_and_passages():
    s = KeywordScorer(passage_chars=40)
    hi = s.score("Where did Mary travel?", "Mary travelled to the office. Mary went home.")
    lo = s.score("Where did Mary travel?", "The weather was cold and grey all winter.")
    assert hi.rerank > lo.rerank
    assert len(split_passages("a. " * 100, 20)) > 1
