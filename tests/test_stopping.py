from gate_rlm.stopping import Checkpoint, StoppingPolicy, Thresholds, knee_ratio, replay


def cp(i, tokens, rerank=0.9, conf=0.9, found=True, answer="office", cosine=0.8):
    return Checkpoint(idx=i, cum_tokens=tokens, cosine=cosine, rerank=rerank,
                      conf=conf, found=found, answer=answer)


def test_sufficiency_stops_on_confident_relevant_answer():
    policy = StoppingPolicy(Thresholds(k_agree=1, knee_enabled=False))
    assert not policy.update(cp(0, 100, rerank=0.1, found=False, answer="")).stop
    d = policy.update(cp(1, 200))
    assert d.stop and d.reason == "sufficient" and d.answer == "office"


def test_low_confidence_or_irrelevant_votes_do_not_count():
    policy = StoppingPolicy(Thresholds(k_agree=1, knee_enabled=False))
    assert not policy.update(cp(0, 100, conf=0.3)).stop        # not confident
    assert not policy.update(cp(1, 200, rerank=0.2)).stop      # not relevant
    assert not policy.update(cp(2, 300, cosine=0.05)).stop     # fails cosine prefilter


def test_agreement_requires_k_matching_answers():
    policy = StoppingPolicy(Thresholds(k_agree=2, knee_enabled=False))
    assert not policy.update(cp(0, 100, answer="The office")).stop
    assert not policy.update(cp(1, 200, answer="kitchen")).stop
    assert policy.update(cp(2, 300, answer="office.")).stop    # normalized match


def test_knee_detects_flat_gain_curve():
    rho, _ = knee_ratio([1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    assert rho >= 6
    rho_steady, _ = knee_ratio([1, 2, 3, 4, 5, 6])
    assert rho_steady < 2


def test_knee_stop_via_policy():
    th = Thresholds(tau_hi=2.0, rho=4.0, min_checkpoints=4)  # sufficiency disabled
    policy = StoppingPolicy(th)
    decisions = [policy.update(cp(0, 100, conf=0.1))]
    decisions += [policy.update(cp(i, 100 * (i + 1), rerank=0.0, found=False, answer=""))
                  for i in range(1, 12)]
    assert any(d.stop and d.reason == "knee" for d in decisions)


def test_disabled_policy_never_stops_but_still_labels_relevance():
    policy = StoppingPolicy(Thresholds(), enabled=False)
    c = cp(0, 100)
    assert not policy.update(c).stop
    assert c.relevant is True


def test_replay_matches_live_policy():
    cps = [cp(0, 100, rerank=0.1, found=False, answer=""), cp(1, 250), cp(2, 400)]
    i, d = replay(cps, Thresholds(knee_enabled=False))
    assert i == 1 and d.answer == "office"
    assert cps[1].relevant is None  # replay must not mutate the logged checkpoints
