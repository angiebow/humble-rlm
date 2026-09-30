import math

from dragin_rlm.rind import RindState, entropy_from_distribution, rind_score, semantic_score


def test_entropy_is_zero_for_a_certain_distribution():
    # log-probs of a near-one-hot distribution: entropy -> 0
    logprobs = [0.0, -20.0, -20.0, -20.0]
    assert entropy_from_distribution(logprobs) < 1e-6


def test_entropy_is_maximal_for_a_uniform_distribution():
    v = 8
    uniform_logprobs = [-math.log(v)] * v
    h = entropy_from_distribution(uniform_logprobs)
    assert abs(h - math.log(v)) < 1e-6


def test_entropy_is_idempotent_under_raw_logits_or_logprobs():
    logits = [2.0, 0.5, -1.0, 3.0]
    from_logits = entropy_from_distribution(logits)
    # log-softmax(logits) is already-normalized log-probs; re-normalizing is a no-op
    m = max(logits)
    shifted = [x - m for x in logits]
    log_z = m + math.log(sum(math.exp(s) for s in shifted))
    logprobs = [x - log_z for x in logits]
    from_logprobs = entropy_from_distribution(logprobs)
    assert abs(from_logits - from_logprobs) < 1e-9


def test_stopwords_get_zero_semantic_score():
    assert semantic_score("the") == 0
    assert semantic_score(" The") == 0
    assert semantic_score(",") == 0  # punctuation-only: no semantic content either
    assert semantic_score("Colisée") == 1


def test_rind_score_is_the_product():
    assert rind_score(entropy=2.0, max_attn=0.5, semantic=1) == 1.0
    assert rind_score(entropy=2.0, max_attn=0.5, semantic=0) == 0.0  # stopword -> always 0


def test_rind_state_checkability_is_one_step_lookahead():
    state = RindState()
    # first token: nothing precedes it, not checkable yet
    assert state.update("Hello", entropy=1.0, attn_row=[]) is None
    # second token's attention row covers position 0 -> position 0 now checkable
    idx = state.update(" world", entropy=1.0, attn_row=[0.9])
    assert idx == 0
    assert state.max_attn[0] == 0.9


def test_rind_state_max_attn_only_increases():
    state = RindState()
    state.update("a", entropy=1.0, attn_row=[])
    state.update("b", entropy=1.0, attn_row=[0.3])
    assert state.max_attn[0] == 0.3
    state.update("c", entropy=1.0, attn_row=[0.1, 0.5])
    # position 0's max shouldn't drop even though this row's value (0.1) is lower
    assert state.max_attn[0] == 0.3
    assert state.max_attn[1] == 0.5


def test_rind_state_qfs_row_is_generation_time_not_retrospective():
    state = RindState()
    state.update("The", entropy=1.0, attn_row=[])
    state.update(" cat", entropy=1.0, attn_row=[0.7])  # " cat"'s own row over ["The"]
    state.update(" sat", entropy=1.0, attn_row=[0.1, 0.9])
    assert state.qfs_row(1) == [0.7]
    assert state.qfs_row(2) == [0.1, 0.9]
    # max_attn[0] aggregates incoming attention from LATER tokens (retrospective),
    # a different array from qfs_row (each token's own outgoing row) -- both were
    # updated by the same calls but track different things.
    assert state.max_attn[0] == max(0.7, 0.1)


def test_reset_from_drops_the_trigger_position_and_everything_after():
    state = RindState()
    for i, tok in enumerate(["a", "b", "c", "d"]):
        row = [0.1 * j for j in range(i)]
        state.update(tok, entropy=float(i), attn_row=row)
    state.reset_from(2)
    assert state.tokens == ["a", "b"]
    assert len(state.entropies) == 2
    assert len(state.max_attn) == 2
    assert len(state.own_attn_rows) == 2


def test_score_at_uses_current_state():
    state = RindState()
    state.update("The", entropy=0.0, attn_row=[])
    idx = state.update(" Colisée", entropy=3.0, attn_row=[0.8])
    assert idx == 0
    # position 0 ("The") is a stopword -> score is 0 regardless of entropy/attn
    assert state.score_at(0) == 0.0
