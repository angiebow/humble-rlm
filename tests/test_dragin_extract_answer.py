"""_extract_answer is a pure function (no mlx/model dependency), so unlike
run_dragin() itself this can be exercised directly without real hardware.
"""

from dragin_rlm.harness import _extract_answer


def test_extract_answer_uses_the_cue_when_present():
    text = "Some reasoning.\nSo the answer is: Paris\n"
    answer, source = _extract_answer(text, worker_fallback="ignored")
    assert answer == "Paris"
    assert source == "cue"


def test_extract_answer_falls_back_to_raw_reasoning_without_a_cue():
    text = "Some unfinished reasoning that never reached a conclusion"
    answer, source = _extract_answer(text, worker_fallback="ignored")
    assert answer == text
    assert source == "raw_reasoning"


def test_extract_answer_falls_back_to_worker_answer_when_root_produced_nothing():
    # Real case: every retrieval truncated back to nothing (see
    # DRAGIN_RLM_TEST_RESULTS.md, browsecomp-279) -- generated_text is just
    # whitespace. The last retrieval's own worker sub-call answer, which at
    # least saw real retrieved passages, is used instead of an empty string.
    answer, source = _extract_answer("\n\n", worker_fallback="Fort Smith Museum of History")
    assert answer == "Fort Smith Museum of History"
    assert source == "worker_fallback"


def test_extract_answer_empty_with_no_fallback_available():
    answer, source = _extract_answer("   ", worker_fallback="")
    assert answer == ""
    assert source == "worker_fallback"
