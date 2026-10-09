"""_extract_answer is a pure function (no mlx/model dependency), so unlike
run_dragin() itself this can be exercised directly without real hardware.
"""

from dragin_rlm.harness import DraginConfig, _build_result, _extract_answer


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


def test_extract_answer_strips_end_of_text_token():
    # Real case from the smoke run: the cue answer came back as "Yes<|endoftext|>".
    answer, source = _extract_answer("So the answer is: Yes<|endoftext|>", worker_fallback="")
    assert answer == "Yes"
    assert source == "cue"


def _cfg():
    return DraginConfig(model_path="unused", worker_model="unused")


def _result(triggers):
    return _build_result(
        "", 0.0, 0, 1, triggers, False, _cfg(), 0.0, None, 0, 0, partial=False,
    )


def test_fallback_skips_empty_latest_worker_answer():
    # Regression: the last retrieval's worker reply was "" (thinking ate the
    # token budget), which used to make the whole answer empty even though an
    # earlier retrieval had a real worker answer.
    triggers = [
        {"worker_answer": "Fort Smith Museum of History"},
        {"worker_answer": ""},
    ]
    result = _result(triggers)
    assert result["answer"] == "Fort Smith Museum of History"
    assert result["answer_source"] == "worker_fallback"
    assert result["worker_answers"] == ["Fort Smith Museum of History", ""]


def test_worker_payload_disables_thinking(monkeypatch):
    # Without enable_thinking=False the worker writes its reply into `reasoning`
    # and `content` comes back empty.
    import io
    import json
    import urllib.request

    from dragin_rlm import harness

    sent = {}

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data.decode("utf-8")))
        body = json.dumps({"choices": [{"message": {"content": "Yes"}}]}).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    answer = harness._call_worker("unused", "query", "passages")
    assert answer == "Yes"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
