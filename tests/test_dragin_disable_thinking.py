"""disable_thinking only changes how the prompt is rendered, and only when asked;
the default path must stay byte-for-byte what the harness always built."""

import re

from dragin_rlm import harness, pipeline
from gate_rlm.config import load_config


class FakeTok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        tail = "<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n{tail}"


FIELDS = dict(context="DOC", query="Q?")
RFIELDS = dict(context="DOC", query="Q?", passages="[1] P\n(worker's read: x)")


def test_default_is_identical_to_old_formatting():
    assert harness._render_prompt(harness.DIRECT_PROMPT, FakeTok(), False, **FIELDS) == \
        harness.DIRECT_PROMPT.format(**FIELDS)
    assert harness._render_prompt(harness.RETRIEVAL_TEMPLATE, FakeTok(), False, prefix="so far", **RFIELDS) == \
        harness.RETRIEVAL_TEMPLATE.format(prefix="so far", **RFIELDS)


def test_disabled_puts_empty_think_block_in_prompt_and_prefix_after_it():
    p = harness._render_prompt(harness.RETRIEVAL_TEMPLATE, FakeTok(), True, prefix="so far", **RFIELDS)
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nso far")
    assert "Question: Q?\nAnswer:<|im_end|>" in p  # prefix is NOT inside the user turn
    d = harness._render_prompt(harness.DIRECT_PROMPT, FakeTok(), True, **FIELDS)
    assert d.endswith("<think>\n\n</think>\n\n") and "so far" not in d


def test_config_default_off_and_round_regex():
    cfg = load_config("configs/experiments/dragin_rlm.yaml")
    assert pipeline.dragin_config_from_cfg(cfg).disable_thinking is False  # config alone never turns it on
    assert cfg["dragin"]["disable_thinking_from_round"] == 2
    get = lambda n: int(re.search(r"round(\d+)", n).group(1))
    assert get("dragin_rlm__browsecomp_batch10b_round12_theta0.001_local.jsonl") == 12
    assert re.search(r"round(\d+)", "dragin_rlm__browsecomp_sample50.jsonl") is None
