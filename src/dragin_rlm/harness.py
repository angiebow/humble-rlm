"""DRAGIN-RLM's generation loop: early-stop-if-confident, trigger-sub-call-if-
uncertain (DRAGIN, Su et al. 2024, adapted to this project's RLM setting).

Not a GateRLM subclass. GateRLM's hooks (_call_llm / _call_leaf, gate_rlm/gate.py)
sit at the boundary of whole LLM calls made through recursive-llm's litellm
plumbing -- RIND needs control INSIDE a single generation call, at the token
level, which that boundary doesn't expose. So this is its own loop, built
directly on attention_probe's synchronous mlx-lm wrapper.

Mapping the paper's algorithm onto this project's setting:
  - "Generate" = attention_probe.generate_with_probe on the root model, direct
    (non-REST) mlx-lm access, root turns.
  - "When to retrieve" = RIND (rind.RindState), checked online as each token is
    produced.
  - "What to retrieve" = QFS (qfs.format_query) over the triggering token's own
    attention row.
  - "Retrieval module" = retrieval.slice_context, BM25-lite over the document
    BrowseComp-Plus already assembled as this question's context (not an
    external index -- see retrieval.py) feeding a worker LLM sub-call (regular
    litellm/REST, like GateRLM's leaf calls -- the worker doesn't need attention
    introspection, it's just answering over an already-selected passage).
  - "Continue generation after retrieval" (Eq. 6: truncate T at t_i, inject
    retrieved passages, resume) = truncate generated_text and RindState at the
    trigger index, build a new prompt with the passages inserted, start a new
    generate_with_probe segment that continues from the truncated text.

Per-token decision, explicitly (three outcomes, one check):
  S_RIND(t_i) > theta   -> RETRIEVE. Truncate at t_i, QFS-slice the context,
                            worker sub-call, resume generation from t_i with
                            the retrieved passage injected.
  S_RIND(t_i) <= theta  -> CONTINUE. No action; the next token is generated
                            normally. This is the common case for every token
                            that isn't a trigger -- there is no per-token
                            "stop" branch distinct from it.
  natural end reached    -> STOP. EOS, the "So the answer is:" cue appears, or
                            the generate_length budget runs out -- whichever
                            comes first ends the loop and the response is
                            returned as-is. High confidence throughout
                            generation shows up as never triggering RETRIEVE,
                            which is exactly what lets this reach STOP without
                            ever invoking a worker sub-call -- there's no
                            separate proactive-early-stop path beyond that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import qfs, retrieval
from .attention_probe import generate_with_probe, load_dragin_model
from .rind import RindState

ANSWER_CUE = "So the answer is"

DIRECT_PROMPT = (
    "Answer the question using the document. Reason step by step, then end your "
    "answer with '{cue}: <short answer>'.\n\nDocument:\n{{context}}\n\n"
    "Question: {{query}}\nAnswer:"
).format(cue=ANSWER_CUE)

RETRIEVAL_TEMPLATE = (
    "Answer the question using the document. Reason step by step, then end your "
    "answer with '{cue}: <short answer>'.\n\nDocument:\n{{context}}\n\n"
    "Below are passages relevant to what you were about to say next -- use them "
    "if they help, and stay consistent with what you already wrote:\n{{passages}}\n\n"
    "Question: {{query}}\nAnswer:{{prefix}}"
).format(cue=ANSWER_CUE)

WORKER_PROMPT = (
    "Answer the question using only the passages below. Reply with the short "
    "answer only, or 'NOT FOUND' if the passages don't contain it.\n\n"
    "Passages:\n{passages}\n\nQuestion: {query}\nAnswer:"
)


@dataclass
class DraginConfig:
    model_path: str              # local mlx-community model id/path for the root
    worker_model: str            # litellm alias for the sub-call, e.g. openai/qwen35-worker
    theta: float = 1.0           # [SWEEP] RIND trigger threshold
    top_n: int = 25              # QFS query length (paper Table 9 range: 25-35)
    generate_length: int = 256   # total root tokens budget across all segments
    max_retrieval_seconds: float = 600.0  # wall-clock cap per retrieval cycle (root
                                  # generation to the trigger + worker sub-call +
                                  # next-prompt rebuild) -- the first retrieval cycle
                                  # that takes longer than this many seconds is treated
                                  # as the last one; no further triggers are allowed
                                  # after it.
    max_triggers: int = 20       # cap on total retrieval count per question; the paper
                                  # leaves this unbounded, but an all-fast-retrievals run
                                  # could otherwise go on indefinitely under the time cap
                                  # alone (observed: 24+ retrievals, still climbing, on a
                                  # single question -- see DRAGIN_RLM_TEST_RESULTS.md).
                                  # Whichever cap (this or max_retrieval_seconds) is hit
                                  # first stops triggering; either way nothing extra is
                                  # generated afterward -- whatever's in generated_text
                                  # at that point is recorded as-is.
    retrieval_top_k: int = 3     # passages handed to the worker per trigger
    passage_chars: int = 1000
    temperature: float = 0.0
    disable_thinking: bool = False  # render the prompt through the model's chat
                                  # template with enable_thinking=False (the
                                  # empty "<think></think>" block is part of the
                                  # prompt, not generated). Off = the original raw
                                  # "...Answer:" completion prompt, unchanged.


def _render_prompt(template: str, tokenizer: Any, disable_thinking: bool, prefix: str = "", **fields) -> str:
    """Fill ``template`` and return the text to feed the model. With thinking left
    alone (the default) this is exactly ``template.format(..., prefix=prefix)``,
    byte-for-byte what the harness always built. With ``disable_thinking`` the
    template minus ``prefix`` becomes the user turn of the model's chat template
    (enable_thinking=False), and ``prefix`` -- the already-generated text -- is
    appended after the assistant header so generation resumes mid-answer."""
    if not disable_thinking:
        return template.format(prefix=prefix, **fields)
    user_text = template.format(prefix="", **fields)
    head = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return head + prefix


def _call_worker(worker_model: str, query: str, passages: str) -> str:
    import json
    import os
    import urllib.request

    # Call the local mlx_lm.server directly instead of through litellm. Port 8005
    # by default; set DRAGIN_WORKER_URL when 8005 is already taken on the host.
    url = os.environ.get("DRAGIN_WORKER_URL", "http://localhost:8005/v1/chat/completions")
    payload = {
        "model": "mlx-community/Qwen3.5-2B-4bit",
        "messages": [
            {
                "role": "user",
                "content": WORKER_PROMPT.format(
                    passages=passages or "(nothing found)", query=query
                ),
            }
        ],
        "max_tokens": 128,
        "temperature": 0,
        # Without this the worker's answer goes into `reasoning` and `content`
        # comes back empty, so every retrieval produced an empty worker answer.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return (result.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def _extract_answer(text: str, worker_fallback: str = "") -> Tuple[str, str]:
    """Same "So the answer is" extraction convention the paper's own evaluation
    uses (Appendix B), reused here for BrowseComp-Plus's short-answer format.

    Returns (answer, source):
      "cue"            -- a real short answer followed the cue phrase.
      "raw_reasoning"  -- the root model wrote something but never reached
                           (or finished) the cue, e.g. a cap cut it off
                           mid-thought; the raw text is returned as-is.
      "worker_fallback" -- the root model produced nothing at all (every
                           retrieval truncated back to nothing, as happened
                           on a real run -- see DRAGIN_RLM_TEST_RESULTS.md),
                           so the most recent retrieval's own worker sub-call
                           answer is used instead. That worker call already
                           read real retrieved passages, which is still
                           better-grounded than an empty string.
    """
    if ANSWER_CUE in text:
        tail = text.split(ANSWER_CUE, 1)[1]
        extracted = tail.strip(" :\n").split("\n")[0].strip()
        if extracted:
            return extracted, "cue"
    stripped = text.strip()
    if stripped:
        return stripped, "raw_reasoning"
    return worker_fallback, "worker_fallback"


def _cue_line_complete(text: str) -> bool:
    """True once ANSWER_CUE has appeared AND a full line has been written after
    it. Stopping the instant the cue substring itself appears (the previous
    behavior) could cut generation off before the short answer was written --
    confirmed on a real run: generation ended with "...So the answer is" and
    nothing after it, so _extract_answer had nothing to return."""
    if ANSWER_CUE not in text:
        return False
    return "\n" in text.split(ANSWER_CUE, 1)[1]


def _build_result(
    generated_text: str,
    t0: float,
    total_new_tokens: int,
    segments_run: int,
    triggers: List[Dict[str, Any]],
    time_capped: bool,
    cfg: DraginConfig,
    max_rind_score: float,
    max_rind_detail: Optional[Dict[str, Any]],
    rind_checked_tokens: int,
    rind_nonstopword_tokens: int,
    partial: bool,
) -> Dict[str, Any]:
    """Shared by the final return and every checkpoint call so the two can
    never drift out of sync with each other."""
    # Most recent retrieval that actually produced a worker answer; an empty
    # worker reply on the last trigger should not hide an earlier good one.
    worker_fallback = next(
        (t["worker_answer"] for t in reversed(triggers) if t["worker_answer"]), ""
    )
    answer, answer_source = _extract_answer(generated_text, worker_fallback)
    return {
        "answer": answer,
        "answer_source": answer_source,
        "raw_generation": generated_text,
        "latency_s": time.perf_counter() - t0,
        "completion_tokens": total_new_tokens,
        "llm_calls": segments_run + len(triggers),
        "leaf_calls": len(triggers),
        "root_iterations": segments_run,
        "n_retrievals": len(triggers),
        "time_capped": time_capped,
        "rind_triggers": triggers,
        "worker_answers": [t["worker_answer"] for t in triggers],
        "theta": cfg.theta,
        "top_n": cfg.top_n,
        "max_rind_score": max_rind_score,
        "max_rind_detail": max_rind_detail,
        "rind_checked_tokens": rind_checked_tokens,
        "rind_nonstopword_tokens": rind_nonstopword_tokens,
        "partial": partial,
    }


def run_dragin(
    query: str,
    context: str,
    cfg: DraginConfig,
    model: Any = None,
    tokenizer: Any = None,
    on_checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Generate an answer to ``query`` over ``context`` with RIND-gated,
    QFS-sliced sub-calls. Returns a result dict compatible with gate_rlm's
    record schema (answer, latency_s, completion_tokens, llm_calls, leaf_calls,
    root_iterations) plus DRAGIN-specific fields (n_retrievals, rind_triggers).

    ``model``/``tokenizer`` can be passed in to reuse an already-loaded model
    across many examples (experiments/run_dragin.py does this -- loading a 35B
    model per example would be absurd); if omitted they're loaded fresh here.

    ``on_checkpoint``, if given, is called with the same-shaped result dict
    (``partial: True``) after every completed retrieval -- a single question
    at a low theta can run for a very long time and hold many retrievals'
    worth of real reasoning in memory with nothing on disk yet; killing the
    process before it reaches a natural stop previously lost all of it. This
    is the recovery point: whatever the last checkpoint call captured is the
    most that's recoverable if the process dies mid-question.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_dragin_model(cfg.model_path)

    t0 = time.perf_counter()
    prompt = _render_prompt(
        DIRECT_PROMPT, tokenizer, cfg.disable_thinking, context=context, query=query
    )
    state = RindState()
    generated_text = ""
    triggers: List[Dict[str, Any]] = []
    total_new_tokens = 0
    segments_run = 0
    budget = cfg.generate_length

    # Diagnostic: S_RIND is computed for every checkable token regardless of
    # whether it crosses theta, so tracking the max seen (and which of the
    # three factors it came from) is free -- no extra generation needed. Use
    # this to calibrate theta from real score distributions before trusting
    # any run's retrieval count: a threshold picked without knowing the real
    # range a model/task actually produces is indistinguishable from a
    # disabled trigger (see docs/PLAN.md / DRAGIN_RLM_TEST_RESULTS.md).
    max_rind_score = 0.0
    max_rind_detail: Optional[Dict[str, Any]] = None
    rind_checked_tokens = 0
    rind_nonstopword_tokens = 0

    time_capped = False
    while budget > 0 and len(triggers) < cfg.max_triggers:
        segments_run += 1
        seg_start = time.perf_counter()
        stream = generate_with_probe(
            model, tokenizer, prompt, max_tokens=budget, temperature=cfg.temperature
        )
        triggered_at: Optional[int] = None
        try:
            for probed in stream:
                idx = state.update(probed.text, probed.entropy, probed.attn_row)
                generated_text += probed.text
                total_new_tokens += 1
                budget -= 1
                if idx is not None:
                    score = state.score_at(idx)
                    rind_checked_tokens += 1
                    if state.semantic[idx]:
                        rind_nonstopword_tokens += 1
                    if score > max_rind_score:
                        max_rind_score = score
                        max_rind_detail = {
                            "token": state.tokens[idx],
                            "entropy": state.entropies[idx],
                            "max_attn": state.max_attn[idx],
                            "semantic": state.semantic[idx],
                        }
                    if score > cfg.theta:
                        triggered_at = idx
                        break
                if _cue_line_complete(generated_text) or budget <= 0:
                    break
        finally:
            stream.close()

        if triggered_at is None:
            break  # finished naturally: EOS, answer cue seen, or budget exhausted

        keep_chars = sum(len(t) for t in state.tokens[:triggered_at])
        generated_text = generated_text[:keep_chars]
        prior_tokens = state.tokens[:triggered_at]
        row = state.qfs_row(triggered_at)
        score = state.score_at(triggered_at)
        query_str = qfs.format_query(prior_tokens, row, cfg.top_n)
        state.reset_from(triggered_at)

        passages = retrieval.slice_context(
            context,
            query_str or query,
            top_k=cfg.retrieval_top_k,
            passage_chars=cfg.passage_chars,
        )
        worker_answer = _call_worker(cfg.worker_model, query_str or query, passages)
        triggers.append(
            {
                "index": triggered_at,
                "query": query_str,
                "score": score,
                "worker_answer": worker_answer,
            }
        )

        prompt = _render_prompt(
            RETRIEVAL_TEMPLATE,
            tokenizer,
            cfg.disable_thinking,
            context=context,
            passages=f"[1] {passages}\n(worker's read: {worker_answer})"
            if passages
            else f"(nothing found; worker's read: {worker_answer})",
            query=query,
            prefix=generated_text,
        )

        if on_checkpoint is not None:
            on_checkpoint(
                _build_result(
                    generated_text, t0, total_new_tokens, segments_run, triggers,
                    time_capped, cfg, max_rind_score, max_rind_detail,
                    rind_checked_tokens, rind_nonstopword_tokens, partial=True,
                )
            )

        if time.perf_counter() - seg_start > cfg.max_retrieval_seconds:
            time_capped = True
            break  # this retrieval cycle alone exceeded the cap -- stop here and
            # report whatever's been generated so far, no extra generation pass:
            # a retrieval that alone took this long means generated_text already
            # holds real, substantial reasoning (unlike a near-instant trigger),
            # so there's something worth recording as-is rather than spending
            # more time trying to force a clean "So the answer is" cue.

    return _build_result(
        generated_text, t0, total_new_tokens, segments_run, triggers, time_capped,
        cfg, max_rind_score, max_rind_detail, rind_checked_tokens,
        rind_nonstopword_tokens, partial=False,
    )
