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
from typing import Any, Dict, List, Optional, Tuple

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
                                  # next-prompt rebuild). No cap on retrieval COUNT --
                                  # the paper leaves that unbounded too -- but the first
                                  # retrieval cycle that takes longer than this many
                                  # seconds is treated as the last one; no further
                                  # triggers are allowed after it.
    min_answer_tokens: int = 128 # guaranteed, trigger-immune budget for the final
                                  # answer pass once the retrieval-time cap is hit --
                                  # tops up whatever's left of generate_length if it's
                                  # smaller than this. Without it, a low enough theta
                                  # makes every segment (including the last) get cut
                                  # off after a couple of tokens, so the run ends with
                                  # no answer at all regardless of remaining budget
                                  # (see DRAGIN_RLM_TEST_RESULTS.md, theta=0.0001).
    retrieval_top_k: int = 3     # passages handed to the worker per trigger
    passage_chars: int = 1000
    temperature: float = 0.0


def _call_worker(worker_model: str, query: str, passages: str) -> str:
    import litellm

    resp = litellm.completion(
        model=worker_model,
        messages=[
            {
                "role": "user",
                "content": WORKER_PROMPT.format(
                    passages=passages or "(nothing found)", query=query
                ),
            }
        ],
        max_tokens=128,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_answer(text: str) -> str:
    """Same "So the answer is" extraction convention the paper's own evaluation
    uses (Appendix B), reused here for BrowseComp-Plus's short-answer format."""
    if ANSWER_CUE in text:
        tail = text.split(ANSWER_CUE, 1)[1]
        return tail.strip(" :\n").split("\n")[0].strip()
    return text.strip()


def _cue_line_complete(text: str) -> bool:
    """True once ANSWER_CUE has appeared AND a full line has been written after
    it. Stopping the instant the cue substring itself appears (the previous
    behavior) could cut generation off before the short answer was written --
    confirmed on a real run: generation ended with "...So the answer is" and
    nothing after it, so _extract_answer had nothing to return."""
    if ANSWER_CUE not in text:
        return False
    return "\n" in text.split(ANSWER_CUE, 1)[1]


def run_dragin(
    query: str,
    context: str,
    cfg: DraginConfig,
    model: Any = None,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Generate an answer to ``query`` over ``context`` with RIND-gated,
    QFS-sliced sub-calls. Returns a result dict compatible with gate_rlm's
    record schema (answer, latency_s, completion_tokens, llm_calls, leaf_calls,
    root_iterations) plus DRAGIN-specific fields (n_retrievals, rind_triggers).

    ``model``/``tokenizer`` can be passed in to reuse an already-loaded model
    across many examples (experiments/run_dragin.py does this -- loading a 35B
    model per example would be absurd); if omitted they're loaded fresh here.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_dragin_model(cfg.model_path)

    t0 = time.perf_counter()
    prompt = DIRECT_PROMPT.format(context=context, query=query)
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
    while budget > 0:
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

        prompt = RETRIEVAL_TEMPLATE.format(
            context=context,
            passages=f"[1] {passages}\n(worker's read: {worker_answer})"
            if passages
            else f"(nothing found; worker's read: {worker_answer})",
            query=query,
            prefix=generated_text,
        )

        if time.perf_counter() - seg_start > cfg.max_retrieval_seconds:
            time_capped = True
            break  # this retrieval cycle alone exceeded the cap -- no more triggers

    # Retrieval-time cap hit with no answer cue yet: a further RIND trigger here
    # can't lead to another retrieval (we're done triggering either way), so
    # checking theta at all would only cut this pass off after a token or two
    # and discard it for nothing -- give the model one uninterrupted shot at
    # finishing, with at least min_answer_tokens even if generate_length's
    # shared budget is already spent (see DRAGIN_RLM_TEST_RESULTS.md,
    # theta=0.0001: every segment including this one was getting cut at ~2
    # tokens, so the run ended with literally no answer).
    if time_capped and not _cue_line_complete(generated_text):
        segments_run += 1
        final_budget = max(budget, cfg.min_answer_tokens)
        stream = generate_with_probe(
            model, tokenizer, prompt, max_tokens=final_budget, temperature=cfg.temperature
        )
        try:
            for probed in stream:
                generated_text += probed.text
                total_new_tokens += 1
                if _cue_line_complete(generated_text):
                    break
        finally:
            stream.close()

    answer = _extract_answer(generated_text)
    latency = time.perf_counter() - t0
    return {
        "answer": answer,
        "raw_generation": generated_text,
        "latency_s": latency,
        "completion_tokens": total_new_tokens,
        "llm_calls": segments_run + len(triggers),
        "leaf_calls": len(triggers),
        "root_iterations": segments_run,
        "n_retrievals": len(triggers),
        "time_capped": time_capped,
        "rind_triggers": triggers,
        "theta": cfg.theta,
        "top_n": cfg.top_n,
        "max_rind_score": max_rind_score,
        "max_rind_detail": max_rind_detail,
        "rind_checked_tokens": rind_checked_tokens,
        "rind_nonstopword_tokens": rind_nonstopword_tokens,
    }
