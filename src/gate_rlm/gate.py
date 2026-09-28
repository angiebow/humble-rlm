"""GateRLM: the recursive-llm RLM with relevance gating and adaptive termination.

Integration points in grishahq/recursive-llm (pinned commit, see pyproject.toml):
  * ``_call_leaf``  every worker sub-call (llm_query / rlm_query at max_depth=1,
                    and their batched forms) goes through here. We add structured
                    output, score relevance, update the stopping policy, and filter
                    what the root sees.
  * ``_call_llm``   root turns. After a stop fires, the root gets
                    ``grace_root_iterations`` turns to finalize on its own; after
                    that FINAL(best_answer) is injected without an API call.
  * ``event_handler`` REPL steps: logs when gold evidence was printed to the root
                    (logging only, used for the over-read ratio).

Three modes share this class so all conditions are logged identically:
  vanilla  unmodified worker prompt, no scoring, no actions (evidence logging only)
  observe  structured worker output + all signals logged, no actions
  gate     full GATE-RLM
Gold evidence fingerprints are used ONLY to write ``contains_gold`` into the logs.
No decision reads them.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from rlm import RLM

from .confidence import WORKER_INSTRUCTION, logprob_confidence, parse_worker_reply
from .relevance import Scorer, contains_any
from .stopping import Checkpoint, StopDecision, StoppingPolicy, Thresholds

STOP_NOTE = (
    "[GATE] Enough evidence has been gathered. Best-supported answer: {answer!r}. "
    "Do not make further sub-calls. Return FINAL now."
)
IRRELEVANT_NOTE = "[GATE] Chunk judged not relevant to the question; skipped."


class GoldTracker:
    """Records the token count at which all required gold evidence had been read."""

    def __init__(self, gold: Optional[List[List[str]]], mode: str = "any") -> None:
        self.gold = gold or []          # list of evidence items, each a list of fingerprints
        self.mode = mode                # "any": one item suffices; "all": every item needed
        self.seen = [False] * len(self.gold)
        self.seen_at_tokens: Optional[int] = None

    def check(self, text: str) -> bool:
        hit = False
        for i, fps in enumerate(self.gold):
            if contains_any(text, fps):
                self.seen[i] = True
                hit = True
        return hit

    def satisfied(self) -> bool:
        if not self.gold:
            return False
        return any(self.seen) if self.mode == "any" else all(self.seen)


class GateRLM(RLM):
    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        user_query: str,
        scorer: Optional[Scorer] = None,
        gold: Optional[List[List[str]]] = None,
        gold_mode: str = "any",
        evidence: Optional[List[List[str]]] = None,
        **rlm_kwargs: Any,
    ) -> None:
        # Attributes used by overridden hooks must exist before RLM.__init__ runs.
        self.cfg = cfg
        self.mode = cfg.get("mode", "gate")
        self.user_query = user_query
        self.scorer = scorer
        self._lock = threading.Lock()
        self._live_state = None
        th = Thresholds.from_config(cfg)
        self.policy = StoppingPolicy(
            th, enabled=(self.mode == "gate" and cfg.get("stopping", {}).get("enabled", True))
        )
        self.filter_irrelevant = (
            self.mode == "gate"
            and cfg.get("relevance", {}).get("enabled", True)
            and cfg.get("relevance", {}).get("filter_irrelevant", True)
        )
        self.grace = int(cfg.get("stopping", {}).get("grace_root_iterations", 1))
        self.conf_source = cfg.get("confidence", {}).get("source", "verbalized")
        self.logprob_scorer = cfg.get("confidence", {}).get("logprob_scorer")
        self.gold = GoldTracker(gold, gold_mode)
        self.evidence = GoldTracker(evidence, "any")  # broader relevant set, flags only
        self.stop: Optional[StopDecision] = None
        self.stop_at_tokens: Optional[int] = None
        self.root_turns_after_stop = 0
        self.forced_final = False
        self.checkpoint_log: List[Dict[str, Any]] = []
        self.aux_usage: List[Dict[str, Any]] = []
        self.n_worker_calls_skipped = 0
        self.n_parse_failures = 0
        if rlm_kwargs.get("max_depth", 1) != 1:
            raise ValueError("GateRLM supports max_depth=1 (the paper's default).")
        super().__init__(event_handler=self._on_event, **rlm_kwargs)

    # ------------------------------------------------------------------ helpers
    def _new_run_state(self, loop=None):  # type: ignore[override]
        state = super()._new_run_state(loop)
        self._live_state = state
        return state

    def cum_tokens(self) -> int:
        if self._live_state is None:
            return 0
        return int(self._live_state.usage.snapshot().get("total_tokens", 0))

    def _mark_gold(self, text: str) -> bool:
        with self._lock:
            hit = self.gold.check(text)
            if self.gold.seen_at_tokens is None and self.gold.satisfied():
                self.gold.seen_at_tokens = self.cum_tokens()
        return hit

    def _on_event(self, event) -> None:
        if event.kind == "repl_step":
            output = event.data.get("output")
            if isinstance(output, str) and output:
                self._mark_gold(output)

    # -------------------------------------------------------------- root turns
    async def _call_llm(self, messages, *, _run_state=None, _node_id="", **kwargs):  # type: ignore[override]
        if self.mode == "gate" and self._current_depth == 0 and self.stop is not None:
            answer = self.stop.answer or self.policy.best_answer()
            if self.root_turns_after_stop >= self.grace and answer:
                self.forced_final = True
                return f"FINAL({answer!r})"
            self.root_turns_after_stop += 1
        return await super()._call_llm(messages, _run_state=_run_state, _node_id=_node_id, **kwargs)

    # ------------------------------------------------------------ worker calls
    async def _call_leaf(  # type: ignore[override]
        self,
        sub_query: str,
        sub_context: str = "",
        model: Optional[str] = None,
        *,
        _run_state=None,
        _node_id: str = "",
    ) -> str:
        chunk = sub_context or sub_query
        contains_gold = self._mark_gold(chunk)
        contains_evidence = self.evidence.check(chunk) if self.evidence.gold else None

        if self.mode == "vanilla":
            text = await super()._call_leaf(
                sub_query, sub_context, model, _run_state=_run_state, _node_id=_node_id
            )
            with self._lock:
                self.checkpoint_log.append(
                    asdict(Checkpoint(len(self.checkpoint_log), self.cum_tokens(),
                                      contains_gold=contains_gold,
                                      contains_evidence=contains_evidence))
                )
            return text

        if self.mode == "gate" and self.stop is not None:
            with self._lock:
                self.n_worker_calls_skipped += 1
            return STOP_NOTE.format(answer=self.stop.answer)

        text = await super()._call_leaf(
            sub_query + WORKER_INSTRUCTION, sub_context, model,
            _run_state=_run_state, _node_id=_node_id,
        )
        reply = parse_worker_reply(text)
        score = None
        if self.scorer is not None:
            score = await asyncio.to_thread(self.scorer.score, self.user_query, chunk)
        conf = reply.confidence
        if self.conf_source == "logprob" and self.logprob_scorer and reply.found:
            lp, usage = await asyncio.to_thread(
                logprob_confidence, self.logprob_scorer, self.user_query, chunk, reply.answer
            )
            self.aux_usage.append({"kind": "logprob", **usage})
            conf = lp if lp is not None else conf

        with self._lock:
            if not reply.parsed:
                self.n_parse_failures += 1
            cp = Checkpoint(
                idx=len(self.checkpoint_log),
                cum_tokens=self.cum_tokens(),
                cosine=score.cosine if score else None,
                rerank=score.rerank if score else None,
                conf=conf,
                found=reply.found,
                answer=reply.answer,
                contains_gold=contains_gold,
                contains_evidence=contains_evidence,
            )
            decision = self.policy.update(cp)
            if decision.stop and self.stop is None:
                self.stop = decision
                self.stop_at_tokens = cp.cum_tokens
            self.checkpoint_log.append(asdict(cp))
            stopped = self.stop

        if self.mode == "observe":
            return text

        if self.filter_irrelevant and not cp.relevant:
            out = IRRELEVANT_NOTE
        elif reply.found and reply.answer:
            out = f"{reply.answer} (confidence {conf:.2f})"
            if reply.evidence:
                out += f'\nEvidence: "{reply.evidence[:300]}"'
        else:
            out = "NOT FOUND in this chunk."
        if stopped is not None:
            out += "\n" + STOP_NOTE.format(answer=stopped.answer)
        return out

    # ------------------------------------------------------------------ report
    def gate_report(self) -> Dict[str, Any]:
        return {
            "stop_reason": self.stop.reason if self.stop else None,
            "stop_at_tokens": self.stop_at_tokens,
            "forced_final": self.forced_final,
            "gold_seen_at_tokens": self.gold.seen_at_tokens,
            "worker_calls_skipped": self.n_worker_calls_skipped,
            "worker_parse_failures": self.n_parse_failures,
            "checkpoints": self.checkpoint_log,
            "aux_usage": self.aux_usage,
        }
