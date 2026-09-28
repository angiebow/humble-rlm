"""Stage 2 stopping rule, shared by live runs and offline threshold replay.

Stop when EITHER
  (a) sufficiency: an answer is supported by >= k_agree relevant, confident worker
      outputs, and the best supporting relevance is >= tau_hi
      (FLARE/DeepConf-style confidence + ESC-style agreement), OR
  (b) knee: the gain curve of relevant chunks per chunk read has flattened
      (Cormack & Grossman knee method from technology-assisted review).

Keeping this logic pure (no model calls) means `sweep.py` can replay logged
checkpoints under any threshold setting and get exactly the decision a live run
would have made at that point.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Checkpoint:
    """One piece of evidence the RLM read (a worker sub-call)."""

    idx: int
    cum_tokens: int                  # total tokens spent up to and including this call
    cosine: Optional[float] = None
    rerank: Optional[float] = None
    conf: Optional[float] = None     # worker's (verbalized or log-prob) confidence
    found: Optional[bool] = None     # worker says the chunk answers the query
    answer: str = ""
    contains_gold: Optional[bool] = None      # LOGGING ONLY; never used for decisions
    contains_evidence: Optional[bool] = None  # LOGGING ONLY; never used for decisions
    relevant: Optional[bool] = None       # set by the policy


@dataclass
class Thresholds:
    tau_cos: float = 0.30
    tau_rel: float = 0.50
    tau_hi: float = 0.70
    tau_conf: float = 0.70
    k_agree: int = 1
    knee_enabled: bool = True
    rho: float = 6.0
    min_checkpoints: int = 4
    use_relevance: bool = True       # False: every chunk counts as relevant

    @classmethod
    def from_config(cls, cfg: dict) -> "Thresholds":
        rel, stop = cfg.get("relevance", {}), cfg.get("stopping", {})
        knee = stop.get("knee", {})
        return cls(
            tau_cos=rel.get("tau_cos", 0.30),
            tau_rel=rel.get("tau_rel", 0.50),
            tau_hi=stop.get("tau_hi", 0.70),
            tau_conf=stop.get("tau_conf", 0.70),
            k_agree=int(stop.get("k_agree", 1)),
            knee_enabled=knee.get("enabled", True),
            rho=knee.get("rho", 6.0),
            min_checkpoints=knee.get("min_checkpoints", 4),
            use_relevance=rel.get("enabled", True),
        )


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation and articles."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def knee_ratio(gains: List[int]) -> Tuple[float, int]:
    """Slope ratio at the knee of a cumulative gain curve.

    ``gains[t]`` is the cumulative number of relevant chunks after reading t+1
    chunks. The knee i maximises the distance to the line from the origin to the
    last point; rho = slope before the knee / slope after it (+1 smoothing).
    """
    t = len(gains)
    if t < 2 or gains[-1] == 0:
        return 0.0, -1
    x_t, y_t = float(t), float(gains[-1])
    best_i, best_d = 0, -1.0
    for i in range(t):
        x_i, y_i = float(i + 1), float(gains[i])
        d = abs(y_t * x_i - x_t * y_i)  # proportional to perpendicular distance
        if d > best_d:
            best_i, best_d = i, d
    x_i, y_i = float(best_i + 1), float(gains[best_i])
    slope_before = y_i / x_i
    slope_after = (y_t - y_i + 1.0) / max(1.0, x_t - x_i)
    return slope_before / slope_after, best_i


@dataclass
class StopDecision:
    stop: bool
    reason: str = ""
    answer: str = ""


@dataclass
class StoppingPolicy:
    th: Thresholds
    enabled: bool = True
    checkpoints: List[Checkpoint] = field(default_factory=list)
    _votes: Dict[str, List[Tuple[float, float, str]]] = field(default_factory=dict)

    def is_relevant(self, cp: Checkpoint) -> bool:
        if not self.th.use_relevance:
            return True
        if cp.cosine is not None and cp.cosine < self.th.tau_cos:
            return False
        return cp.rerank is not None and cp.rerank >= self.th.tau_rel

    def best_answer(self) -> str:
        if not self._votes:
            return ""
        key = max(
            self._votes,
            key=lambda k: (len(self._votes[k]), max(c for c, _, _ in self._votes[k])),
        )
        return self._votes[key][0][2]

    def update(self, cp: Checkpoint) -> StopDecision:
        cp.relevant = self.is_relevant(cp)
        self.checkpoints.append(cp)
        confident = cp.conf is not None and cp.conf >= self.th.tau_conf
        if cp.relevant and cp.found and confident and cp.answer.strip():
            key = normalize_answer(cp.answer)
            if key:
                self._votes.setdefault(key, []).append(
                    (cp.conf or 0.0, cp.rerank if cp.rerank is not None else 1.0, cp.answer)
                )
        if not self.enabled:
            return StopDecision(False)

        # (a) sufficiency
        for key, votes in self._votes.items():
            if len(votes) >= self.th.k_agree and max(r for _, r, _ in votes) >= self.th.tau_hi:
                return StopDecision(True, "sufficient", votes[0][2])

        # (b) knee / diminishing returns
        if self.th.knee_enabled and len(self.checkpoints) >= self.th.min_checkpoints:
            gains, total = [], 0
            for c in self.checkpoints:
                total += int(bool(c.relevant))
                gains.append(total)
            rho, _ = knee_ratio(gains)
            if total >= 1 and rho >= self.th.rho:
                return StopDecision(True, "knee", self.best_answer())
        return StopDecision(False)


def replay(checkpoints: List[Checkpoint], th: Thresholds) -> Tuple[int, StopDecision]:
    """Offline replay: index where GATE-RLM would stop (-1 = never) and its decision."""
    policy = StoppingPolicy(th)
    for i, cp in enumerate(checkpoints):
        fresh = Checkpoint(**{**cp.__dict__, "relevant": None})
        decision = policy.update(fresh)
        if decision.stop:
            return i, decision
    return -1, StopDecision(False, "", policy.best_answer())
