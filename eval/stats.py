"""Paired significance tests: GATE-RLM vs. a baseline on the same questions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (accuracy)."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5).pvalue)
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000,
                     seed: int = 0) -> dict:
    """95% CI for mean(a - b), resampling questions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return {"mean_diff": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0}
    res = stats.wilcoxon(a, b)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}
