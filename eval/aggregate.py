"""Aggregate run logs into the paper's main table (RQ1, RQ3) and paired tests.

    python eval/aggregate.py --runs results/runs/*__babilong__test.jsonl \
        --reference b2_vanilla --method gate_full --out results/table_main.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import f1, is_correct, over_read_ratio  # noqa: E402
from eval.stats import mcnemar_exact, paired_bootstrap, wilcoxon  # noqa: E402

FINALIZE_TOKENS = 1500
warnings.filterwarnings("ignore", message="Mean of empty slice")


def load(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["correct"] = [is_correct(a or "", g, d) for a, g, d in
                     zip(df.answer, df.gold_answer, df.dataset)]
    df["f1"] = [f1(a or "", g) for a, g in zip(df.answer, df.gold_answer)]
    df["over_read"] = [over_read_ratio(t, s) for t, s in
                       zip(df.total_tokens, df.get("gold_seen_at_tokens", [None] * len(df)))]
    return df


def evidence_stats(checkpoints: list) -> tuple:
    """(precision, recall) of what the gate let through, at chunk level."""
    kept = [c for c in checkpoints if c.get("relevant")]
    evid = [c for c in checkpoints if c.get("contains_evidence")]
    if not checkpoints or all(c.get("relevant") is None for c in checkpoints):
        return None, None
    prec = np.mean([bool(c.get("contains_evidence")) for c in kept]) if kept else np.nan
    rec = np.mean([bool(c.get("relevant")) for c in evid]) if evid else np.nan
    return prec, rec


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cfg, g in df.groupby("config"):
        per_seed_acc = g.groupby("seed").correct.mean()
        cost = g.cost_usd.fillna(0.0)
        leaf = g.get("leaf_calls", pd.Series(0, index=g.index)).fillna(0)
        ev = [evidence_stats(c) for c in g.get("checkpoints", pd.Series([[]] * len(g)))
              if isinstance(c, list)]
        prec = np.nanmean([p for p, _ in ev if p is not None]) if ev else np.nan
        rec = np.nanmean([r for _, r in ev if r is not None]) if ev else np.nan
        gold_seen = g.get("gold_seen_at_tokens")
        oracle = (gold_seen.dropna() + FINALIZE_TOKENS).mean() if gold_seen is not None else np.nan
        stop_at = g.get("stop_at_tokens")
        premature = np.nan
        if stop_at is not None and stop_at.notna().any():
            stopped = g[stop_at.notna()]
            early = stopped.gold_seen_at_tokens.isna() | (stopped.gold_seen_at_tokens > stopped.stop_at_tokens)
            premature = float((early & ~stopped.correct).mean())
        out.append({
            "config": cfg,
            "n_runs": len(g),
            "acc_mean": per_seed_acc.mean(),
            "acc_std": per_seed_acc.std(ddof=0),
            "f1": g.f1.mean(),
            "tokens_mean": g.total_tokens.mean(),
            "tokens_p50": g.total_tokens.median(),
            "tokens_p95": g.total_tokens.quantile(0.95),
            "cost_mean": cost.mean(),
            "cost_p95": cost.quantile(0.95),
            "latency_p50": g.latency_s.median() if "latency_s" in g else np.nan,
            "latency_p95": g.latency_s.quantile(0.95) if "latency_s" in g else np.nan,
            "subcalls_mean": leaf.mean(),
            "cost_per_correct": cost.sum() / max(1, int(g.correct.sum())),
            "subcall_productivity": g.correct.sum() / max(1.0, leaf.sum()),
            "over_read_mean": g.over_read.dropna().mean(),
            "oracle_tokens_mean": oracle,
            "evidence_precision": prec,
            "evidence_recall": rec,
            "premature_stop_rate": premature,
            "direct_route_share": (g.get("route") == "direct").mean() if "route" in g else np.nan,
            "error_rate": g.error.notna().mean() if "error" in g else 0.0,
        })
    return pd.DataFrame(out).sort_values("tokens_mean")


def paired_tests(df: pd.DataFrame, reference: str, method: str) -> dict:
    key = ["qid", "seed"]
    a = df[df.config == method].set_index(key)
    b = df[df.config == reference].set_index(key)
    common = a.index.intersection(b.index)
    if not len(common):
        return {"error": "no paired runs"}
    a, b = a.loc[common], b.loc[common]
    return {
        "n_pairs": len(common),
        "accuracy_mcnemar": mcnemar_exact(a.correct, b.correct),
        "tokens_bootstrap": paired_bootstrap(a.total_tokens, b.total_tokens),
        "tokens_wilcoxon": wilcoxon(a.total_tokens, b.total_tokens),
        "cost_bootstrap": paired_bootstrap(a.cost_usd.fillna(0), b.cost_usd.fillna(0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reference", default="b2_vanilla")
    ap.add_argument("--method", default="gate_full")
    ap.add_argument("--out", default="results/table_main.csv")
    args = ap.parse_args()

    df = load(args.runs)
    table = summarize(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    # RQ3: accuracy and tokens by length x complexity
    rq3 = (df.groupby(["config", "length_bucket", "complexity"])
             .agg(acc=("correct", "mean"), tokens=("total_tokens", "mean"), n=("qid", "count"))
             .reset_index())
    rq3.to_csv(Path(args.out).with_name("table_rq3.csv"), index=False)

    tests = paired_tests(df, args.reference, args.method)
    Path(args.out).with_name("paired_tests.json").write_text(json.dumps(tests, indent=2))

    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print("\nRQ3 breakdown:\n", rq3.round(3).to_string(index=False))
    print(f"\n{args.method} vs {args.reference}:", json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
