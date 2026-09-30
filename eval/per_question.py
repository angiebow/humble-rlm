"""Per-question accuracy + inference-time table -- not aggregated, one row per
(qid, seed), so you can see exactly which questions a condition got right or
wrong and how long each one took, against BrowseComp-Plus's gold_answer.

eval/aggregate.py already computes accuracy/latency, but only as a single
rolled-up row per condition (acc_mean, latency_p50, ...); this is the
row-level view underneath that rollup.

    python eval/per_question.py --runs results/runs/dragin_rlm__browsecomp_sample50.jsonl \
        --out results/table_dragin_per_question.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import f1, is_correct  # noqa: E402


def load(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/table_per_question.csv")
    args = ap.parse_args()

    df = load(args.runs)
    df["correct"] = [
        is_correct(a or "", g, d) for a, g, d in zip(df.answer, df.gold_answer, df.dataset)
    ]
    df["f1"] = [f1(a or "", g) for a, g in zip(df.answer, df.gold_answer)]

    cols = [
        c
        for c in (
            "config",
            "qid",
            "gold_answer",
            "answer",
            "correct",
            "f1",
            "latency_s",
            "completion_tokens",
            "n_retrievals",
            "error",
        )
        if c in df.columns
    ]
    table = df[cols].sort_values([c for c in ("config", "qid") if c in cols])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(table.to_string(index=False))
    print(f"\n{len(table)} questions -> {out}")
    if len(table):
        print(
            f"accuracy: {table['correct'].mean():.3f}  "
            f"latency p50: {table['latency_s'].median():.1f}s  "
            f"latency p95: {table['latency_s'].quantile(0.95):.1f}s"
        )


if __name__ == "__main__":
    main()
