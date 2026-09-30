"""Per-question accuracy + inference-time table -- not aggregated, one row per
(qid, seed), so you can see exactly which questions a condition got right or
wrong and how long each one took, against BrowseComp-Plus's gold_answer.

eval/aggregate.py already computes accuracy/latency, but only as a single
rolled-up row per condition (acc_mean, latency_p50, ...); this is the
row-level view underneath that rollup.

    python eval/per_question.py --runs results/runs/dragin_rlm__browsecomp_sample50.jsonl \
        --out results/table_dragin_per_question.csv

Pass --semantic to add a semantic-similarity column (cosine similarity of
sentence embeddings) alongside exact_match/contains_match's is_correct --
useful when the answer field is a long, unfinished reasoning chain rather
than a clean short string (contains_match still catches a verbatim-but-
buried gold string; it can't catch a paraphrase). Off by default since it
loads an embedding model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import f1, is_correct, semantic_correct, semantic_similarity  # noqa: E402


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
    ap.add_argument(
        "--semantic", action="store_true", help="add semantic_sim / semantic_correct columns"
    )
    ap.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.75,
        help="cosine similarity >= this counts as semantic_correct",
    )
    ap.add_argument("--semantic-model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    df = load(args.runs)
    df["correct"] = [
        is_correct(a or "", g, d) for a, g, d in zip(df.answer, df.gold_answer, df.dataset)
    ]
    df["f1"] = [f1(a or "", g) for a, g in zip(df.answer, df.gold_answer)]
    if args.semantic:
        df["semantic_sim"] = [
            semantic_similarity(a or "", g, args.semantic_model)
            for a, g in zip(df.answer, df.gold_answer)
        ]
        df["semantic_correct"] = df["semantic_sim"] >= args.semantic_threshold

    # correct / semantic_correct are boolean pass/fail indicators derived from
    # answer + gold_answer, which are both already in the table -- dropped
    # from the CSV so the table shows the actual answers to compare, not a
    # verdict; accuracy/semantic_accuracy are still printed below from df.
    cols = [
        c
        for c in (
            "config",
            "qid",
            "gold_answer",
            "answer",
            "answer_source",
            "f1",
            "semantic_sim",
            "latency_s",
            "completion_tokens",
            "n_retrievals",
            "root_iterations",
            "max_rind_score",
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
    if len(df):
        semantic = (
            f"  semantic_accuracy: {df['semantic_correct'].mean():.3f}"
            f" (threshold={args.semantic_threshold})"
            if "semantic_correct" in df.columns
            else ""
        )
        print(
            f"accuracy: {df['correct'].mean():.3f}{semantic}  "
            f"latency p50: {df['latency_s'].median():.1f}s  "
            f"latency p95: {df['latency_s'].quantile(0.95):.1f}s"
        )


if __name__ == "__main__":
    main()
