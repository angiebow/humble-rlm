"""Report the distribution of max S_RIND seen per question, and what retrieval
rate each candidate theta would have produced, without needing a new run --
harness.run_dragin now always tracks max_rind_score (the max S_RIND seen across
all checked tokens, regardless of whether it crossed theta), since it's free to
compute alongside the trigger check itself.

Use this BEFORE picking theta for a real run: a threshold chosen blind is
indistinguishable from a disabled trigger (see DRAGIN_RLM_TEST_RESULTS.md).

    python eval/threshold_diagnostic.py --runs results/runs/dragin_rlm__browsecomp_sample50.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    ap.add_argument(
        "--candidates",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
        help="theta values to report the resulting retrieval rate for",
    )
    args = ap.parse_args()

    df = load(args.runs)
    if "max_rind_score" not in df.columns:
        raise SystemExit(
            "no max_rind_score column in these runs -- they predate the "
            "threshold-diagnostic instrumentation in harness.py. Rerun to get it."
        )

    scores = df["max_rind_score"].dropna()
    print(f"{len(scores)} questions with a recorded max S_RIND\n")
    print("max S_RIND distribution:")
    for q in (0, 5, 10, 25, 50, 75, 90, 95, 100):
        print(f"  p{q:>3}: {np.percentile(scores, q):.4f}")

    print("\nwhat each candidate theta would have triggered at least once for:")
    for theta in sorted(args.candidates):
        would_fire = (scores > theta).sum()
        print(f"  theta={theta:<6} -> {would_fire}/{len(scores)} questions "
              f"({100 * would_fire / len(scores):.1f}%) would fire at least once")

    if "rind_checked_tokens" in df.columns and "rind_nonstopword_tokens" in df.columns:
        checked = df["rind_checked_tokens"].sum()
        nonstop = df["rind_nonstopword_tokens"].sum()
        if checked:
            print(
                f"\n{nonstop}/{checked} checked tokens ({100 * nonstop / checked:.1f}%) "
                f"were non-stopword (s_i=1) -- the rest are auto-zeroed by the "
                f"semantic filter regardless of entropy/attention"
            )

    if "max_rind_detail" in df.columns:
        details = df["max_rind_detail"].dropna()
        if len(details):
            entropies = [d["entropy"] for d in details]
            attns = [d["max_attn"] for d in details]
            print(
                f"\nat each question's own max-score token: "
                f"entropy median={np.median(entropies):.3f} (range {min(entropies):.3f}-{max(entropies):.3f}), "
                f"max_attn median={np.median(attns):.3f} (range {min(attns):.3f}-{max(attns):.3f})"
            )


if __name__ == "__main__":
    main()
