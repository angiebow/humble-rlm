"""RQ2: which runtime signal best predicts relevance and evidence sufficiency?

Uses observe-mode logs (all signals recorded, no gating), pooled over checkpoints.
  relevance task:   does THIS chunk contain relevant evidence?  (contains_evidence)
  sufficiency task: has all required gold evidence been read by this point?
                    signals are the running max up to the checkpoint.
Reports AUROC and AUPRC per signal.

    python eval/signals.py --logs results/runs/observe__babilong__val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SIGNALS = {
    "embedding cosine": lambda c: c.get("cosine"),
    "reranker score": lambda c: c.get("rerank"),
    "worker confidence": lambda c: (c.get("conf") or 0.0) if c.get("found") else 0.0,
    "reranker x confidence": lambda c: (c.get("rerank") or 0.0)
    * ((c.get("conf") or 0.0) if c.get("found") else 0.0),
}


def collect(records: list) -> pd.DataFrame:
    rows = []
    for r in records:
        seen_at = r.get("gold_seen_at_tokens")
        running = {k: 0.0 for k in SIGNALS}
        for c in r.get("checkpoints") or []:
            row = {"qid": r["qid"], "seed": r["seed"],
                   "relevant_label": bool(c.get("contains_evidence") or c.get("contains_gold")),
                   "sufficient_label": seen_at is not None and c["cum_tokens"] >= seen_at}
            for name, fn in SIGNALS.items():
                v = fn(c)
                row[name] = np.nan if v is None else float(v)
                running[name] = max(running[name], 0.0 if v is None else float(v))
                row[f"{name} (running max)"] = running[name]
            rows.append(row)
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, label: str, suffix: str = "") -> pd.DataFrame:
    out = []
    y = df[label].astype(int)
    for name in SIGNALS:
        col = f"{name}{suffix}"
        mask = df[col].notna()
        if y[mask].nunique() < 2:
            continue
        out.append({"signal": name, "task": label.replace("_label", ""),
                    "auroc": roc_auc_score(y[mask], df.loc[mask, col]),
                    "auprc": average_precision_score(y[mask], df.loc[mask, col]),
                    "positives": int(y[mask].sum()), "n": int(mask.sum())})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", default="results/table_rq2_signals.csv")
    args = ap.parse_args()
    recs = [json.loads(l) for p in args.logs for l in Path(p).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("mode") == "observe" and not r.get("error")]
    df = collect(recs)
    if df.empty:
        raise SystemExit("no checkpoints found in observe logs")
    table = pd.concat([score(df, "relevant_label"),
                       score(df, "sufficient_label", " (running max)")])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
