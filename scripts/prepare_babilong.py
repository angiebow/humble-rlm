"""Build the BABILong subset (D1) with gold-evidence labels.

BABILong's Hugging Face release has no supporting-fact column. We recover it:
the "0k" config holds only the bAbI facts, so for each sample we (1) align the
0k facts inside the long version, (2) pick the supporting facts with a rule
(qa1, qa2; see gate_rlm.data.babilong_gold_facts), (3) store fingerprints around
their exact positions. Samples that fail alignment or the rule check are skipped
and counted, never mislabelled.

    python scripts/prepare_babilong.py --tasks qa1 qa2 --lengths 4k 32k 128k 512k --n 30
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from gate_rlm.data import (
    align_facts, assign_split, babi_facts, babilong_gold_facts,
    window_fingerprints, write_jsonl,
)
from gate_rlm.router import estimate_tokens

SHORT_LENGTHS = {"0k", "1k", "2k", "4k", "8k"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa2"])
    ap.add_argument("--lengths", nargs="+", default=["4k", "32k", "128k", "512k"])
    ap.add_argument("--n", type=int, default=30, help="samples per (task, length)")
    ap.add_argument("--val-fraction", type=float, default=0.3)
    ap.add_argument("--out", default="data/processed/babilong.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset

    zero = load_dataset("RMT-team/babilong", "0k")
    rows, skipped = [], Counter()
    for task in args.tasks:
        facts_by_idx = [babi_facts(x["input"]) for x in zero[task]]
        for length in args.lengths:
            ds = load_dataset("RMT-team/babilong", length)[task]
            kept = 0
            for i, ex in enumerate(ds):
                if kept >= args.n:
                    break
                facts = facts_by_idx[i]
                positions = align_facts(ex["input"], facts)
                if positions is None:
                    skipped[f"{task}/{length}: not aligned"] += 1
                    continue
                gold_idx, ok = babilong_gold_facts(task, ex["question"], facts, ex["target"])
                if not ok:
                    skipped[f"{task}/{length}: gold rule failed"] += 1
                    continue
                context = ex["input"]
                gold_fps = [
                    window_fingerprints(context, positions[j], positions[j] + len(facts[j]))
                    for j in gold_idx
                ]
                evid_fps = [
                    window_fingerprints(context, positions[j], positions[j] + len(facts[j]))
                    for j in range(len(facts))
                ]
                qid = f"babilong-{task}-{length}-{i}"
                rows.append({
                    "qid": qid,
                    "dataset": "babilong",
                    "split": assign_split(f"babilong-{task}-{i}", args.val_fraction),
                    "query": ex["question"],
                    "context": context,
                    "gold_answer": ex["target"],
                    "gold_fingerprints": gold_fps,
                    "gold_mode": "all",
                    "evidence_fingerprints": evid_fps,
                    "length_bucket": "short" if length in SHORT_LENGTHS else "long",
                    "complexity": "simple" if task == "qa1" else "complex",
                    "context_tokens": estimate_tokens(context),
                    "meta": {"task": task, "length": length, "index": i,
                             "n_facts": len(facts), "gold_fact_idx": gold_idx},
                })
                kept += 1
            print(f"{task}/{length}: kept {kept}")
    n = write_jsonl(rows, args.out)
    print(f"wrote {n} examples to {args.out}")
    for reason, count in sorted(skipped.items()):
        print(f"  skipped {count:4d}  {reason}")
    # Note: the split is keyed on (task, index), not length, so the same underlying
    # question never appears in both val and test at different lengths.


if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    main()
