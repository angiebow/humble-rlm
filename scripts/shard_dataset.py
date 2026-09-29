"""Deterministically split a processed dataset into N shards for parallel runs
across multiple machines. Same hash-based approach as gate_rlm.data.assign_split,
so every machine that runs this against the same input file gets the same
shard assignment -- no coordination needed, and shard membership is stable if
--n-shards changes (each qid only ever moves, never duplicates).

    python scripts/shard_dataset.py --data data/processed/browsecomp.jsonl \
        --n-shards 4 --out-prefix data/processed/browsecomp_shard

Each shard keeps the existing val/test split label on every row untouched --
sharding is an orthogonal partition purely for spreading work across
machines, not a second train/test split. Results from all shards for the same
condition get concatenated back into one file before eval/aggregate.py runs,
so every condition is still scored against the full dataset.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from gate_rlm.data import read_jsonl, write_jsonl


def shard_of(qid: str, n_shards: int) -> int:
    h = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16)
    return h % n_shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    rows = list(read_jsonl(args.data))
    shards: list[list[dict]] = [[] for _ in range(args.n_shards)]
    for row in rows:
        shards[shard_of(row["qid"], args.n_shards)].append(row)

    counts = Counter()
    for i, shard in enumerate(shards):
        out = f"{args.out_prefix}{i}.jsonl"
        write_jsonl(shard, out)
        counts[i] = len(shard)
        print(f"shard {i}: {len(shard)} examples -> {out}")
    print(f"total: {sum(counts.values())} (source had {len(rows)})")


if __name__ == "__main__":
    main()
