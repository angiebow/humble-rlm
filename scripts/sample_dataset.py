"""Draw a deterministic random sample of N examples from a processed dataset,
for quick validation runs (e.g. checking a pipeline/config fix actually works)
without committing to a full shard. Same hash-based determinism approach as
assign_split/shard_dataset -- reproducible across machines given the same
input file, seed, and n.

    python scripts/sample_dataset.py --data data/processed/browsecomp.jsonl \
        --n 50 --seed 42 --out data/processed/browsecomp_sample50.jsonl
"""

from __future__ import annotations

import argparse
import random

from gate_rlm.data import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["val", "test", "all"], default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.data)
            if args.split == "all" or r.get("split") == args.split]
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    n = write_jsonl(sample, args.out)
    print(f"sampled {n} of {len(rows)} {args.split} examples (seed={args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()
