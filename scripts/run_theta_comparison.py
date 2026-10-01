"""Replay the EXACT same question set already used for the running
theta=0.001 batch10 run, at theta=0.1, for a direct controlled comparison --
same questions, same documents, same model, only theta differs.

No new sampling happens here: it reads data/processed/browsecomp_batch10_
round1_theta001.jsonl (or --data, for the batch10b set) as-is, the identical
10-question file the theta=0.001 run already consumed.

Run yourself, in your own terminal, from the repo root on server 1
(the worker mlx_lm.server + litellm proxy are assumed already running --
same ones the theta=0.001 batches are using):

    .venv/bin/python scripts/run_theta_comparison.py
    .venv/bin/python scripts/run_theta_comparison.py --data data/processed/browsecomp_batch10b_round1_theta001.jsonl

Safe to run alongside the theta=0.001 batches: separate output files, same
config otherwise (model, generate_length, max_retrieval_seconds,
max_triggers) -- only --theta differs. Per-retrieval checkpointing and the
live progress line (harness.py's on_checkpoint) apply here exactly like
every other run_dragin.py invocation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/dragin_rlm.yaml"
THETA = 0.1

os.environ.setdefault("OPENAI_API_BASE", "http://localhost:4000/v1")
os.environ.setdefault("OPENAI_API_KEY", "not-needed")


def run(cmd: list) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(ROOT / "data/processed/browsecomp_batch10_round1_theta001.jsonl"),
        help="the exact same question-set file a theta=0.001 run already used",
    )
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(
            f"{data_path} not found -- expected the exact same question-set "
            f"file already used for a running theta=0.001 batch."
        )

    tag = data_path.stem.replace("browsecomp_", "").replace("_theta001", "")
    out = ROOT / f"results/runs/dragin_rlm__browsecomp_{tag}_theta0.1.jsonl"
    csv_out = ROOT / f"results/table_dragin_{tag}_theta0.1_per_question.csv"

    run([
        sys.executable, str(ROOT / "experiments/run_dragin.py"),
        "--config", str(CONFIG),
        "--data", str(data_path),
        "--split", "all",
        "--theta", str(THETA),
        "--out", str(out),
    ])
    run([
        sys.executable, str(ROOT / "eval/per_question.py"),
        "--runs", str(out),
        "--out", str(csv_out),
        "--semantic",
    ])
    print(f"done -> {csv_out}")


if __name__ == "__main__":
    main()
