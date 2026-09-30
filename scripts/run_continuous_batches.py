"""Chain 10-question DRAGIN-RLM batches, each with fresh unique questions,
until a wall-clock deadline. Written for a long unattended local run: round 1
reuses whatever batch file already exists (so a batch prepared and started
by hand isn't re-drawn), every later round samples a fresh set excluding
every qid used so far (tracked in data/processed/used_qids_theta001.txt),
and the consolidated per-question CSV is regenerated after every round so
progress is visible even if the loop is stopped early.

    python scripts/run_continuous_batches.py --hours 7

Requires OPENAI_API_BASE / OPENAI_API_KEY for the worker's litellm calls --
set below to point at the local proxy (configs/litellm_proxy.yaml), matching
what scripts/start_mlx_local.sh's worker+proxy pair serve on.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/browsecomp.jsonl"
RUNS_DIR = ROOT / "results/runs"
USED_QIDS_FILE = ROOT / "data/processed/used_qids_theta001.txt"
BATCH_SIZE = 10
CONFIG = ROOT / "configs/experiments/dragin_rlm.yaml"
CSV_OUT = ROOT / "results/table_dragin_batch10_theta0.001_local_per_question.csv"

os.environ.setdefault("OPENAI_API_BASE", "http://localhost:4000/v1")
os.environ.setdefault("OPENAI_API_KEY", "not-needed")


def load_used() -> set:
    if USED_QIDS_FILE.exists():
        return set(USED_QIDS_FILE.read_text().split())
    return set()


def save_used(used: set) -> None:
    USED_QIDS_FILE.write_text("\n".join(sorted(used)) + "\n")


def draw_batch(used: set, seed: int) -> list:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    test_rows = [r for r in rows if r.get("split") == "test"]
    rng = random.Random(seed)
    rng.shuffle(test_rows)
    return [r for r in test_rows if r["qid"] not in used][:BATCH_SIZE]


def run(cmd: list) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=7.0)
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    used = load_used()
    round_num = 1
    while time.time() < deadline:
        batch_path = ROOT / f"data/processed/browsecomp_batch10_round{round_num}_theta001.jsonl"
        if batch_path.exists():
            batch = [json.loads(line) for line in batch_path.read_text().splitlines() if line.strip()]
            print(f"=== Round {round_num}: reusing existing {batch_path.name} ===", flush=True)
        else:
            batch = draw_batch(used, seed=int(time.time()))
            if len(batch) < BATCH_SIZE:
                print(f"Only {len(batch)} unused test questions left -- stopping.", flush=True)
                break
            with batch_path.open("w") as f:
                for r in batch:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"=== Round {round_num}: {[r['qid'] for r in batch]} ===", flush=True)

        for r in batch:
            used.add(r["qid"])
        save_used(used)

        out_path = RUNS_DIR / f"dragin_rlm__browsecomp_batch10_round{round_num}_theta0.001_local.jsonl"
        run([
            sys.executable, str(ROOT / "experiments/run_dragin.py"),
            "--config", str(CONFIG),
            "--data", str(batch_path),
            "--split", "all",
            "--out", str(out_path),
        ])

        all_runs = sorted(
            RUNS_DIR.glob("dragin_rlm__browsecomp_batch10_round*_theta0.001_local.jsonl"),
            key=lambda p: p.name,
        )
        run([
            sys.executable, str(ROOT / "eval/per_question.py"),
            "--runs", *[str(p) for p in all_runs],
            "--out", str(CSV_OUT),
            "--semantic",
        ])

        round_num += 1

    print("Stopped: deadline reached or ran out of unique questions.", flush=True)


if __name__ == "__main__":
    main()
