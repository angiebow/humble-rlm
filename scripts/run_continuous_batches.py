"""Chain 10-question DRAGIN-RLM batches, each with fresh unique questions,
until a wall-clock deadline. Self-contained: starts the worker mlx_lm.server
+ litellm proxy itself if they aren't already up, and forces HF Hub into
offline mode so it never tries to reach the network (everything it needs --
root model, worker model, BrowseComp-Plus data, the semantic-similarity
model -- is already cached locally from earlier runs).

Run it yourself, in your own terminal, from the repo root:

    .venv/bin/python scripts/run_continuous_batches.py --hours 7

Round 1 reuses whatever data/processed/browsecomp_batch10_round1_theta001.jsonl
already exists (so a batch prepared and started by hand isn't re-drawn);
every later round samples a fresh set of 10, excluding every qid used so far
(tracked in data/processed/used_qids_theta001.txt, gitignored). The
consolidated per-question CSV (results/table_dragin_batch10_theta0.001_local_per_question.csv)
is regenerated after every round, so you have a readable result on disk even
if you stop the script (Ctrl-C) or it hits the deadline mid-round -- the
current question's own progress is additionally checkpointed after every
retrieval (harness.py's on_checkpoint), so at most one in-flight segment is
ever at risk, never a whole question.

Safe to run fully offline: this only ever talks to localhost (the worker
server + proxy it starts) and reads already-downloaded files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/browsecomp.jsonl"
RUNS_DIR = ROOT / "results/runs"
LOGS_DIR = ROOT / "logs"
USED_QIDS_FILE = ROOT / "data/processed/used_qids_theta001.txt"
BATCH_SIZE = 10
CONFIG = ROOT / "configs/experiments/dragin_rlm.yaml"
CSV_OUT = ROOT / "results/table_dragin_batch10_theta0.001_local_per_question.csv"
VENV_BIN = ROOT / ".venv/bin"
WORKER_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"

os.environ.setdefault("OPENAI_API_BASE", "http://localhost:4000/v1")
os.environ.setdefault("OPENAI_API_KEY", "not-needed")
# Everything needed (root model, worker model, dataset, semantic-similarity
# model) is already cached from earlier runs -- offline mode skips the
# network round-trip HF Hub would otherwise make to check for updates,
# which is exactly the failure point if this runs with no internet.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def ensure_local_servers() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    if not _reachable("http://localhost:8002/v1/models"):
        print("Starting worker mlx_lm.server on :8002 ...", flush=True)
        subprocess.Popen(
            [
                str(VENV_BIN / "mlx_lm.server"), "--model", WORKER_MODEL, "--port", "8002",
                "--chat-template-args", '{"enable_thinking": false}',
            ],
            stdout=open(LOGS_DIR / "mlx_worker.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survives this script exiting
        )
        for _ in range(60):
            if _reachable("http://localhost:8002/v1/models"):
                break
            time.sleep(2)
        else:
            sys.exit("worker mlx_lm.server never came up on :8002 -- check logs/mlx_worker.log")

    if not _reachable("http://localhost:4000/v1/models"):
        print("Starting litellm proxy on :4000 ...", flush=True)
        subprocess.Popen(
            [str(VENV_BIN / "litellm"), "--config", str(ROOT / "configs/litellm_proxy.yaml"), "--port", "4000"],
            stdout=open(LOGS_DIR / "litellm_proxy.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(60):
            if _reachable("http://localhost:4000/v1/models"):
                break
            time.sleep(2)
        else:
            sys.exit("litellm proxy never came up on :4000 -- check logs/litellm_proxy.log")


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

    ensure_local_servers()

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
