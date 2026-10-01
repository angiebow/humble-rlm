"""Chain 10-question DRAGIN-RLM batches, each with fresh unique questions,
until a wall-clock deadline. Self-contained: starts the worker mlx_lm.server
+ litellm proxy itself if they aren't already up. Pass --offline to force HF
Hub offline once root model, worker model, BrowseComp-Plus data and the
semantic-similarity model are all cached.

Run it yourself, in your own terminal, from the repo root:

    .venv/bin/python scripts/run_continuous_batches.py --hours 7

Round 1 reuses whatever data/processed/browsecomp_batch10_round1_theta001.jsonl
already exists (so a batch prepared and started by hand isn't re-drawn);
every later round samples a fresh set of 10, excluding every qid used so far
(tracked in data/processed/used_qids_theta001.txt, gitignored). Each question
runs as its own run_dragin.py call (not the whole batch at once), so the
consolidated per-question CSV (results/table_dragin_batch10_theta0.001_local_per_question.csv)
updates right after every question finishes, not only once all 10 in a
round are done -- a single question can take 20-90+ min. Within a question,
every retrieval prints a live progress line (qid, retrieval count, tokens,
elapsed) and checkpoints to disk (harness.py's on_checkpoint), so at most
one in-flight segment is ever at risk, never a whole question, and you're
never watching a silent terminal for an hour.

Offline-safe with --offline once everything is cached: this only talks to
localhost (the worker server + proxy it starts) and reads downloaded files.
Without --offline it may reach HF Hub to fetch missing models first.
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
# Resolve the mlx_lm.server / litellm executables from whichever environment is
# running this script (works for ./.venv, ~/.venv, conda, ...), not a hardcoded
# repo-local .venv that only exists on the laptop.
VENV_BIN = Path(sys.executable).parent
WORKER_MODEL = "mlx-community/Qwen3.5-2B-bf16"

os.environ.setdefault("OPENAI_API_BASE", "http://localhost:4000/v1")
os.environ.setdefault("OPENAI_API_KEY", "not-needed")
# Everything needed (root model, worker model, dataset, semantic-similarity
# model) is already cached from earlier runs -- offline mode skips the
# network round-trip HF Hub would otherwise make to check for updates,
# which is exactly the failure point if this runs with no internet.
# Offline mode is opt-in (--offline): on a machine that hasn't downloaded the
# models yet (e.g. a fresh Mac Studio) forcing it would fail on the first load.


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
    ap.add_argument(
        "--offline", action="store_true",
        help="force HF Hub offline (only if every model + dataset is already cached)",
    )
    args = ap.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
        all_runs = sorted(
            RUNS_DIR.glob("dragin_rlm__browsecomp_batch10_round*_theta0.001_local.jsonl"),
            key=lambda p: p.name,
        )
        if out_path not in all_runs:
            all_runs.append(out_path)

        # One question per run_dragin.py call (not the whole batch at once)
        # so the consolidated CSV updates right after each question finishes,
        # not only after all 10 in the round are done -- a single question
        # can take 20-90+ min, so that would otherwise be a long, silent wait.
        one_question_path = ROOT / "data/processed/_one_question.jsonl"
        for r in batch:
            with one_question_path.open("w") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            run([
                sys.executable, str(ROOT / "experiments/run_dragin.py"),
                "--config", str(CONFIG),
                "--data", str(one_question_path),
                "--split", "all",
                "--out", str(out_path),
            ])
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
