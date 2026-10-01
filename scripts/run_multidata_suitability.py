"""Run DRAGIN-RLM over one question from each of five datasets with very
different context shapes (BABILong, BrowseComp-Plus, HotpotQA, OOLONG,
RepoQA) to see which task shape this system is actually suited to -- same
model, theta, and caps across all five, so the only thing that varies is the
dataset. Builds data/processed/multidata_probe.jsonl first if it doesn't
already exist (scripts/build_multidata_probe.py).

Uses configs/experiments/dragin_rlm_multidata.yaml (root Qwen3.5-2B-4bit,
worker Qwen3.5-0.8B-4bit) and configs/litellm_proxy_multidata.yaml (worker
proxy on :4001) -- both separate from the main dragin_rlm.yaml / 35B setup
used elsewhere, so this can't disturb any other run in progress.

    .venv/bin/python scripts/run_multidata_suitability.py

Starts the worker mlx_lm.server + its litellm proxy itself if they aren't
already up (reuses the project's own .venv entry points).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data/processed/multidata_probe.jsonl"
CONFIG = ROOT / "configs/experiments/dragin_rlm_multidata.yaml"
LITELLM_CONFIG = ROOT / "configs/litellm_proxy_multidata.yaml"
OUT = ROOT / "results/runs/dragin_rlm__multidata_probe.jsonl"
CSV_OUT = ROOT / "results/table_dragin_multidata_probe_per_question.csv"
VENV_BIN = ROOT / ".venv/bin"
WORKER_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
LITELLM_PORT = 4001

os.environ["OPENAI_API_BASE"] = f"http://localhost:{LITELLM_PORT}/v1"
os.environ.setdefault("OPENAI_API_KEY", "not-needed")


def _reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def ensure_local_servers() -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    if not _reachable("http://localhost:8002/v1/models"):
        print("Starting worker mlx_lm.server on :8002 ...", flush=True)
        subprocess.Popen(
            [
                str(VENV_BIN / "mlx_lm.server"), "--model", WORKER_MODEL, "--port", "8002",
                "--chat-template-args", '{"enable_thinking": false}',
            ],
            stdout=open(logs / "mlx_worker.log", "a"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(60):
            if _reachable("http://localhost:8002/v1/models"):
                break
            time.sleep(2)
        else:
            sys.exit("worker mlx_lm.server never came up on :8002")

    if not _reachable(f"http://localhost:{LITELLM_PORT}/v1/models"):
        print(f"Starting litellm proxy on :{LITELLM_PORT} ...", flush=True)
        subprocess.Popen(
            [str(VENV_BIN / "litellm"), "--config", str(LITELLM_CONFIG), "--port", str(LITELLM_PORT)],
            stdout=open(logs / "litellm_proxy_multidata.log", "a"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(60):
            if _reachable(f"http://localhost:{LITELLM_PORT}/v1/models"):
                break
            time.sleep(2)
        else:
            sys.exit(f"litellm proxy never came up on :{LITELLM_PORT}")


def run(cmd: list) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    if not PROBE.exists():
        run([sys.executable, str(ROOT / "scripts/build_multidata_probe.py")])

    ensure_local_servers()

    run([
        sys.executable, str(ROOT / "experiments/run_dragin.py"),
        "--config", str(CONFIG),
        "--data", str(PROBE),
        "--split", "all",
        "--out", str(OUT),
    ])
    run([
        sys.executable, str(ROOT / "eval/per_question.py"),
        "--runs", str(OUT),
        "--out", str(CSV_OUT),
        "--semantic",
    ])
    print(f"\ndone -> {CSV_OUT}")


if __name__ == "__main__":
    main()
