"""Run DRAGIN-RLM over one question from each of five datasets with very
different context shapes (BABILong, BrowseComp-Plus, HotpotQA, OOLONG,
RepoQA) to see which task shape this system is actually suited to -- same
model, theta, and caps across all five, so the only thing that varies is the
dataset. Builds data/processed/multidata_probe.jsonl first if it doesn't
already exist (scripts/build_multidata_probe.py) -- the SAME 5 questions are
reused across every model tier, so tiers are directly comparable.

Parametrized by --tier so the exact same probe can be rerun at a different
model-pair capacity without duplicating this script or touching any other
tier's config/ports/output:

    .venv/bin/python scripts/run_multidata_suitability.py --tier 08b_2b   # default
    .venv/bin/python scripts/run_multidata_suitability.py --tier 2b_4b

Each tier starts its own worker mlx_lm.server + litellm proxy on its own
ports (never reusing another tier's), so multiple tiers can run back to
back -- or even concurrently, GPU contention notwithstanding -- without
interfering with each other or with the main theta-sweep setup (:4000).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data/processed/multidata_probe.jsonl"
VENV_BIN = ROOT / ".venv/bin"

# tier name -> (root/thinker model, worker model, worker port, litellm port)
TIERS = {
    "08b_2b": ("mlx-community/Qwen3.5-2B-4bit", "mlx-community/Qwen3.5-0.8B-4bit", 8002, 4001),
    "2b_4b": ("mlx-community/Qwen3.5-4B-4bit", "mlx-community/Qwen3.5-2B-4bit", 8003, 4002),
}


def _reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def ensure_local_servers(worker_model: str, worker_port: int, litellm_port: int, litellm_config: Path) -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    if not _reachable(f"http://localhost:{worker_port}/v1/models"):
        print(f"Starting worker mlx_lm.server ({worker_model}) on :{worker_port} ...", flush=True)
        subprocess.Popen(
            [
                str(VENV_BIN / "mlx_lm.server"), "--model", worker_model, "--port", str(worker_port),
                "--chat-template-args", '{"enable_thinking": false}',
            ],
            stdout=open(logs / f"mlx_worker_{worker_port}.log", "a"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(60):
            if _reachable(f"http://localhost:{worker_port}/v1/models"):
                break
            time.sleep(2)
        else:
            sys.exit(f"worker mlx_lm.server never came up on :{worker_port}")

    if not _reachable(f"http://localhost:{litellm_port}/v1/models"):
        print(f"Starting litellm proxy on :{litellm_port} ...", flush=True)
        subprocess.Popen(
            [str(VENV_BIN / "litellm"), "--config", str(litellm_config), "--port", str(litellm_port)],
            stdout=open(logs / f"litellm_proxy_{litellm_port}.log", "a"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(60):
            if _reachable(f"http://localhost:{litellm_port}/v1/models"):
                break
            time.sleep(2)
        else:
            sys.exit(f"litellm proxy never came up on :{litellm_port}")


def run(cmd: list) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), default="08b_2b")
    args = ap.parse_args()

    root_model, worker_model, worker_port, litellm_port = TIERS[args.tier]
    config = ROOT / f"configs/experiments/dragin_rlm_multidata_{args.tier}.yaml"
    litellm_config = ROOT / f"configs/litellm_proxy_multidata_{args.tier}.yaml"
    out = ROOT / f"results/runs/dragin_rlm__multidata_probe_{args.tier}.jsonl"
    csv_out = ROOT / f"results/table_dragin_multidata_probe_{args.tier}_per_question.csv"

    if not config.exists() or not litellm_config.exists():
        sys.exit(
            f"Missing config for tier '{args.tier}': expected {config} and "
            f"{litellm_config} to both exist. (root={root_model}, worker={worker_model})"
        )

    os.environ["OPENAI_API_BASE"] = f"http://localhost:{litellm_port}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "not-needed")

    if not PROBE.exists():
        run([sys.executable, str(ROOT / "scripts/build_multidata_probe.py")])

    ensure_local_servers(worker_model, worker_port, litellm_port, litellm_config)

    run([
        sys.executable, str(ROOT / "experiments/run_dragin.py"),
        "--config", str(config),
        "--data", str(PROBE),
        "--split", "all",
        "--out", str(out),
    ])
    run([
        sys.executable, str(ROOT / "eval/per_question.py"),
        "--runs", str(out),
        "--out", str(csv_out),
        "--semantic",
    ])
    print(f"\ndone (tier={args.tier}: root={root_model}, worker={worker_model}) -> {csv_out}")


if __name__ == "__main__":
    main()
