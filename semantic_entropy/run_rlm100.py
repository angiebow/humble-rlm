#!/usr/bin/env python
"""Plain-RLM benchmark on 100 BrowseComp-Plus questions (two disjoint random sets of 50) on ONE server,
with a checkpoint after the first 50.

Usage (from the folder containing bcp_rlm_runner.py and bcp_two_sets_50_seed20260930.json):
    nohup python run_rlm100.py SERVER_INDEX > runs/rlm100_orchestrator.log 2>&1 &
    python run_rlm100.py SERVER_INDEX --dry      # only write the configs

Phase 1 = set1, phase 2 = set2. Each phase runs LANES runner processes in parallel (each on its own slice of the
set, sharing one root + one worker mlx_lm server). When all lanes of a phase have finished and judged, the phase's
results/logs/traces are zipped into rlm100_checkpoint_<set>.zip. Re-running resumes (runners skip done questions,
finished phases are skipped).

Settings: plain RLM (20-turn budget, 30-min safety stop, 1 sequential worker, thinking off). Hidden per-turn probes
(draft answer, semantic entropy) are logged but never fed back, so trajectories stay plain RLM;
they let semantic-entropy stop rules be evaluated offline later.
"""
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

K = int(sys.argv[1])
DRY = "--dry" in sys.argv
BASE = Path.cwd()
LANES = 3
TAG = "rlm100"
SETS = json.loads((BASE / "bcp_two_sets_50_seed20260930.json").read_text())

eb = {"top_k": 20, "repetition_penalty": 1.1, "repetition_context_size": 256,
      "chat_template_kwargs": {"enable_thinking": False}}
COMMON = {
    "shard_id": K, "n_shards": 4, "base_dir": str(BASE), "seed": 0,
    "root_model": "mlx-community/Qwen3.5-35B-A3B-4bit",
    "worker_model": "mlx-community/Qwen3.5-2B-bf16",
    "max_iterations": 20,          # turn budget per question
    "max_timeout_s": None,         # no wall-clock cap (depends on GPU sharing)
    "hard_timeout_s": 1800,        # 30-min safety stop
    "call_timeout_s": 600,
    "socket_timeout_s": 7200,
    "max_concurrent_subcalls": 1,  # 1 worker per question, sequential sub-calls (as in the RLM paper)
    "worker_concurrency": LANES,
    "root_sampling": {"temperature": 0.7, "top_p": 0.8, "max_tokens": 2048, "logprobs": True, "extra_body": dict(eb)},
    "worker_sampling": {"temperature": 0.7, "top_p": 0.8, "max_tokens": 2048, "logprobs": True, "extra_body": dict(eb)},
    "probe": {"draft_max_tokens": 48, "draft_temperature": 0.1, "se_samples": 10, "se_temperature": 1.0, "se_top_p": 0.9, "se_top_k": 50},
    "stop_rule": None, "condition": "plain",
    "judge_drafts": True, "answer_fallback": True,
}


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def configs(set_name):
    ids = SETS[set_name]
    out = []
    for j in range(LANES):
        cfg = dict(COMMON, run_tag=f"{TAG}_{set_name}_r{j}", only_ids=ids[j::LANES], set_name=set_name)
        p = BASE / f"bcp_rlm_{TAG}_{set_name}_r{j}_config.json"
        p.write_text(json.dumps(cfg, indent=1))
        out.append((j, cfg, p))
    return out


def done_file(cfg):
    return BASE / "results" / f"bcp_rlm_{cfg['run_tag']}_shard{K}_DONE.json"


def checkpoint(set_name):
    z = BASE / f"{TAG}_checkpoint_{set_name}.zip"
    pats = [f"results/*{TAG}_{set_name}*", f"runs/shard{K}/*{TAG}_{set_name}*", f"bcp_rlm_{TAG}_{set_name}_*config.json"]
    extra = [f"runs/shard{K}/{f}" for f in ("mem.csv", "system.json", "servers.json")] + ["bcp_two_sets_50_seed20260930.json"]
    files = sorted({str(p) for pat in pats for p in BASE.glob(pat)} | {str(BASE / f) for f in extra if (BASE / f).exists()})
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, os.path.relpath(f, BASE))
    log(f"CHECKPOINT {set_name}: {z.name} ({z.stat().st_size // 1024} KB, {len(files)} files)")


def run_phase(set_name):
    cfgs = configs(set_name)
    if DRY:
        for j, cfg, p in cfgs:
            log(f"wrote {p.name}: {len(cfg['only_ids'])} questions")
        return
    if all(done_file(cfg).exists() for _, cfg, _ in cfgs):
        log(f"phase {set_name} already done")
    else:
        procs = []
        for j, cfg, p in cfgs:
            if done_file(cfg).exists():
                continue
            lf = open(BASE / "runs" / f"shard{K}" / f"runner_{TAG}_{set_name}_r{j}.log", "a")
            pr = subprocess.Popen([sys.executable, str(BASE / "bcp_rlm_runner.py"), str(p)], stdout=lf,
                                  stderr=subprocess.STDOUT, start_new_session=True, cwd=str(BASE))
            procs.append((j, pr))
            log(f"{set_name} lane r{j}: pid {pr.pid}, {len(cfg['only_ids'])} questions")
            time.sleep(20)  # let the first lane start/claim the model servers
        for j, pr in procs:
            rc = pr.wait()
            log(f"{set_name} lane r{j} exited rc={rc}")
        missing = [j for j, cfg, _ in cfgs if not done_file(cfg).exists()]
        if missing:
            log(f"phase {set_name}: lanes {missing} have no DONE file; rerun this script to resume")
            sys.exit(1)
    checkpoint(set_name)


(BASE / "results").mkdir(exist_ok=True)
(BASE / "runs" / f"shard{K}").mkdir(parents=True, exist_ok=True)
log(f"{TAG} on server {K}: {LANES} lanes, sets {list(k for k in SETS if k.startswith('set'))}, pid {os.getpid()}")
for s in ("set1", "set2"):
    run_phase(s)
log("ALL DONE")
