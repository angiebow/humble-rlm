"""Run one configuration over one processed dataset split.

    python experiments/run.py --config configs/experiments/gate_full.yaml \
        --data data/processed/babilong.jsonl --split test \
        --thresholds results/thresholds.json --workers 4

Writes one JSON line per (example, seed) to
results/runs/<config>__<dataset>__<split>.jsonl and resumes safely: pairs already
present in the output file are skipped, so a crashed run can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gate_rlm.config import load_config  # noqa: E402
from gate_rlm.data import read_jsonl  # noqa: E402
from gate_rlm.pipeline import run_example  # noqa: E402
from gate_rlm.relevance import build_scorer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--thresholds", default=None, help="frozen thresholds from sweep.py")
    ap.add_argument("--limit", type=int, default=None, help="max examples (smoke tests)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override config seeds")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--filter", default=None, help="substring that qid must contain")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config, args.thresholds)
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
    if args.split == "test" and cfg["mode"] == "gate" and not args.thresholds:
        print("WARNING: gate run on TEST without --thresholds; using config defaults.")

    examples = [
        ex for ex in read_jsonl(args.data)
        if (args.split == "all" or ex.get("split") == args.split)
        and (args.filter is None or args.filter in ex["qid"])
    ]
    if args.limit:
        examples = examples[: args.limit]
    dataset = examples[0]["dataset"] if examples else "empty"
    out = Path(args.out or f"{cfg['logging']['out_dir']}/{cfg['name']}__{dataset}__{args.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["qid"], r["seed"]))
    jobs = [(ex, s) for ex in examples for s in seeds if (ex["qid"], s) not in done]
    print(f"{cfg['name']}: {len(jobs)} runs to do ({len(done)} already in {out})")

    scorer = build_scorer(cfg) if cfg["mode"] in ("gate", "observe") else None
    lock = threading.Lock()
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_example, ex, cfg, scorer, s) for ex, s in jobs]
        errors = 0
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            rec.pop("context", None)
            errors += int(bool(rec.get("error")))
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    print(f"done; {errors} runs recorded an error (see the 'error' field)")


if __name__ == "__main__":
    main()
