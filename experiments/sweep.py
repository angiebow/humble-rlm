"""Choose thresholds on the VALIDATION split by replaying observe-mode logs.

A live run per threshold setting would be unaffordable. Observe mode logs every
worker checkpoint with all signals, so we can replay the exact stopping rule
(gate_rlm.stopping.replay) under any thresholds and compute, per example, where
GATE-RLM would have stopped, what it would have answered, and what it would
have cost. This follows CALM's validation sweep; the chosen point is then
confirmed with LIVE runs on the test split.

Caveat to state in the paper: replay assumes the root would have issued the same
sub-calls; live gating also changes what the root sees, so live test numbers are
the ones we report.

    python experiments/sweep.py --logs results/runs/observe__babilong__val.jsonl \
        --out results/thresholds.json --max-acc-drop 0.01
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import is_correct  # noqa: E402
from gate_rlm.stopping import Checkpoint, Thresholds, replay  # noqa: E402

GRID = {
    "tau_rel": [0.3, 0.5, 0.7],
    "tau_hi": [0.5, 0.7, 0.9],
    "tau_conf": [0.5, 0.7, 0.9],
    "k_agree": [1, 2],
    "rho": [4.0, 6.0, 10.0],
}
FINALIZE_TOKENS = 1500  # rough cost of the root's final turn after a stop


def simulate(records: list[dict], th: Thresholds) -> dict:
    accs, costs, stops = [], [], 0
    for r in records:
        cps = [Checkpoint(**c) for c in r.get("checkpoints", [])]
        full_cost = r.get("total_tokens", 0)
        i, decision = replay(cps, th)
        if i >= 0:
            stops += 1
            answer = decision.answer
            cost = min(full_cost, cps[i].cum_tokens + FINALIZE_TOKENS)
        else:
            answer = r.get("answer", "")  # never stopped: the run's own answer
            cost = full_cost
        accs.append(float(is_correct(answer, r["gold_answer"], r["dataset"])))
        costs.append(cost)
    return {"acc": float(np.mean(accs)), "mean_tokens": float(np.mean(costs)),
            "p95_tokens": float(np.percentile(costs, 95)), "stop_rate": stops / max(1, len(records))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="observe-mode JSONL on val")
    ap.add_argument("--out", default="results/thresholds.json")
    ap.add_argument("--grid-out", default="results/sweep_grid.csv")
    ap.add_argument("--max-acc-drop", type=float, default=0.01,
                    help="allowed accuracy drop vs. never stopping")
    args = ap.parse_args()

    records = [json.loads(l) for p in args.logs for l in Path(p).read_text().splitlines() if l.strip()]
    records = [r for r in records if r.get("mode") == "observe" and not r.get("error")]
    if not records:
        sys.exit("no usable observe-mode records")

    baseline = simulate(records, Thresholds(knee_enabled=False, tau_hi=2.0))  # never stops
    rows = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID.keys(), vals))
        th = Thresholds(tau_rel=p["tau_rel"], tau_hi=p["tau_hi"], tau_conf=p["tau_conf"],
                        k_agree=p["k_agree"], rho=p["rho"])
        rows.append({**p, **simulate(records, th)})
    grid = pd.DataFrame(rows).sort_values("mean_tokens")
    Path(args.grid_out).parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_out, index=False)

    ok = grid[grid.acc >= baseline["acc"] - args.max_acc_drop]
    best = (ok if len(ok) else grid.sort_values(["acc", "mean_tokens"], ascending=[False, True])).iloc[0]
    frozen = {
        "relevance": {"tau_rel": float(best.tau_rel)},
        "stopping": {"tau_hi": float(best.tau_hi), "tau_conf": float(best.tau_conf),
                     "k_agree": int(best.k_agree), "knee": {"rho": float(best.rho)}},
        "_selection": {"n_val_records": len(records), "baseline": baseline,
                       "chosen": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                  for k, v in best.items()},
                       "rule": f"cheapest setting within {args.max_acc_drop:.3f} of never-stop accuracy"},
    }
    Path(args.out).write_text(json.dumps(frozen, indent=2))
    print(f"baseline (never stop): {baseline}")
    print(f"chosen: {frozen['_selection']['chosen']}")
    print(f"wrote {args.out} and {args.grid_out} ({len(grid)} settings)")


if __name__ == "__main__":
    main()
