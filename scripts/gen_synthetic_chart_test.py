"""Generate results/randomized_SYNTHETIC_TEST.csv -- FABRICATED data for
chart/dashboard testing only, never a real experimental result. Explicitly
NOT part of any reported finding: every row carries a `synthetic` column
(False for the 8 real rows copied in from the actual batch10 theta=0.001
run, True for the 42 generated ones) so real vs. fake is unambiguous even
if the file gets renamed or its rows get copied elsewhere.

Rows 9-50 use the qids at data-row-index 9-50 in results/analysis.csv (a
real, separate theta=1.0 run) purely as a source of realistic-looking qids
and a true/false ratio (16/50 = 32% in that file) to shape the synthetic
metrics around -- not as a claim that these qids were actually re-tested
under theta=0.001. gold_answer is pulled from the real dataset (factual
metadata, not a result); every other column is randomly generated.
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = random.Random(42)

# Real rows 1-8, copied as-is (synthetic=False).
real_rows = list(csv.DictReader(open(ROOT / "results/randomized.csv")))
for r in real_rows:
    r["synthetic"] = "False"

# qids + true/false guide from analysis.csv rows 9-50.
analysis_rows = [
    line.strip().split(",") for line in open(ROOT / "results/analysis.csv").readlines()[1:]
][8:50]
assert len(analysis_rows) == 42

gold_by_qid = {}
for line in open(ROOT / "data/processed/browsecomp.jsonl"):
    row = json.loads(line)
    gold_by_qid[row["qid"]] = row["gold_answer"]

ANSWER_SOURCES = ["raw_reasoning"] * 4 + ["cue"] * 2 + ["worker_fallback"] * 2  # matches real 8-row mix

RAW_REASONING_TEMPLATE = (
    "<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Goal: "
)
WORKER_FALLBACK_SAMPLES = ["NOT FOUND", "Unable to determine from passages"]


def gen_row(qid: str, looks_correct: bool) -> dict:
    source = rng.choice(ANSWER_SOURCES)
    gold = gold_by_qid.get(qid, "Unknown")

    if source == "cue":
        answer = "<short answer>'." if not looks_correct else gold
    elif source == "worker_fallback":
        answer = rng.choice(WORKER_FALLBACK_SAMPLES) if not looks_correct else gold
    else:
        answer = RAW_REASONING_TEMPLATE

    semantic_sim = rng.uniform(0.62, 0.86) if looks_correct else rng.uniform(0.35, 0.62)
    f1 = round(rng.uniform(0.02, 0.1), 6) if (looks_correct and rng.random() < 0.4) else 0.0
    hit_cap = rng.random() < (0.3 if looks_correct else 0.75)  # matches real 2/8 early-stop, 6/8 capped
    n_retrievals = 20 if hit_cap else rng.randint(1, 8)
    root_iterations = n_retrievals if hit_cap else n_retrievals + 1
    completion_tokens = rng.randint(30, 220)
    latency_s = n_retrievals * rng.uniform(140, 200)
    max_rind_score = round(rng.uniform(0.0010, 0.0055), 16)

    return {
        "config": "dragin_rlm",
        "qid": qid,
        "gold_answer": gold,
        "answer": answer,
        "answer_source": source,
        "f1": f1,
        "semantic_sim": semantic_sim,
        "latency_s": latency_s,
        "completion_tokens": completion_tokens,
        "n_retrievals": n_retrievals,
        "root_iterations": root_iterations,
        "max_rind_score": max_rind_score,
        "synthetic": "True",
    }


synthetic_rows = []
for qid, correct, *_ in analysis_rows:
    synthetic_rows.append(gen_row(qid, looks_correct=(correct == "True")))

out_path = ROOT / "results/randomized_SYNTHETIC_TEST.csv"
fieldnames = [
    "config", "qid", "gold_answer", "answer", "answer_source", "f1", "semantic_sim",
    "latency_s", "completion_tokens", "n_retrievals", "root_iterations", "max_rind_score",
    "synthetic",
]
with out_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in real_rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})
    for r in synthetic_rows:
        w.writerow(r)

print(f"wrote {len(real_rows)} real + {len(synthetic_rows)} synthetic rows -> {out_path}")
