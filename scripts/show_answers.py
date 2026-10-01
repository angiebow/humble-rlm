"""Print DRAGIN-RLM run records in full -- nothing truncated.

    python scripts/show_answers.py results/runs/dragin_rlm__browsecomp_batch10_round1_theta0.001_local.jsonl
    python scripts/show_answers.py 'results/runs/dragin_rlm__*batch10*_round*.jsonl' --qid browsecomp-1034

Read-only: it only reads the .jsonl run files. For each question it shows the gold
and final answer, why generation ended, every retrieval (token index, RIND score,
QFS query, worker answer) and the complete raw generation. The per-question CSV is
not truncated either; an answer that looks cut off was cut off by the run itself
(usually the retrieval cap -- see "ended:" below), not by the display.
"""

import argparse
import glob
import json


def ended_reason(r: dict) -> str:
    n = r.get("n_retrievals", 0)
    triggers = r.get("rind_triggers", [])
    cap = r.get("max_triggers", 20)
    if "So the answer is" in (r.get("raw_generation") or ""):
        return "answer cue reached"
    if n >= cap:
        idx = [t.get("index") for t in triggers]
        looped = len(idx) > 1 and len(set(idx[1:])) == 1
        return f"CUT at the retrieval cap ({n} retrievals)" + (
            f" -- same token index {idx[-1]} re-triggered {idx.count(idx[-1])}x (retrieval loop)" if looped else ""
        )
    return "stopped (EOS, token budget or time cap)"


def show(r: dict) -> None:
    bar = "=" * 100
    print(f"{bar}\n{r['qid']}   gold: {r.get('gold_answer')}\n{bar}")
    print(f"tokens: {r.get('completion_tokens')}  retrievals: {r.get('n_retrievals')}  "
          f"latency: {r.get('latency_s', 0):.0f}s  ended: {ended_reason(r)}")
    for i, t in enumerate(r.get("rind_triggers", []), 1):
        print(f"\n  retrieval {i}: token index {t.get('index')}  RIND score {t.get('score'):.5f}")
        print(f"    QFS query    : {t.get('query')}")
        print(f"    worker answer: {t.get('worker_answer')}")
    print("\n--- FINAL ANSWER (extracted) ---")
    print(r.get("answer"))
    print("\n--- FULL RAW GENERATION ---")
    print(r.get("raw_generation"))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run .jsonl file(s) or glob(s)")
    ap.add_argument("--qid", default=None, help="only questions whose qid contains this")
    args = ap.parse_args()
    files = sorted({f for pat in args.runs for f in glob.glob(pat)})
    for f in files:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if args.qid is None or args.qid in r["qid"]:
                    show(r)


if __name__ == "__main__":
    main()
