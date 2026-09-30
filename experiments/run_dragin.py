"""Run DRAGIN-RLM over one processed dataset split.

    python experiments/run_dragin.py --config configs/experiments/dragin_rlm.yaml \
        --data data/processed/browsecomp_sample50.jsonl --split all \
        --out results/runs/dragin_rlm__browsecomp_sample50.jsonl

Deliberately separate from experiments/run.py, and deliberately sequential
(no --workers fan-out): the other conditions call out to a litellm proxy over
HTTP, so many cheap concurrent requests make sense. DRAGIN-RLM's root model is
loaded directly in this one process (src/dragin_rlm/attention_probe.py) because
RIND needs per-token attention that the REST bridge can't expose -- see that
module's docstring. A 35B-A3B model only makes sense resident once; running
several workers here would mean loading it several times over, which won't fit
in memory and wouldn't be faster (MoE compute is memory-bandwidth bound, not
helped by process-level parallelism on one machine). Examples are processed one
at a time, worker sub-calls (regular litellm/REST) run inline within each.

Resumes safely like run.py: pairs already present in the output file are
skipped, so a crashed run can simply be restarted.

Before pointing this at a real run, call
``dragin_rlm.attention_probe.verify_against_fused(model_path)`` once (e.g. from
a Python REPL on the machine that has mlx + the model weights) and confirm it
prints PASS -- the unfused attention patch this depends on was written and
reviewed against mlx-lm's source but has not been executed anywhere; see
attention_probe.py's module docstring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dragin_rlm.attention_probe import load_dragin_model  # noqa: E402
from dragin_rlm.pipeline import dragin_config_from_cfg, run_example  # noqa: E402
from gate_rlm.config import load_config  # noqa: E402
from gate_rlm.data import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--limit", type=int, default=None, help="max examples (smoke tests)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--filter", default=None, help="substring that qid must contain")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the one-time unfused-attention sanity check (not recommended)",
    )
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    dcfg = dragin_config_from_cfg(cfg)

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
    jobs = [(ex, s) for ex in examples for s in args.seeds if (ex["qid"], s) not in done]
    print(f"{cfg['name']}: {len(jobs)} runs to do ({len(done)} already in {out})")
    if not jobs:
        return

    print(f"loading {dcfg.model_path} (direct mlx-lm, not the litellm proxy)...")
    model, tokenizer = load_dragin_model(dcfg.model_path)

    if not args.skip_verify:
        from dragin_rlm.attention_probe import verify_against_fused

        if not verify_against_fused(dcfg.model_path, model=model, tokenizer=tokenizer):
            print(
                "REFUSING to run: the unfused-attention patch does not reproduce "
                "the fused kernel's output on this model -- RIND scores would be "
                "wrong. Investigate attention_probe.py before proceeding, or pass "
                "--skip-verify if you've already confirmed this out-of-band."
            )
            sys.exit(1)

    with out.open("a", encoding="utf-8") as f:
        errors = 0
        for ex, s in tqdm(jobs):
            rec = run_example(ex, cfg, s, model=model, tokenizer=tokenizer)
            rec.pop("context", None)
            errors += int(bool(rec.get("error")))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print(f"done; {errors} runs recorded an error (see the 'error' field)")


if __name__ == "__main__":
    main()
