"""DRAGIN-RLM HotpotQA evaluation with RIND scores.

Evaluates HotpotQA using DRAGIN-RLM with:
- Root model (35B) with attention probing for RIND computation
- Worker model (2B) for retrieval-based answer refinement
- Metrics: accuracy, F1, semantic similarity, RIND score, iterations, latency, tokens
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import f1, is_correct, semantic_similarity
from gate_rlm.config import load_config
from gate_rlm.data import write_jsonl, read_jsonl
from dragin_rlm.harness import DraginConfig, run_dragin
from dragin_rlm.attention_probe import load_dragin_model


def load_hotpotqa(limit: int = 10) -> list[dict]:
    """Load HotpotQA validation questions."""
    from datasets import load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    examples = []

    for i, ex in enumerate(dataset):
        if i >= limit:
            break
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        context = "\n\n".join(
            f"{title}\n" + " ".join(sents) for title, sents in zip(titles, sentences)
        )
        examples.append({
            "qid": f"hotpotqa-{i:03d}",
            "dataset": "hotpotqa",
            "query": ex["question"],
            "context": context,
            "gold_answer": ex["answer"],
            "supporting_facts": ex.get("supporting_facts"),
        })

    return examples


def main():
    import argparse
    ap = argparse.ArgumentParser(description="DRAGIN-RLM HotpotQA Evaluation with RIND")
    ap.add_argument("--limit", type=int, default=150, help="Number of questions to evaluate (default: 150)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results",
                    help="Where results are written (default: results/). Use a separate dir for smoke tests, since the jsonl is appended to and the csv/summary are overwritten.")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "dragin_hotpotqa_eval_results.jsonl"
    csv_file = out_dir / "table_dragin_hotpotqa_eval.csv"

    print("=" * 80)
    print("DRAGIN-RLM HotpotQA Evaluation with RIND Scores")
    print("=" * 80)

    # Load configuration
    print("\nLoading DRAGIN configuration...")
    config_path = ROOT / "configs/experiments/dragin_rlm_hotpotqa.yaml"
    cfg = load_config(str(config_path))

    from dragin_rlm.pipeline import dragin_config_from_cfg
    dcfg = dragin_config_from_cfg(cfg)

    print(f"  Model: {dcfg.model_path}")
    print(f"  Theta: {dcfg.theta}")
    print(f"  Max triggers: {dcfg.max_triggers}")

    # Load model once
    print("\nLoading root model...")
    model, tokenizer = load_dragin_model(dcfg.model_path)
    print(f"  ✓ Loaded {dcfg.model_path}")

    # Load questions
    print("\nLoading HotpotQA questions...")
    questions = load_hotpotqa(limit=args.limit)
    print(f"  ✓ Loaded {len(questions)} questions")

    # Run evaluation
    print("\nRunning DRAGIN-RLM evaluation...")
    results = []

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['qid']}: {q['query'][:50]}...", end=" ", flush=True)

        t_start = time.perf_counter()

        # Run DRAGIN
        dragin_result = run_dragin(
            query=q["query"],
            context=q["context"],
            cfg=dcfg,
            model=model,
            tokenizer=tokenizer,
        )

        latency_s = time.perf_counter() - t_start

        # Compile result
        result = {
            "qid": q["qid"],
            "dataset": "hotpotqa",
            "query": q["query"],
            "gold_answer": q["gold_answer"],
            "answer": dragin_result.get("answer", ""),
            "answer_source": dragin_result.get("answer_source", ""),
            "worker_answers": dragin_result.get("worker_answers", []),
            "retrieval_queries": dragin_result.get("retrieval_queries", []),

            # Efficiency metrics
            "latency_s": latency_s,
            "completion_tokens": dragin_result.get("completion_tokens", 0),
            "llm_calls": dragin_result.get("llm_calls", 0),
            "leaf_calls": dragin_result.get("leaf_calls", 0),
            "root_iterations": dragin_result.get("root_iterations", 0),
            "n_retrievals": dragin_result.get("n_retrievals", 0),

            # RIND metrics
            "max_rind_score": dragin_result.get("max_rind_score", 0.0),
            "rind_checked_tokens": dragin_result.get("rind_checked_tokens", 0),
            "rind_nonstopword_tokens": dragin_result.get("rind_nonstopword_tokens", 0),
            "rind_nonstopword_ratio": (
                dragin_result.get("rind_nonstopword_tokens", 0) /
                max(1, dragin_result.get("rind_checked_tokens", 1))
            ),

            # Time cap indicator
            "time_capped": dragin_result.get("time_capped", False),
        }
        results.append(result)

        with open(out_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"✓ (RIND:{result['max_rind_score']:.3f}, retrievals:{result['n_retrievals']})")

    print(f"\n✓ Wrote {len(results)} results to {out_file}")

    # Evaluate
    print("\nEvaluating results...")
    df = pd.DataFrame(results)

    df["correct"] = [
        is_correct(a or "", g, "hotpotqa")
        for a, g in zip(df.answer, df.gold_answer)
    ]
    df["f1"] = [f1(a or "", g) for a, g in zip(df.answer, df.gold_answer)]
    df["semantic_similarity"] = [
        semantic_similarity(a or "", g) for a, g in zip(df.answer, df.gold_answer)
    ]

    # Compute aggregates
    accuracy = df["correct"].mean()
    f1_score = df["f1"].mean()
    semantic_sim = df["semantic_similarity"].mean()
    avg_tokens = df["completion_tokens"].mean()
    avg_latency = df["latency_s"].mean()
    avg_rind = df["max_rind_score"].mean()
    total_retrievals = df["n_retrievals"].sum()

    # Write per-question CSV
    csv_cols = [
        "qid", "gold_answer", "answer", "correct", "f1", "semantic_similarity",
        "completion_tokens", "latency_s", "n_retrievals", "leaf_calls",
        "max_rind_score", "rind_checked_tokens", "rind_nonstopword_ratio",
        "answer_source", "time_capped"
    ]
    csv_df = df[[c for c in csv_cols if c in df.columns]].sort_values("qid")
    csv_df.to_csv(csv_file, index=False)
    print(f"✓ Per-question metrics -> {csv_file}")

    # Summary report
    summary = f"""
{'=' * 80}
DRAGIN-RLM HotpotQA Evaluation Summary
{'=' * 80}

ACCURACY METRICS:
  - Accuracy:              {accuracy:.3f}
  - F1 Score:              {f1_score:.3f}
  - Semantic Similarity:   {semantic_sim:.3f}

EFFICIENCY METRICS:
  - Avg Tokens:            {avg_tokens:.0f} per question
  - Total Tokens:          {df['completion_tokens'].sum():.0f}
  - Avg Latency:           {avg_latency:.2f}s per question

RETRIEVAL METRICS:
  - Total Retrievals:      {int(total_retrievals)} (worker calls)
  - Avg Retrievals:        {df['n_retrievals'].mean():.2f} per question
  - Max Retrievals:        {int(df['n_retrievals'].max())}

RIND (Retrieval-Induced Neuron Divergence) METRICS:
  - Avg RIND Score:        {avg_rind:.3f}
  - Max RIND Score:        {df['max_rind_score'].max():.3f}
  - Min RIND Score:        {df['max_rind_score'].min():.3f}
  - Avg Nonstopword Ratio: {df['rind_nonstopword_ratio'].mean():.3f}

RIND INTERPRETATION:
  - Higher RIND score → more uncertain/ambiguous
  - Nonstopword ratio → fraction of semantic tokens checked
  - Theta threshold: {dcfg.theta} (triggers when RIND > theta)

TRIGGERING ANALYSIS:
  - Questions with retrievals: {(df['n_retrievals'] > 0).sum()}/{len(df)}
  - Avg confidence (no retrieval): {(df['n_retrievals'] == 0).sum()} questions
  - Avg uncertainty (with retrieval): {(df['n_retrievals'] > 0).sum()} questions

Questions evaluated: {len(df)}
Config: {config_path.name}
Root model: {dcfg.model_path}
Worker model: mlx-community/Qwen3.5-2B
"""

    summary_file = out_dir / "dragin_hotpotqa_eval_summary.txt"
    summary_file.write_text(summary)
    print(summary)
    print(f"✓ Summary report -> {summary_file}")


if __name__ == "__main__":
    main()
