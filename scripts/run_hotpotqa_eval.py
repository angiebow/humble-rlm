"""Test RLM-DRAGIN on HotpotQA with comprehensive accuracy and efficiency metrics.

Runs DRAGIN-RLM on 10 HotpotQA questions and reports:

ACCURACY:
  - Accuracy: fraction of correct answers
  - Recall: fraction of relevant passages retrieved
  - Precision: fraction of retrieved passages that are relevant
  - F1: harmonic mean of precision and recall
  - ECE: Expected Calibration Error (confidence vs correctness)

EFFICIENCY:
  - Token Footprint: total tokens used per question
  - Recursion Depth: max depth of reasoning steps (retrieval iterations)

    python scripts/run_hotpotqa_eval.py --config configs/experiments/dragin_rlm.yaml \\
        --out results/hotpotqa_eval_results.jsonl

Output files:
  - results/hotpotqa_eval_results.jsonl (raw runs)
  - results/table_hotpotqa_eval.csv (per-question metrics)
  - results/hotpotqa_eval_summary.txt (aggregate report)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import f1, is_correct, semantic_similarity
from gate_rlm.data import write_jsonl, read_jsonl
from gate_rlm.config import load_config


def load_hotpotqa_questions(limit: int = 10) -> list[dict]:
    """Load first N HotpotQA questions."""
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
            "split": "test",
            "query": ex["question"],
            "context": context,
            "gold_answer": ex["answer"],
            "supporting_facts": ex.get("supporting_facts"),  # for recall/precision
        })

    return examples


def compute_ece(confidences: list[float], correctness: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error: mean absolute difference between
    confidence and accuracy in each bin."""
    if not confidences:
        return np.nan

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correctness[mask].mean()
            ece += np.abs(bin_conf - bin_acc) * mask.sum() / len(correctness)

    return ece


def compute_retrieval_recall_precision(
    checkpoints: list[dict],
    supporting_facts: list[tuple[str, int]] | None
) -> tuple[float, float]:
    """Compute recall and precision of retrieved passages against supporting facts."""
    if not supporting_facts or not checkpoints:
        return np.nan, np.nan

    # supporting_facts is list of (title, sentence_idx)
    relevant_passages = set(supporting_facts)
    retrieved_passages = set()

    for cp in checkpoints:
        if cp.get("document"):
            # Extract title and try to match
            doc = cp["document"]
            for title, _ in supporting_facts:
                if title.lower() in doc.lower():
                    retrieved_passages.add(title)
                    break

    if not retrieved_passages:
        return 0.0, np.nan

    recall = len(retrieved_passages & relevant_passages) / len(relevant_passages)
    precision = len(retrieved_passages & relevant_passages) / len(retrieved_passages)

    return recall, precision


def compute_recursion_depth(checkpoints: list[dict]) -> int:
    """Get max depth of retrieval iterations."""
    return len(checkpoints) if checkpoints else 0


def load_config_and_run(
    config_path: str,
    hotpotqa_data: list[dict],
    output_file: Path,
) -> None:
    """Load config and run DRAGIN-RLM on HotpotQA data."""
    from dragin_rlm.pipeline import dragin_config_from_cfg, run_example
    from dragin_rlm.attention_probe import load_dragin_model
    from dotenv import load_dotenv
    from tqdm import tqdm

    load_dotenv()
    cfg = load_config(config_path)
    dcfg = dragin_config_from_cfg(cfg)

    # Load model once
    print(f"Loading model: {dcfg.model_path}")
    model = load_dragin_model(dcfg.model_path)

    # Prepare output file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Run on HotpotQA data
    results = []
    for ex in tqdm(hotpotqa_data, desc="Running DRAGIN-RLM"):
        result = run_example(ex, dcfg, model)
        result["seed"] = 0
        result["config"] = cfg["name"]
        results.append(result)

        # Append to file immediately (checkpoint)
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"Wrote {len(results)} results to {output_file}")


def evaluate_results(run_file: Path, output_csv: Path, summary_txt: Path) -> None:
    """Evaluate DRAGIN-RLM results and compute metrics."""
    print(f"\nEvaluating results from {run_file}...")

    # Load results
    results = list(read_jsonl(run_file))
    df = pd.DataFrame(results)

    # Compute correctness and F1
    df["correct"] = [
        is_correct(a or "", g, d)
        for a, g, d in zip(df.answer, df.gold_answer, df.dataset)
    ]
    df["f1"] = [f1(a or "", g) for a, g in zip(df.answer, df.gold_answer)]

    # Compute efficiency metrics
    df["token_footprint"] = df.get("total_tokens", 0)
    df["recursion_depth"] = [
        compute_recursion_depth(cp) if isinstance(cp, list) else 0
        for cp in df.get("checkpoints", pd.Series([[]] * len(df)))
    ]

    # Compute retrieval metrics (recall/precision)
    retrieval_metrics = []
    for _, row in df.iterrows():
        checkpoints = row.get("checkpoints") or []
        supporting_facts = row.get("supporting_facts")
        recall, precision = compute_retrieval_recall_precision(
            checkpoints if isinstance(checkpoints, list) else [],
            supporting_facts
        )
        retrieval_metrics.append((recall, precision))

    df["retrieval_recall"] = [r for r, _ in retrieval_metrics]
    df["retrieval_precision"] = [p for _, p in retrieval_metrics]

    # Compute confidence-based ECE (using F1 as confidence proxy)
    confidences = df["f1"].fillna(0.5).tolist()
    correctness = df["correct"].tolist()
    ece = compute_ece(confidences, correctness)

    # Write per-question table
    cols_to_keep = [
        "config", "qid", "gold_answer", "answer", "f1", "correct",
        "token_footprint", "recursion_depth", "retrieval_recall",
        "retrieval_precision"
    ]
    cols = [c for c in cols_to_keep if c in df.columns]
    table = df[cols].sort_values("qid")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    print(f"Per-question metrics -> {output_csv}")

    # Compute aggregates
    accuracy = df["correct"].mean()
    recall_mean = df["retrieval_recall"].mean()
    precision_mean = df["retrieval_precision"].mean()
    f1_mean = df["f1"].mean()
    token_footprint_mean = df["token_footprint"].mean()
    token_footprint_p95 = df["token_footprint"].quantile(0.95)
    recursion_depth_mean = df["recursion_depth"].mean()
    recursion_depth_max = df["recursion_depth"].max()

    # Write summary report
    summary = f"""
====== RLM-DRAGIN HotpotQA Evaluation Summary ======

ACCURACY Metrics:
  - Accuracy:           {accuracy:.3f}
  - Recall (retrieval): {recall_mean:.3f}
  - Precision (retrieval): {precision_mean:.3f}
  - F1 Score:           {f1_mean:.3f}
  - ECE (calibration):  {ece:.3f}

EFFICIENCY Metrics:
  - Token Footprint (mean): {token_footprint_mean:.0f} tokens
  - Token Footprint (p95):  {token_footprint_p95:.0f} tokens
  - Recursion Depth (mean): {recursion_depth_mean:.2f} iterations
  - Recursion Depth (max):  {int(recursion_depth_max)} iterations

Questions evaluated: {len(df)}
Config: {df["config"].iloc[0] if len(df) > 0 else "unknown"}
"""

    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text(summary)
    print(summary)
    print(f"Summary report -> {summary_txt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to DRAGIN config")
    ap.add_argument("--limit", type=int, default=10, help="Number of HotpotQA questions to test")
    ap.add_argument("--out", default="results/hotpotqa_eval_results.jsonl")
    ap.add_argument("--skip-run", action="store_true", help="Skip running pipeline, only evaluate")
    args = ap.parse_args()

    out = Path(args.out)
    csv_out = out.parent / f"table_{out.stem}.csv"
    summary_out = out.parent / f"{out.stem}_summary.txt"

    # Load HotpotQA data
    if not args.skip_run:
        print(f"Loading {args.limit} HotpotQA questions...")
        hotpotqa_data = load_hotpotqa_questions(args.limit)
        write_jsonl(hotpotqa_data, out.parent / "hotpotqa_input.jsonl")
        print(f"Loaded {len(hotpotqa_data)} questions")

        # Run DRAGIN-RLM
        print(f"\nRunning DRAGIN-RLM on {len(hotpotqa_data)} questions...")
        load_config_and_run(args.config, hotpotqa_data, out)

    # Evaluate results
    if out.exists():
        evaluate_results(out, csv_out, summary_out)
    else:
        print(f"Error: {out} not found. Did the run complete?")


if __name__ == "__main__":
    main()
