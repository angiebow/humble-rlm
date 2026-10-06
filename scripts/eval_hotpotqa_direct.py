"""HotpotQA evaluation using local Qwen models via direct HTTP (no litellm).

Loads HotpotQA questions and evaluates using:
- Root model (35B) for initial reasoning
- Worker model (2B) for retrieving answers from passages
- Direct HTTP calls to mlx_lm.server on localhost:8005

Metrics: accuracy, F1, token usage, retrieval recall/precision
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import f1, is_correct
from gate_rlm.data import write_jsonl, read_jsonl


def call_model(query: str, context: str = "", role: str = "root") -> dict:
    """Call local Qwen model via HTTP."""
    url = "http://localhost:8005/v1/chat/completions"

    if role == "root":
        prompt = f"Answer the following question based on the context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:  # worker
        prompt = f"Answer the question using ONLY the passages below. Reply with just the answer or 'NOT FOUND'.\n\nPassages:\n{context}\n\nQuestion: {query}\nAnswer:"

    payload = {
        "model": "mlx-community/Qwen3.5-2B-4bit" if role == "worker" else "mlx-community/Qwen3.5-35B-A3B-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256 if role == "root" else 128,
        "temperature": 0,
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            tokens = result.get("usage", {}).get("total_tokens", 0)
            return {"answer": answer, "tokens": tokens}
    except Exception as e:
        return {"answer": "", "tokens": 0, "error": str(e)}


def load_hotpotqa(limit: int = 10) -> list[dict]:
    """Load HotpotQA questions."""
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
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "hotpotqa_eval_results.jsonl"

    print("Loading HotpotQA questions...")
    questions = load_hotpotqa(limit=10)
    print(f"Loaded {len(questions)} questions")

    print("\nRunning evaluation...")
    results = []

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['qid']}: {q['query'][:60]}...")

        # Call root model for initial answer
        root_result = call_model(q["query"], q["context"], role="root")
        root_answer = root_result.get("answer", "")
        root_tokens = root_result.get("tokens", 0)

        # If root couldn't answer, try worker on context
        if not root_answer or "not found" in root_answer.lower():
            worker_result = call_model(q["query"], q["context"][:2000], role="worker")
            final_answer = worker_result.get("answer", "")
            worker_tokens = worker_result.get("tokens", 0)
        else:
            final_answer = root_answer
            worker_tokens = 0

        result = {
            "qid": q["qid"],
            "dataset": "hotpotqa",
            "query": q["query"],
            "gold_answer": q["gold_answer"],
            "answer": final_answer,
            "total_tokens": root_tokens + worker_tokens,
            "root_tokens": root_tokens,
            "worker_tokens": worker_tokens,
        }
        results.append(result)

        with open(out_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"\nWrote {len(results)} results to {out_file}")

    # Evaluate
    print("\nEvaluating results...")
    df = pd.DataFrame(results)

    df["correct"] = [
        is_correct(a or "", g, "hotpotqa")
        for a, g in zip(df.answer, df.gold_answer)
    ]
    df["f1"] = [f1(a or "", g) for a, g in zip(df.answer, df.gold_answer)]

    accuracy = df["correct"].mean()
    f1_score = df["f1"].mean()
    avg_tokens = df["total_tokens"].mean()

    summary = f"""
====== HotpotQA Direct Evaluation (Local Qwen Models) ======

ACCURACY Metrics:
  - Accuracy:     {accuracy:.3f}
  - F1 Score:     {f1_score:.3f}

EFFICIENCY Metrics:
  - Avg Tokens (root + worker): {avg_tokens:.0f}
  - Total Tokens Used: {df['total_tokens'].sum():.0f}

Questions evaluated: {len(df)}

Root model: mlx-community/Qwen3.5-35B-A3B-4bit
Worker model: mlx-community/Qwen3.5-2B-4bit
Endpoint: http://localhost:8005/v1/chat/completions
"""

    summary_file = out_dir / "hotpotqa_eval_summary.txt"
    summary_file.write_text(summary)
    print(summary)
    print(f"Summary report -> {summary_file}")


if __name__ == "__main__":
    main()
