"""HotpotQA evaluation using local Qwen models via direct HTTP (no litellm)."""
import json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import f1, is_correct
from gate_rlm.data import write_jsonl, read_jsonl

def call_model(query: str, context: str = "", role: str = "root") -> dict:
    url = "http://localhost:8005/v1/chat/completions"
    prompt = f"Answer: {query}" if not context else f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    payload = {"model": "mlx-community/Qwen3.5-2B-4bit", "messages": [{"role": "user", "content": prompt}], "max_tokens": 128, "temperature": 0}
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return {"answer": result.get("choices", [{}])[0].get("message", {}).get("content", "").strip(), "tokens": result.get("usage", {}).get("total_tokens", 0)}
    except Exception as e:
        return {"answer": "", "tokens": 0, "error": str(e)}

def load_hotpotqa(limit: int = 10) -> list:
    from datasets import load_dataset
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    examples = []
    for i, ex in enumerate(dataset):
        if i >= limit: break
        context = "\n\n".join(f"{t}\n{' '.join(s)}" for t, s in zip(ex["context"]["title"], ex["context"]["sentences"]))
        examples.append({"qid": f"hotpotqa-{i:03d}", "query": ex["question"], "context": context, "gold_answer": ex["answer"]})
    return examples

out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "hotpotqa_eval_results.jsonl"

print("Loading HotpotQA...")
questions = load_hotpotqa(limit=10)
print(f"Loaded {len(questions)} questions\nRunning evaluation...")

results, total_tokens = [], 0
for i, q in enumerate(questions):
    result = call_model(q["query"], q["context"], role="root")
    result.update({"qid": q["qid"], "query": q["query"], "gold_answer": q["gold_answer"], "dataset": "hotpotqa"})
    results.append(result)
    total_tokens += result.get("tokens", 0)
    with open(out_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  [{i+1}/{len(questions)}] {q['qid']}")

print(f"\nWrote {len(results)} results\nEvaluating...")
df = pd.DataFrame(results)
df["correct"] = [is_correct(a or "", g, "hotpotqa") for a, g in zip(df.answer, df.gold_answer)]
df["f1"] = [f1(a or "", g) for a, g in zip(df.answer, df.gold_answer)]

summary = f"""
====== HotpotQA Direct Evaluation (Local Qwen) ======

ACCURACY: {df['correct'].mean():.3f}
F1 Score: {df['f1'].mean():.3f}
Avg Tokens: {df['tokens'].mean():.0f}
Total Tokens: {total_tokens:.0f}

Questions: {len(df)}
Endpoint: http://localhost:8005/v1/chat/completions
"""
(out_dir / "hotpotqa_eval_summary.txt").write_text(summary)
print(summary)
