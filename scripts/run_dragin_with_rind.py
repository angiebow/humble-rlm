#!/usr/bin/env python3
import json, sys, time
from pathlib import Path

ROOT = Path("/opt/jupyter-homes/u257878277/humble-rlm")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

print("=" * 80)
print("DRAGIN-RLM HotpotQA Evaluation with RIND Scores")
print("=" * 80)

SAMPLE_QUESTIONS = [
    {"qid": "hotpotqa-000", "query": "Were Scott Derrickson and Ed Wood of the same nationality?", "context": "Scott Derrickson is an American film director. Ed Wood was an American filmmaker.", "gold": "yes"},
    {"qid": "hotpotqa-001", "query": "Are both Local H and For Against from the United States?", "context": "Local H is an American rock band. For Against was an American band.", "gold": "Yes"},
    {"qid": "hotpotqa-002", "query": "Who wrote metaphysical poetry?", "context": "John Donne was an English poet known for metaphysical poetry.", "gold": "John Donne"},
]

print("\nImporting DRAGIN...")
sys.path.insert(0, str(ROOT / "src"))
print(f"Python path: {sys.path[:3]}")

try:
    from dragin_rlm.harness import DraginConfig, run_dragin
    from dragin_rlm.attention_probe import load_dragin_model
    print("✓ DRAGIN imported successfully")
    HAS_DRAGIN = True
except Exception as e:
    print(f"✗ DRAGIN import failed: {e}")
    HAS_DRAGIN = False

if not HAS_DRAGIN:
    print("Fallback to Direct HTTP...")
    import urllib.request
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "hotpotqa_direct_fallback.jsonl"
    
    for i, q in enumerate(SAMPLE_QUESTIONS):
        print(f"[{i+1}/{len(SAMPLE_QUESTIONS)}] {q['qid']}")
        t_start = time.perf_counter()
        
        try:
            prompt = f"Q: {q['query']}\nContext: {q['context']}\nA:"
            payload = {"model": "mlx-community/Qwen3.5-35B-A3B-4bit", "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0}
            req = urllib.request.Request("http://localhost:8005/v1/chat/completions", data=json.dumps(payload).encode('utf-8'), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as response:
                resp = json.loads(response.read().decode('utf-8'))
                answer = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                tokens = resp.get("usage", {}).get("total_tokens", 0)
        except Exception as e:
            answer, tokens = "", 0
        
        latency = time.perf_counter() - t_start
        result = {"qid": q["qid"], "gold": q["gold"], "answer": answer, "tokens": tokens, "latency": latency}
        
        with open(out_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        print(f"  ✓ {tokens} tokens in {latency:.2f}s")
    
    print(f"\n✓ Fallback complete: {out_file}")

else:
    print("\nLoading DRAGIN model...")
    try:
        model, tokenizer = load_dragin_model("mlx-community/Qwen3.5-35B-A3B-4bit")
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        sys.exit(1)
    
    dcfg = DraginConfig(
        model_path="mlx-community/Qwen3.5-35B-A3B-4bit",
        worker_model="mlx-community/Qwen3.5-2B-4bit",
        theta=1.0, top_n=25, generate_length=256, max_retrieval_seconds=600.0,
        max_triggers=20, retrieval_top_k=3, passage_chars=1000, temperature=0.0,
    )
    
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "dragin_hotpotqa_with_rind.jsonl"
    
    print(f"\nEvaluating {len(SAMPLE_QUESTIONS)} with DRAGIN+RIND...")
    results = []
    
    for i, q in enumerate(SAMPLE_QUESTIONS):
        print(f"[{i+1}/{len(SAMPLE_QUESTIONS)}] {q['qid']}", end=" ", flush=True)
        
        t_start = time.perf_counter()
        try:
            result_d = run_dragin(query=q["query"], context=q["context"], cfg=dcfg, model=model, tokenizer=tokenizer)
            latency = time.perf_counter() - t_start
            
            result = {
                "qid": q["qid"],
                "gold": q["gold"],
                "answer": result_d.get("answer", ""),
                "latency": latency,
                "tokens": result_d.get("completion_tokens", 0),
                "retrievals": result_d.get("n_retrievals", 0),
                "max_rind": result_d.get("max_rind_score", 0.0),
                "rind_tokens": result_d.get("rind_checked_tokens", 0),
            }
            results.append(result)
            
            with open(out_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            
            print(f"✓ RIND:{result['max_rind']:.3f} ret:{result['retrievals']}")
        
        except Exception as e:
            print(f"✗ {str(e)[:40]}")
            continue
    
    print(f"\n✓ {len(results)} evaluated")
    print(f"✓ Results: {out_file}")
    
    if results:
        avg_rind = sum(r['max_rind'] for r in results) / len(results)
        print(f"\nMetrics:")
        print(f"  Avg RIND: {avg_rind:.4f}")
        print(f"  Total retrievals: {sum(r['retrievals'] for r in results)}")
        print(f"  Avg latency: {sum(r['latency'] for r in results)/len(results):.2f}s")
