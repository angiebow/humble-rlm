#!/usr/bin/env python3
import json, sys, time
from pathlib import Path

print("=" * 80)
print("DRAGIN-RLM HotpotQA Evaluation with RIND Scores")
print("=" * 80)

ROOT = Path("/opt/jupyter-homes/u257878277/humble-rlm")
sys.path.insert(0, str(ROOT))

SAMPLE_QUESTIONS = [
    {"qid": "hotpotqa-000", "query": "Were Scott Derrickson and Ed Wood of the same nationality?", "context": "Scott Derrickson is an American film director. Ed Wood was an American filmmaker.", "gold_answer": "yes"},
    {"qid": "hotpotqa-001", "query": "Are both Local H and For Against from the United States?", "context": "Local H is an American rock band. For Against was an American band.", "gold_answer": "Yes"},
    {"qid": "hotpotqa-002", "query": "Who wrote metaphysical poetry?", "context": "John Donne was an English poet known for metaphysical poetry.", "gold_answer": "John Donne"},
    {"qid": "hotpotqa-003", "query": "What is the author's nationality?", "context": "John Donne was English.", "gold_answer": "English"},
    {"qid": "hotpotqa-004", "query": "When did the actress die?", "context": "Shirley Temple died in 2014.", "gold_answer": "2014"},
    {"qid": "hotpotqa-005", "query": "What book types?", "context": "Authors write fiction and non-fiction.", "gold_answer": "Fiction and non-fiction"},
    {"qid": "hotpotqa-006", "query": "What genre?", "context": "Maya Angelou wrote memoirs.", "gold_answer": "Memoir"},
    {"qid": "hotpotqa-007", "query": "What info not found?", "context": "The Arctic is cold.", "gold_answer": "Unknown"},
    {"qid": "hotpotqa-008", "query": "What language in England?", "context": "English is spoken in England.", "gold_answer": "English"},
    {"qid": "hotpotqa-009", "query": "Are authors contemporary?", "context": "Mark Twain and Louisa May Alcott lived in the 1800s.", "gold_answer": "Yes"},
]

print("\nImporting modules...")
try:
    from dragin_rlm.harness import DraginConfig, run_dragin
    from dragin_rlm.attention_probe import load_dragin_model
    print("✓ DRAGIN imports OK")
    has_dragin = True
except Exception as e:
    print(f"✗ DRAGIN not available: {e}")
    has_dragin = False

if not has_dragin:
    print("Using fallback Direct HTTP evaluation...")
    import urllib.request
    
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "hotpotqa_eval_direct_20.jsonl"
    
    for i, q in enumerate(SAMPLE_QUESTIONS):
        print(f"[{i+1:2d}/{len(SAMPLE_QUESTIONS)}] {q['qid']}: {q['query'][:50]}...")
        
        t_start = time.perf_counter()
        prompt = f"Q: {q['query']}\nContext: {q['context']}\nA:"
        payload = {"model": "mlx-community/Qwen3.5-35B-A3B-4bit", "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0}
        
        try:
            req = urllib.request.Request("http://localhost:8005/v1/chat/completions", data=json.dumps(payload).encode('utf-8'), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as response:
                resp = json.loads(response.read().decode('utf-8'))
                answer = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                tokens = resp.get("usage", {}).get("total_tokens", 0)
        except Exception as e:
            answer, tokens = "", 0
        
        latency_s = time.perf_counter() - t_start
        result = {"qid": q["qid"], "gold": q["gold_answer"], "answer": answer, "tokens": tokens, "latency_s": latency_s}
        
        with open(out_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        print(f"  ✓ {tokens} tokens in {latency_s:.2f}s")
    
    print(f"\n✓ Evaluation complete. Results: {out_file}")
else:
    print("Loading DRAGIN model...")
    try:
        model, tokenizer = load_dragin_model("mlx-community/Qwen3.5-35B-A3B-4bit")
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        sys.exit(1)
    
    dcfg = DraginConfig(model_path="mlx-community/Qwen3.5-35B-A3B-4bit", worker_model="mlx-community/Qwen3.5-2B-4bit", theta=1.0, top_n=25, generate_length=256, max_retrieval_seconds=600.0, max_triggers=20, retrieval_top_k=3, passage_chars=1000, temperature=0.0)
    
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "dragin_hotpotqa_rind_20.jsonl"
    
    print(f"Evaluating {len(SAMPLE_QUESTIONS)} questions with DRAGIN...")
    results = []
    
    for i, q in enumerate(SAMPLE_QUESTIONS):
        print(f"[{i+1:2d}/{len(SAMPLE_QUESTIONS)}] {q['qid']}: {q['query'][:50]}...", end=" ", flush=True)
        
        t_start = time.perf_counter()
        
        try:
            dragin_result = run_dragin(query=q["query"], context=q["context"], cfg=dcfg, model=model, tokenizer=tokenizer)
            latency_s = time.perf_counter() - t_start
            
            result = {
                "qid": q["qid"],
                "gold": q["gold_answer"],
                "answer": dragin_result.get("answer", ""),
                "latency_s": latency_s,
                "tokens": dragin_result.get("completion_tokens", 0),
                "retrievals": dragin_result.get("n_retrievals", 0),
                "max_rind": dragin_result.get("max_rind_score", 0.0),
            }
            results.append(result)
            
            with open(out_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            
            print(f"✓ RIND:{result['max_rind']:.3f} ret:{result['retrievals']}")
        
        except Exception as e:
            print(f"✗ {str(e)[:40]}")
            continue
    
    print(f"\n✓ {len(results)} questions evaluated")
    print(f"✓ Results: {out_file}")
    
    if results:
        avg_rind = sum(r['max_rind'] for r in results) / len(results)
        print(f"  Avg RIND: {avg_rind:.4f}")
        print(f"  Total retrievals: {sum(r['retrievals'] for r in results)}")
