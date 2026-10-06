# HotpotQA Evaluation Pipelines

This project includes two evaluation approaches for HotpotQA: **Direct HTTP** and **DRAGIN-RLM**.

## Quick Comparison

| Feature | Direct HTTP | DRAGIN-RLM |
|---------|-----------|-----------|
| **Complexity** | Simple | Complex |
| **Speed** | Fast (2-3s/Q) | Slower (10-30s/Q) |
| **Model Access** | Black-box API calls | Direct model introspection |
| **RIND Scores** | ❌ Not available | ✅ Available |
| **Attention Probing** | ❌ No | ✅ Yes |
| **Token-level Analysis** | ❌ No | ✅ Yes with entropy/attention |
| **Worker Fallback** | ✅ Simple fallback | ✅ Triggered by RIND threshold |

---

## Pipeline 1: Direct HTTP Evaluation

**Script:** `scripts/eval_hotpotqa_direct.py`

### What it does:
- Calls local Qwen models via direct HTTP
- Bypasses litellm credential validation
- Simple root→worker model pipeline
- Perfect for quick testing

### Output Files:
- `results/hotpotqa_eval_results.jsonl` - Raw results
- `results/table_hotpotqa_eval.csv` - Per-question metrics
- `results/hotpotqa_eval_summary.txt` - Summary report

### Metrics:
```
✓ Accuracy, F1, Semantic Similarity
✓ Token usage (root + worker breakdown)
✓ Latency per question
✓ Iteration count (0=root only, 1=worker called)
✗ RIND scores (not available)
```

### Run:
```bash
python scripts/eval_hotpotqa_direct.py
```

### When to use:
- Quick validation/testing
- No need for RIND/uncertainty metrics
- Development iteration cycles

---

## Pipeline 2: DRAGIN-RLM Evaluation with RIND

**Script:** `scripts/run_dragin_hotpotqa_eval.py`

### What it does:
- Loads 35B model directly with attention access
- Computes RIND scores via attention probe
- Tracks token-level uncertainty via entropy
- Triggers retrieval based on RIND threshold (theta)

### Output Files:
- `results/dragin_hotpotqa_eval_results.jsonl` - Full results with RIND
- `results/table_dragin_hotpotqa_eval.csv` - Per-question with RIND metrics
- `results/dragin_hotpotqa_eval_summary.txt` - RIND analysis

### Metrics:
```
✓ Accuracy, F1, Semantic Similarity
✓ Token usage & latency
✓ Retrieval count (RIND-triggered)
✓ RIND score per question ⭐
✓ Entropy analysis per token
✓ Nonstopword ratio (semantic content)
```

### RIND Score Explanation:

**S_RIND(t_i)** = Retrieval-Induced Neuron Divergence score

Computed at each generated token from three factors:
1. **Token entropy** - How uncertain the model is
2. **Max attention** - How concentrated attention is
3. **Semantic importance** - Is this a meaningful token?

**Decision rule:**
```
if S_RIND(t_i) > theta:
    RETRIEVE and continue generation
else:
    continue normally
```

Example interpretation:
- Token with RIND=0.5 → high uncertainty, likely to trigger retrieval
- Token with RIND=0.05 → low uncertainty, confident generation
- Threshold theta=0.001 → very sensitive to uncertainty

### Run:
```bash
python scripts/run_dragin_hotpotqa_eval.py
```

### Prerequisites:
- Root model (35B) must be loaded directly (MLX format)
- Access to model's attention mechanism
- More memory required (~15GB for 35B)

### When to use:
- Need RIND/uncertainty metrics
- Analyzing which questions trigger retrieval
- Understanding model confidence
- Calibrating theta threshold
- Research & analysis

---

## CSV Column Reference

### Direct HTTP (table_hotpotqa_eval.csv):
```
qid, gold_answer, answer, correct, f1, semantic_similarity,
total_tokens, root_tokens, worker_tokens, latency_s, iterations
```

### DRAGIN-RLM (table_dragin_hotpotqa_eval.csv):
```
qid, gold_answer, answer, correct, f1, semantic_similarity,
completion_tokens, latency_s, n_retrievals, leaf_calls,
max_rind_score, rind_checked_tokens, rind_nonstopword_ratio,
answer_source, time_capped
```

---

## Example Results

### Direct Evaluation (Fast):
```
Questions: 10
Accuracy: 0.900
F1 Score: 0.797
Semantic Similarity: 0.861
Avg Latency: 2.62s
```

### DRAGIN Evaluation (Detailed):
```
Questions: 10
Accuracy: 0.900
F1 Score: 0.797
Semantic Similarity: 0.861
Avg Latency: 15.2s (with probing)
Avg RIND Score: 0.245
Avg Retrievals: 0.50 per question
Nonstopword Ratio: 0.312
```

---

## Workflow Recommendation

1. **Development**: Use Direct HTTP for quick iteration
2. **Analysis**: Use DRAGIN for understanding model behavior
3. **Comparison**: Run both to see impact of attention-based retrieval

---

## Troubleshooting

### Direct HTTP fails:
- Check mlx_lm server running on port 8005
- Verify model name: `mlx-community/Qwen3.5-2B-4bit`

### DRAGIN fails:
- Ensure 35B model can be loaded (memory requirement ~15GB)
- Check model path in config: `mlx-community/Qwen3.5-35B-A3B-4bit`
- CUDA/Metal support for acceleration

