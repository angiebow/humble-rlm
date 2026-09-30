# DRAGIN-RLM: hardware validation results

Machine: `jupyter-01` / `macos-proj-01` (256GB RAM, 28 cores). Model: `mlx-community/Qwen3.5-35B-A3B-4bit`,
loaded directly via mlx-lm (`src/dragin_rlm/attention_probe.py`), not through the litellm/REST bridge.
Worker sub-calls go through the regular litellm proxy (port 4000).

## Infrastructure

| Item | Status |
|---|---|
| Raw BrowseComp-Plus data | present |
| `data/processed/browsecomp.jsonl` (800 examples) | built |
| `data/processed/browsecomp_sample50.jsonl` (50 examples, seed=42) | built |
| litellm proxy (port 4000) | running, `qwen35-root` + `qwen35-worker` registered |

## `verify_against_fused()`

**PASS** — the unfused attention-probe patch reproduces the fused kernel's output exactly
(`max|delta logits| = 0`). This is the mandatory gate `experiments/run_dragin.py` runs
automatically before any real generation; it refuses to proceed if this fails.

## Bugs found and fixed on real hardware

None of these were catchable without actually running against Qwen3.5-35B-A3B on real
Apple Silicon — `verify_against_fused()`'s own short test prompt (~12 tokens) didn't
exercise the code paths that broke.

1. **Dead instance-level monkeypatch.** `attn.__call__ = patched_call` sets an *instance*
   attribute, but Python looks up dunder methods like `__call__` on the *type* for the
   implicit call syntax (`attn(x, ...)`) — the patch silently never ran. `verify_against_fused()`
   passed with `max|delta logits| == 0`, which in hindsight was the tell: a real
   independent reimplementation essentially never reproduces a fused kernel bit-for-bit.
   Caught for real by `generate_with_probe`'s own "captured nothing" guard.
   **Fix:** patch `type(attn).__call__` instead, guarded so only the targeted instance
   diverges from the true original.

2. **O(L·S) memory blowup.** The unfused reimplementation materialized the *full*
   attention matrix every forward pass, but only the last query position's row is ever
   read. During prefill (L = S = prompt length, tens of thousands of tokens for
   BrowseComp-Plus documents) this allocated memory quadratic in context length:
   observed `[metal::malloc] Attempting to allocate 211645384832 bytes`.
   **Fix:** compute the real output via the original fused kernel (O(L) memory) and only
   manually compute the one attention row actually needed (O(S) memory).

3. **bfloat16 → numpy crash.** `np.asarray()` on an `mx.array` of dtype bfloat16 raises
   `ValueError: 'bfloat16' is not a valid PEP 3118 buffer format string` — numpy has no
   native bfloat16 buffer-protocol equivalent, and the model runs in bf16.
   **Fix:** keep the attention-probe readout and logits in float32 (they're never fed
   back into the model, only read out for RIND/QFS).

4. **No chunked prefill.** The custom generation loop fed the entire prompt through
   `model()` in a single call. One BrowseComp-Plus example measured at **81,305 prompt
   tokens** — a single monolithic forward pass of that length silently killed the process
   with no Python traceback at all. `mlx_lm.generate_step` avoids exactly this via
   `prefill_step_size` chunking (default 2048); the custom loop had skipped it entirely.
   **Fix:** chunk the prompt into `prefill_step_size`-token pieces, mirroring
   `mlx_lm.generate_step`; only the final chunk's output feeds sampling/probing.

5. **Model loaded twice.** `experiments/run_dragin.py` loaded the root model once for the
   real run, then `verify_against_fused()` loaded a *second* full ~20GB+ copy internally
   just for its own check. Diagnosed via a standalone script that loaded the model once
   and completed the same 81k-token document cleanly; the double-load path exited with
   zero output and no traceback on the identical document.
   **Fix:** `verify_against_fused()` now accepts an already-loaded `model`/`tokenizer`
   and reuses them when given.

## Smoke test (`--limit 2`, real BrowseComp-Plus questions)

**0 errors.** Both examples ran end-to-end on real ~80K-token documents.

| | browsecomp-303 | browsecomp-469 |
|---|---|---|
| Gold answer | Luciana Lixandru | 978-0197549704 |
| Prompt length | 81,305 tokens | (similarly long) |
| Completion tokens | 256 (budget exhausted) | 256 (budget exhausted) |
| Latency | 81.9s | 112.8s |
| RIND triggers | 0 | 0 |
| Error | none | none |

The model's reasoning is substantively on track — browsecomp-303's raw output literally
names "**Luciana Lixandru**" mid-reasoning, matching gold — but neither example reached
the "So the answer is:" cue within the token budget, so the extracted `answer` field is
the raw unfinished reasoning rather than a clean short answer.

## Open items (tuning, not bugs)

- **`generate_length: 256`** (`configs/experiments/dragin_rlm.yaml`) is too short for this
  reasoning style over long documents. The paper's own values (64–128) were tuned for a
  much shorter, non-document-heavy setup. Try 512–1024.
- **`theta: 1.0`** never triggered a RIND retrieval in either example — the trigger path
  (QFS → BM25 slice → worker sub-call → resume) is implemented and error-free but still
  unexercised on real data. Needs either a lower threshold or a look at the actual RIND
  score distribution before concluding whether 1.0 is too high.
