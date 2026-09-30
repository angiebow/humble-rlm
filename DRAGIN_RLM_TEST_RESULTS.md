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

## Full 50-question run (`theta: 1.0`, `generate_length: 256`)

`results/table_dragin_sample50_theta1_0_per_question.csv`. **0 errors, 1h30m total
(108s/question average).** `accuracy: 0.320`, `semantic_accuracy: 0.020`. Every single
question exhausted the 256-token budget without reaching `"So the answer is:"`, and
`n_retrievals = 0` for all 50 — `theta=1.0` never crossed once across 50 real questions.

## `generate_length` raised to 768; theta still uncalibrated (10 questions, seed=17)

`results/table_dragin_sample10c_theta1_0_per_question.csv` (predates the theta-tagged
naming convention's introduction, but run under `theta=1.0`). Built a
`threshold_diagnostic.py` report from this run's `max_rind_score` per question (tracked
for free alongside the existing trigger check): scores ranged **0.0013–0.0040**
(p50=0.0028, p75=0.0031) — `theta=1.0` was **~250x above the highest score ever
observed**, confirming the zero-trigger result was pure threshold miscalibration, not a
property of the model or task. Re-checked against the paper's own settings (Table 8/9):
DRAGIN's evaluation contexts run a few hundred to ~1500 tokens (Wikipedia, top-k=3,
100-token passages, generate_length 64–128) versus BrowseComp-Plus's 80K+-token
documents — softmax attention weights (one factor of `S_RIND`) shrink with the number of
competing key positions, so the paper's `theta ∈ [0.6, 1.5]` was never going to transfer
numerically to this context-length regime.

## Calibrated rerun (`theta: 0.003`, same 10 questions)

`results/table_dragin_sample10c_theta0.003_per_question.csv`. `theta` set to the
p50–p75 line of the measured distribution above. **0 errors, 58m25s total** (vs. ~20m for
the same 10 questions at `theta=1.0`).

| | theta=1.0 (no recursion) | theta=0.003 (recursion enabled) |
|---|---|---|
| accuracy | 0.100 | **0.100 (unchanged)** |
| semantic_accuracy | 0.000 | 0.000 (unchanged) |
| latency p50 | 118.1s | 159.0s (+35%) |
| latency p95 | 165.2s | **1145.8s (+593%)** |

**Real recursion happened for the first time** — 4 of 10 questions triggered at least
once, 2 of them (`browsecomp-1103`, `browsecomp-6`) hit the `max_triggers: 8` safety cap
(805.1s and 1424.6s respectively — each retrieval re-prefills the *entire* document from
scratch with no KV-cache reuse across segments, so cost compounds with trigger count).
**Accuracy did not move.** The single correct answer (`browsecomp-1103`) was already
correct in the `theta=1.0` run before any retrieval happened; recursion never flipped a
wrong answer to right, including the two 8-trigger cases. On this small sample, DRAGIN's
gating fired on genuine uncertainty signal (mechanism confirmed working end-to-end) but
that uncertainty didn't translate into better answers — cost went up substantially,
outcomes didn't change. Whether that holds at a larger `n`, or whether the worker's
answers on triggering questions are themselves the bottleneck (not the trigger timing),
is the open question.

A structural inefficiency, found while diagnosing why one question ran 46+ minutes: once
`max_triggers` is reached, the loop's cap check only runs *after* one more full segment
generates (`harness.py`'s `while budget > 0 and len(triggers) <= cfg.max_triggers`) —
that final segment isn't aware it's over budget until it finishes on its own (finds
another trigger, hits the answer cue, or exhausts the token budget). Not a correctness
bug (trigger *count* still caps at exactly `max_triggers`, confirmed: both capped
questions show `n_retrievals: 8` exactly), but it wastes one full prefill+generation
cycle's worth of time. Worth moving the check to before starting a new segment rather
than after.

## theta lowered to 0.0001 (below the observed floor); smoke test, 2 fresh questions

`results/table_dragin_smoke2_theta0.0001_per_question.csv` (`browsecomp-1058`,
`browsecomp-242`, seed=57 -- don't overlap with any earlier sample). **0 errors.**
Both questions hit `max_triggers: 8` (`root_iterations: 9`) and both `answer` fields
came back as the literal 7-character string `<think>` -- `completion_tokens: 19`
each, meaning essentially the entire `generate_length: 768` budget was consumed by
the 8 retrieval segments' own generated text before the model could even close its
first `<think>` tag, let alone answer. `f1: 0.0`, `semantic_accuracy: 0.000` for both.

| | theta=0.003 (10 q, 4 triggered) | theta=0.0001 (2 q, both capped) |
|---|---|---|
| n_retrievals | 0-8 (mixed) | 8, 8 (both capped) |
| latency | 81s-1425s | 841s, 1165s |
| answer content | present (partial reasoning) | empty (`<think>` only) |

Pushing theta far below the measured floor doesn't just increase retrieval
frequency, it starves the token budget entirely -- every question now pays the
`max_triggers` cost with nothing left to answer with. This is a clearer, harsher
version of the same finding as the theta=0.003 run: more recursion here made
things *strictly worse* (0/2 vs. partial credit before), not neutral. Confirms the
`generate_length`/`max_triggers` budget interaction is the real bottleneck, not
theta calibration precision -- retrieval segments need their own separate budget
from answer generation, or `max_triggers` needs to scale down as `theta` goes down,
otherwise a low enough theta always degenerates to this.

## Open items (tuning, not bugs)

- The `max_triggers` off-by-one above (real waste, not correctness-affecting).
- `n=10` is too small to conclude recursion doesn't help at all — needs a larger
  calibrated-theta run (the full 50, or another larger sample) before treating "accuracy
  unchanged" as a stable finding rather than small-sample noise.
- Worth inspecting the worker's actual answers on the 4 triggering questions directly —
  if the worker itself isn't resolving the uncertainty (e.g. QFS's raw-subword-token
  query, or BM25 slicing missing the relevant passage), that's a different fix than "theta
  is still wrong."
- New, sharper priority after theta=0.0001: separate the retrieval-segment token
  budget from the answer-generation budget (or shrink `max_triggers` as `theta`
  drops), since right now a low enough theta guarantees the model never reaches an
  answer at all, independent of whether retrieval helped.
