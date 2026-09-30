# P(IK)-RLM: Confidence-Gated Routing for Recursive Language Models

Current focus of this repository: constructing and evaluating **P(IK)-RLM**, one condition
of a broader GATE-RLM study (see `docs/PLAN.md` for the full context), against
**BrowseComp-Plus**. Everything else in the original GATE-RLM design (relevance gate,
stopping rule, BABILong) still exists in the codebase but is out of scope for the current
work; this document describes only what's active.

## Motivation

Recursive Language Models (RLMs; Zhang, Kraska and Khattab, 2025) store a long prompt as a
variable in a Python REPL. The model writes code that splits it up and calls itself
recursively on the pieces, removing the context-length ceiling of a normal LLM call. But
the framework never answers one question: **does this query even need recursion?**

In the RLM loop, every query goes through the full recursive REPL process, even ones a
single direct read could answer more cheaply and just as accurately. P(IK)-RLM adds a
gate in front of that loop: a cheap, up-front confidence check that decides whether to
answer directly or hand off to the full recursive process.

## What P(IK)-RLM is

A single gating stage — the **router** — placed in front of an otherwise unmodified RLM
loop. No relevance filtering, no adaptive stopping inside the recursion: those are
separate conditions in the broader GATE-RLM study, disabled here so the router's own
effect is isolated.

**The construction, step by step** (`src/gate_rlm/router.py`, function `_logprob_pik`):

1. Take a short **preview** of the context (first ~4000 characters, not the full
   document) — the point of the router is to be cheap, so it never sees the whole thing.
2. Generate a candidate answer on that preview, with `logprobs=True`.
3. Compute **P(IK) = exp(mean(log P(token)))** over the generated tokens (Kadavath et al.,
   2022) — the model's own calibrated confidence, derived from its actual token
   probabilities rather than a verbalized self-rating.
4. If `P(IK) >= tau_route` (0.75) **and** the query doesn't match a "complex query" regex
   (aggregation, counting, comparison, etc.) **and** the model actually produced an
   answer → route **direct**, and reuse that same candidate answer as the final output.
   A confident example costs exactly one call.
5. Otherwise → route **rlm**: the full recursive REPL loop runs (root writes Python,
   delegates to worker sub-calls) until the model emits `FINAL()` on its own or hits
   `max_iterations` / the elapsed-time cap.

One implementation detail worth knowing: the confidence probe bypasses `litellm` and talks
to the local model server directly over HTTP (`_raw_logprob_probe`), because the MLX
backend's logprobs response omits the `token` field, which crashes `litellm`'s strict
response validation even though generation itself succeeds.

**Models:** Qwen 3.5, 2B (worker, bf16) + 35B-A3B (root/"thinker", 4-bit; MoE with ~3B
active parameters per token), served locally via `mlx_lm.server` + a `litellm` proxy
bridge (`configs/litellm_proxy.yaml`, `scripts/start_mlx_local.sh`). Thinking mode
disabled (`enable_thinking: false`) to match the team's agreed setting.

**Config:** `configs/experiments/pik_rlm.yaml` — `router: {enabled: true, method:
logprob, tau_route: 0.75}`, `relevance: {enabled: false}`, `stopping: {enabled: false}`.

## Evaluation: BrowseComp-Plus

[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) (Chen et al., 2025) is a
retrieval benchmark whose questions carry human-verified **gold** (sufficient) and
**evidence** (relevant) document labels. `scripts/prepare_browsecomp.py` builds each
example's context from gold + evidence + sampled hard-negative documents, shuffled so
neither position nor length gives away which documents matter.

Gold/evidence labels are **logging only** — they let us detect whether the pipeline
actually read the needle, but nothing in `src/gate_rlm` uses them to make a live decision.

**Indicators:** accuracy and inference time, computed per condition by
`eval/aggregate.py`'s `summarize()` (`acc_mean`/`acc_std`, `latency_p50`/`latency_p95`),
plus `direct_route_share` — the fraction of questions the router judged confident enough
to skip recursion, which is the headline number specific to this condition.

## Quick start

```bash
git clone <this repo> && cd humble-rlm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tokens]" && pip install mlx-lm "litellm[proxy]" "fastapi==0.115.6"
make test                      # no API calls; sanity check before anything else
```

Getting BrowseComp-Plus data (one-time, manual — see `scripts/prepare_browsecomp.py`):

```bash
git clone https://github.com/texttron/BrowseComp-Plus.git && cd BrowseComp-Plus
python scripts_build_index/decrypt_dataset.py --output data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv
cp data/browsecomp_plus_decrypted.jsonl topics-qrels/qrel_golds.txt topics-qrels/qrel_evidence.txt \
    <this-repo>/data/raw/
```

The current workflow — a 50-question random sample, for fast validation before committing
to the full set:

```bash
make browsecomp-data         # full 800-question set -> data/processed/browsecomp.jsonl
make browsecomp-sample50     # deterministic 50-question sample (seed=42, test split)

# start the MLX bridge yourself -- model downloads need manual confirmation
./scripts/start_mlx_local.sh
export OPENAI_API_BASE=http://localhost:4000/v1
export OPENAI_API_KEY=not-needed

make pik-sample50            # run P(IK)-RLM against the sample
make pik-sample50-eval       # accuracy + inference-time table
```

## Repository layout

```
configs/experiments/pik_rlm.yaml   the condition this repo currently focuses on
configs/litellm_proxy.yaml         routes root/worker aliases to two mlx_lm.server ports
scripts/start_mlx_local.sh         starts both MLX backends + the litellm proxy
scripts/prepare_browsecomp.py      BrowseComp-Plus: decrypted rows -> processed JSONL
scripts/sample_dataset.py          deterministic N-question random sample
scripts/shard_dataset.py           deterministic hash-based N-way split (multi-machine runs)
src/gate_rlm/
  router.py                        Stage 0: the P(IK) router itself (_logprob_pik)
  gate.py                          GateRLM: hooks into recursive-llm for the "rlm" branch
  pipeline.py                      one example -> one result record (route -> direct/rlm)
  data.py                          JSONL I/O, splits, gold-evidence fingerprints
experiments/run.py                 run a config over a data file (resumable, parallel)
eval/aggregate.py                  accuracy + latency table, paired significance tests
docs/PLAN.md                       full GATE-RLM study context and team roles
```

Everything else referenced by `docs/PLAN.md` and the wider GATE-RLM design (relevance
gate, stopping rule, BABILong, the other four conditions) still exists in the codebase —
`src/gate_rlm/relevance.py`, `src/gate_rlm/stopping.py`, `scripts/prepare_babilong.py`,
`configs/experiments/{conventional,reranker,flare,confident}_rlm.yaml` — just not the
current focus of this document.

## DRAGIN-RLM (separate construction, in progress)

`src/dragin_rlm/` is a second gating mechanism, adapted from DRAGIN (Su et al. 2024,
arXiv:2403.10081), built alongside P(IK)-RLM rather than as a GateRLM subclass — RIND
needs control *inside* a single generation call (token by token), which GateRLM's
`_call_llm`/`_call_leaf` hooks don't reach.

**The idea:** while the root model generates, score every token with
`S_RIND(t_i) = H_i · a_max(i) · s_i` — entropy × max self-attention any later token pays
back to it × a stopword mask (paper Eq. 5). Once a token's score crosses `theta`,
truncate the generation there, build a query from the top-n tokens the triggering token
itself attended to most (QFS, paper §3.2), and hand that query to a worker sub-call over
a slice of the document — then resume generation informed by what came back.

**Two adaptations, both load-bearing for this project's Qwen3.5-35B-A3B root model**
(see `src/dragin_rlm/attention_probe.py` for the full reasoning):

1. **No REST access to attention.** RIND needs raw attention weights; `mlx_lm.server`'s
   OpenAI-compatible API only ever returns the chosen token's logprob (the paper's own
   Limitations, §7, says exactly this: it doesn't work through APIs that hide attention
   scores). So DRAGIN-RLM's root model bypasses litellm entirely and talks to mlx-lm
   in-process (`attention_probe.load_dragin_model` / `generate_with_probe`) — this is why
   it has its own runner, `experiments/run_dragin.py`, instead of reusing
   `experiments/run.py`.
2. **Qwen3.5 is a hybrid linear-attention model**, not a plain Transformer: most decoder
   layers run a recurrent GatedDeltaNet mechanism with no pairwise attention matrix at
   all, and only every 4th layer (`full_attention_interval`) runs real softmax
   self-attention. "The last Transformer layer" (the paper's own choice, footnote 2)
   is reinterpreted as the last layer that actually has one. That layer's fused attention
   kernel is also monkey-patched (instance-level, one layer only) with an unfused
   reimplementation, since the fused kernel never materializes the softmax matrix a
   probe could read.

**Retrieval substitution:** the paper's "retrieval module" is BM25 over Wikipedia.
BrowseComp-Plus already hands each question a single assembled document (gold + evidence
+ hard negatives), so DRAGIN-RLM's "retrieval" (`src/dragin_rlm/retrieval.py`) is BM25
over *that* document's passages, feeding a worker LLM sub-call — the same "read a slice
of the stored context" RLM already does, just triggered by RIND/QFS instead of the
root's own code.

**Status:** the math (RIND, QFS, BM25 slicing) is implemented and unit-tested
(`tests/test_dragin_*.py`, no mlx required). The mlx-lm integration
(`attention_probe.py`, the unfused-attention patch) is written and reviewed against
mlx-lm's source but has never been executed — this development environment has no Apple
GPU access. **Before trusting any RIND score from a real run, call
`dragin_rlm.attention_probe.verify_against_fused(model_path)` on the machine that
actually has the model weights and confirm it prints PASS** — `experiments/run_dragin.py`
does this automatically and refuses to run if it fails (`--skip-verify` to override).

```bash
python experiments/run_dragin.py --config configs/experiments/dragin_rlm.yaml \
  --data data/processed/browsecomp_sample50.jsonl --split all \
  --out results/runs/dragin_rlm__browsecomp_sample50.jsonl
make dragin-sample50-eval
```

Sequential, not `--workers`-parallel like the other conditions: the root model is loaded
once, resident in this one process, for the same reason the litellm-based conditions
fan out and this one can't (see `experiments/run_dragin.py`'s docstring).

## Known gotchas (learned the hard way)

- **litellm deployment cooldowns**: with exactly one deployment per model (one
  `mlx_lm.server` instance each), litellm's default cooldown-after-failure behavior
  blacklists the sole deployment after a single transient timeout, cascading into
  `RateLimitError`/`NotFoundError` on every subsequent call. Fixed via
  `router_settings: {disable_cooldowns: true}` in `configs/litellm_proxy.yaml`.
- **`MaxIterationsError`**: un-gated recursion (the `"rlm"` branch here, since relevance
  and stopping are both off) frequently doesn't converge within `max_iterations: 20`.
  This shows up even on isolated hardware with no resource contention — it's a real
  behavior of un-gated recursion, not an infra artifact.
- **Shared hardware**: if running on a machine that also hosts other `mlx_lm.server`
  processes (yours or other users'), expect slower per-call latency and occasional
  timeouts under contention, on top of the `MaxIterationsError` baseline.
- **`nohup` in some pty-backed terminals** fails with "can't detach from console" — use
  a subshell instead: `(cmd &)`.

## Rules for the team

- Gold annotations are **logging only**. Nothing in `src/gate_rlm` may decide using them.
- Never tune `tau_route` on the test split.
- Keep this repository private until review ends (double-blind), or share an anonymized
  mirror.

## Built on

Zhang, Kraska & Khattab, *Recursive Language Models* (arXiv:2512.24601) ·
P(IK) / P(True) (Kadavath et al., 2022) · BrowseComp-Plus (Chen et al., 2025)
