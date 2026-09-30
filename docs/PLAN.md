# Plan to submission (KNLP track, ACM SAC 2027, due Oct 2)

**Note:** this plan has been superseded in its data/model specifics by the pivot to
BrowseComp-Plus + a local Qwen 3.5 (MLX) model pair, described in `README.md`. The
roles/timeline structure below still applies; the specific commands and cut order in the
original version (BABILong-first, GPT-5 pair) do not — see README's Quick start instead.

## Current focus

Construct and validate **P(IK)-RLM** (router-only condition) against a 50-question random
sample of BrowseComp-Plus, before committing to the full ~800-question set split across
the team's four Mac Studios. See `README.md` for the exact construction and the
`make browsecomp-data` → `make pik-sample50` workflow.

## Status as of this writing

- P(IK)-RLM router implemented and working end-to-end (`src/gate_rlm/router.py`,
  `_logprob_pik`), including the local-MLX-specific fix (bypass litellm for the confidence
  probe; see README's "Known gotchas").
- litellm proxy cooldown bug found and fixed (`configs/litellm_proxy.yaml`) — this was
  responsible for a large share of the ~80% error rates seen in earlier full-shard runs;
  not yet re-validated live due to a remote infrastructure outage (all four Mac Studios
  unreachable at time of writing).
- `50`-question deterministic sampling tooling ready (`scripts/sample_dataset.py`,
  `make browsecomp-sample50`) so the fix can be validated cheaply before re-committing to
  a full run.
- Known, expected baseline behavior (not a bug): un-gated recursion (the `"rlm"` branch,
  since relevance/stopping are off for this condition) frequently hits
  `MaxIterationsError` — observed even on isolated hardware with no resource contention.

## DRAGIN-RLM

A second gating mechanism (`src/dragin_rlm/`), built alongside P(IK)-RLM rather than
replacing it as the current focus — see README's "DRAGIN-RLM" section for the full
construction and the two adaptations required for Qwen3.5's hybrid attention
architecture. Math (RIND/QFS/retrieval slicing) is implemented and unit-tested; the
mlx-lm attention-probe integration is written but unvalidated on real hardware (no Apple
GPU access in this dev environment) — run `verify_against_fused()` on the Mac Studios
before trusting any of its output.

## Immediate next steps (once Mac Studio access returns)

1. `make browsecomp-data && make browsecomp-sample50`
2. Start the MLX bridge, confirm the litellm cooldown fix actually resolves the
   `RateLimitError`/`NotFoundError` cascade seen before.
3. `make pik-sample50 && make pik-sample50-eval` — check accuracy + inference-time numbers
   look sane before scaling back up.
4. If clean: re-run the full ~800-question, 4-way sharded P(IK)-RLM pass across the team's
   Mac Studios (`scripts/shard_dataset.py --n-shards 4`), now with the fix applied.
5. Coordinate with teammates on Conventional-RLM / Reranker-RLM / FLARE-RLM / Confident-RLM
   (the other four conditions in the broader GATE-RLM study) once P(IK)-RLM's pipeline is
   confirmed solid — same codebase, different `configs/experiments/*.yaml`.
6. On real hardware: `python -c "from dragin_rlm.attention_probe import
   verify_against_fused; verify_against_fused('mlx-community/Qwen3.5-35B-A3B-4bit')"` —
   confirm PASS before any DRAGIN-RLM run is trusted (see README's DRAGIN-RLM section).

## Rules that still apply

- Gold annotations are **logging only**. Nothing in `src/gate_rlm` may decide using them.
- Never tune `tau_route` on the test split.
- Keep this repository private until review ends (double-blind), or share an anonymized
  mirror.
