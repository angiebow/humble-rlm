# rlm100: plain-RLM run on 100 BrowseComp-Plus questions

Runs the plain RLM on 100 BrowseComp-Plus questions (two disjoint random sets of 50) on one Mac Studio, with a checkpoint zip after each set. The per-turn semantic entropy is logged so semantic-entropy stop rules can be evaluated offline later.

## Files

| File | Role |
|---|---|
| `run_rlm100.py` | Orchestrator. Runs set 1 in 3 parallel lanes, zips a checkpoint, then does the same for set 2. This is the script you launch. |
| `bcp_rlm_runner.py` | Runner for one lane. Starts or reuses the model servers, runs the RLM on each question, logs hidden probes, parses answers, judges them at the end. |
| `bcp_two_sets_50_seed20260930.json` | The 100 question IDs: `set1` (50) and `set2` (50), sampled from the 830 test questions with seed `20260930`. |
| `mlx_sample_server.py` | Launches `mlx_lm.server` with a patched sampler. Stock mlx-lm 0.31.3 replays the same random draws on every request (and ignores `seed`), so repeated samples at temperature > 0 were identical and semantic entropy was always 0. The runner starts every model server through this file. |

All four files must be in the same folder, and the scripts must be run from that folder.

## Setup

```bash
pip install -r share/requirements.txt   # mlx-lm>=0.31, rlms==0.1.3, datasets, openai, pandas, psutil, huggingface_hub, torch, transformers
python share/verify_env.py 0            # optional check; use your server index
```

Requirements:
- Apple silicon with about 40 GB of free unified memory.
- The first run downloads about 27 GB (both models and the dataset).
- If Hugging Face rate-limits the download, run `export HF_TOKEN=...` before launching.

## Run

```bash
mkdir -p runs
nohup python run_rlm100.py 0 > runs/rlm100_orchestrator.log 2>&1 &
```

- Replace `0` with the server index (0–3). Every server runs the same 100 questions; the index only changes file names and paths.
- `python run_rlm100.py 0 --dry` writes the six lane configs without running anything.
- Create `runs/` first, as above. The shell redirect fails if the folder doesn't exist.

## What happens

1. **Phase 1 (`set1`):** the 50 IDs are split across 3 lanes (`ids[j::3]`, so 17/17/16 questions). Each lane is one `bcp_rlm_runner.py` process with its own config, `bcp_rlm_rlm100_set1_r{j}_config.json`. The lanes are started 20 s apart and share one root server and one worker server.
2. Each lane runs its questions, then judges them, then writes its `DONE` file.
3. When all 3 lanes are done, the phase is zipped into `rlm100_checkpoint_set1.zip`.
4. **Phase 2 (`set2`)** repeats the same steps and produces `rlm100_checkpoint_set2.zip`.

## Settings

| | |
|---|---|
| Root model | `mlx-community/Qwen3.5-35B-A3B-4bit` (served by `mlx_lm.server`) |
| Worker model | `mlx-community/Qwen3.5-2B-bf16` |
| RLM | official `rlms` 0.1.3, plain (no stop rule, no hints) |
| Turn budget | 20 turns per question |
| Timeouts | 30-min safety stop per question, 600 s per model call |
| Sub-calls | 1 worker per question, sequential |
| Thinking | off |
| Sampling | temp 0.7, top_p 0.8, top_k 20, repetition penalty 1.1 over 256 tokens, 2,048 max tokens per turn |

**Hidden probes.** Before every root turn from turn 2 onward, the runner makes side calls on the current conversation:
- **Draft answer:** the "best answer", one low-temperature generation (temperature 0.1, as in the semantic-entropy paper; set `probe.draft_temperature` to 0 for greedy), up to 48 tokens. It is the answer returned if a stop rule fires.
- **Semantic entropy (Farquhar et al. 2024):** 10 samples at temperature 1.0 (top-p 0.9, top-k 50), grouped into meaning clusters by bidirectional entailment. Two answers share a cluster only if `microsoft/deberta-large-mnli` (CPU, question prepended to each answer) finds entailment in both directions. Set `probe.se_strict` to `false` for the looser rule, or `probe.se_cluster` to `"string"` for plain string matching. Two values are logged per iteration: `se` (discrete, from cluster counts) and `se_w` (probability-weighted: each cluster's mass is the sum of its samples' joint sequence probabilities exp(sum of token log-probs), normalised across clusters). `se_logp` and `se_ntok` hold each sample's sequence log-prob and token count, so length-normalised variants can be computed offline. A stop rule can use either as `signal`.

These probe results are logged but never fed back to the model, so the trajectories stay plain RLM.

**Answer parsing.** The runner reads `Exact Answer:` and `Confidence:` even when both are on the same line. If the `Exact Answer:` label is missing, it takes the last non-code line of the response instead.

**Judging.** After all questions in a lane are done, the local root model grades every final answer with the BrowseComp judge prompt. It also grades every distinct per-turn draft, which is what the offline stop-rule analysis uses.

## Monitor

```bash
tail -f runs/rlm100_orchestrator.log                    # phase / lane / checkpoint events
cat results/bcp_rlm_rlm100_set1_r0_shard0_status.json   # done/total, EM so far, ETA, errors
tail -f runs/shard0/runner_rlm100_set1_r0.log           # one line per finished question
```

## Outputs

`K` is the server index, `S` is the set (`set1` or `set2`) and `j` is the lane (0–2).

| Path | Contents |
|---|---|
| `results/bcp_rlm_rlm100_S_rj_shardK.jsonl` | One row per question: answer, EM/F1, timings, turns, tokens, evidence touched, probes, errors |
| `results/..._judged.jsonl` | The same rows plus judge verdicts (final answer and per-turn drafts) |
| `results/..._status.json`, `..._DONE.json` | Live progress, and the final summary for the lane |
| `runs/shardK/trajectories_*.jsonl.gz` | Full RLM trajectories |
| `runs/shardK/calls_*.jsonl.gz` | Every model call (latency, tokens) |
| `runs/shardK/logprobs_*.jsonl.gz` | Root per-token log-probs |
| `runs/shardK/servers.json`, `*_server.log` | Model server ports, PIDs and logs |
| `rlm100_checkpoint_S.zip` | Everything above for one set, plus the configs and the question-ID file |

## Stop / resume

To stop, kill the orchestrator and the runner PIDs (the PIDs are printed in the orchestrator log).

To resume, run the same launch command again. Finished phases are skipped, and runners skip questions already in their `.jsonl`. If a lane exits without a `DONE` file, the orchestrator stops with exit code 1; run it again to resume.

To re-run only the judging for one lane:

```bash
python bcp_rlm_runner.py bcp_rlm_rlm100_set1_r0_config.json --judge-only
```
