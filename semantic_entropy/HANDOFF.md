# Handoff: semantic-entropy (SE) stop-rule study on a plain RLM (BrowseComp-Plus)

Read this first. It gives the goal, what was built and changed, what is verified, the current state on the server, and what to do next.

## 1. Goal

The user runs a plain Recursive Language Model (RLM) on 100 BrowseComp-Plus questions and wants to study **semantic entropy (Farquhar et al. 2024, Nature)** as a signal for stopping the RLM early. The plan:

1. Run the **plain benchmark with no stop rule** (set 1, later set 2). Log SE for every iteration of every question.
2. Judge the final answers and every per-turn draft.
3. Evaluate SE stop rules **offline** (which signal, threshold, min turn) for accuracy versus turns saved.

The user said explicitly: **"I just need semantic entropy, anything not related, drop."** FLARE and P(IK) were removed. Keep it that way. The user likes short answers and result tables. Never fabricate numbers.

## 2. Setup

- **Benchmark:** `Tevatron/browsecomp-plus`, test split (830 questions). `bcp_two_sets_50_seed20260930.json` holds 100 sampled IDs, `set1` and `set2` (50 each, disjoint, seed 20260930).
- **Models:** root `mlx-community/Qwen3.5-35B-A3B-4bit`, worker `mlx-community/Qwen3.5-2B-bf16`, both served by `mlx_lm.server` (mlx-lm 0.31.3, the latest). RLM library: `rlms==0.1.3` (imported as `rlm`).
- **Server:** a JupyterHub on an Apple **M3 Ultra, 256 GB** (`macos-proj-02`), `https://jupyter-03.aml1.id.iosda.org/user/u257878240/lab`. It is shared with other users (their `mlx_lm` processes show up in `pgrep`; never kill those). It needs the user's logged-in browser session.
- **Layout on the server:** working dir `~/work/rlm100` (files listed in section 3), venv `~/.venv` (Python 3.12.13; `torch` 2.12 and `transformers` 5.10 are already installed), HF cache `~/.cache/huggingface` (about 28 GB of models and dataset, plus the ~1.6 GB DeBERTa).
- `runs/shard0/servers.json` records the model server ports and PIDs.

## 3. Files in this zip

| File | Role |
|---|---|
| `run_rlm100.py` | Orchestrator. Writes 3 lane configs per set (17/17/16 questions), runs the lanes in parallel, zips a checkpoint per set. |
| `bcp_rlm_runner.py` | Runner for one lane: starts or reuses the servers, runs the RLM per question, runs the SE probe each iteration, parses answers, judges at the end. |
| `mlx_sample_server.py` | Launches `mlx_lm.server` with a patched sampler (see 5.1). The runner starts all servers through it. |
| `bcp_two_sets_50_seed20260930.json` | The 100 question IDs. |
| `README_rlm100.md` | Run instructions and settings (kept in sync with the code). |
| `requirements.txt` | Python dependencies. |
| `tests/rng_test.py`, `tests/nli_test.py` | Reproduce the sampler finding and the DeBERTa pair behavior. |

The user's project folder also has an `archive/` directory with older tooling (`compile_bcp_rlm.py`, etc.). It is unmodified and not needed for this task.

## 4. How one RLM iteration and the SE probe work

**RLM iteration.** The root model sees the question and its history, not the ~400K-token document context, which lives in a local Python REPL variable (`environment="local"`, `max_depth=1`). Each iteration the root writes code, the REPL runs it (it can call the worker model as a sub-call, one at a time), and the output is appended to the history. It ends when the model writes its final answer (`Explanation / Exact Answer / Confidence`), at 20 iterations, or at a 30-minute safety stop. Root prompts are short (about 7.5K tokens per call in one run).

**SE probe** (`run_probe` in the runner). Before each root call **from iteration 2**, never shown to the model:
1. Append a side question to the conversation: give the current best answer in a few words, or exactly `UNKNOWN`.
2. **Draft** (the "best answer"): one call at temperature **0.1** (the paper's setting), up to 48 tokens. It is what a stop rule returns.
3. **10 samples**: same side question, temperature 1.0, top-p 0.9, top-k 50, up to 48 tokens each.
4. Clean each sample (first line, strip "Answer:", quotes, max 200 chars), then normalize (`norm` lowercases, drops the articles a/an/the, strips punctuation).
5. **Cluster** by bidirectional entailment: `UNKNOWN` forms its own cluster, identical normalized text merges, otherwise `microsoft/deberta-large-mnli` (CPU, ~0.11 s per pass) compares a new sample with the first member of each cluster in both directions, **with the question prepended** to each answer. Strict rule: merge only if entailment both ways. Verdicts are cached per question.
6. Compute `se` (discrete: −Σ (count/n) ln(count/n)) and `se_w` (probability-weighted, see 5.3), plus `se_clusters`.
7. Stop check (only if `stop_rule` is set): stop if the draft is not UNKNOWN, no error, turn >= `min_turn`, and the chosen signal (`se` or `se_w`) <= `theta`. The draft becomes the final answer and the root call is skipped.

Each probe record (in the `probes` list of the per-question row) has: `turn`, `draft`, `draft_raw`, `unknown`, `se`, `se_w`, `se_clusters`, `se_answers` (cleaned samples), `se_logp` (per-sample sequence log-prob), `se_ntok` (per-sample token count), `se_calls` (DeBERTa passes used), `probe_s`, `error`. Raw sample text is **not** saved (optional improvement).

Config keys under `probe`: `draft_max_tokens` 48, `draft_temperature` 0.1, `se_samples` 10, `se_temperature` 1.0, `se_top_p` 0.9, `se_top_k` 50, `se_strict` (default true), `se_cluster` (`"deberta"` default or `"string"`), `se_nli_model`. `stop_rule` is `None` in the benchmark (`condition: "plain"`).

## 5. What was changed and what is verified

### 5.1 Sampler bug in mlx_lm.server 0.31.3 (verified, patched)
Repeated identical requests at temperature > 0 returned the **same** output every time (for example "Penguin" x6, "427" x6, even a 24-token sentence at T=1.5 x5, and identical gibberish at T=3). `seed` in the request and `n`, parallel requests and different server flags did not help. So the original SE samples were identical and **SE was always 0**. The same bug also made every RLM trajectory sampling replay identical draws.

- **Cause (verified at library level with `tests/rng_test.py`):** MLX's random state is per-thread. A fresh thread replays the same draws and seeding from another thread is ignored. (My simple per-request-seed wrapper did not fix the server, so the server's decode path is more involved than that.)
- **Fix:** `mlx_sample_server.py` replaces `mlx_lm.sample_utils.categorical_sampling` with a version that draws from a fresh explicit key (`mx.random.key(secrets.randbits(31))`). Verified on the 2B model: T=1.0 now varies, T=0.0 stays deterministic. The runner launches all servers through this wrapper and tags them `fresh-key`, so older servers are replaced automatically.
- **Consequence:** trajectories are now genuinely stochastic run to run. Earlier results are not directly comparable to new ones.

### 5.2 DeBERTa clustering (verified on pairs)
`tests/nli_test.py` results (a→b, b→a): Alcatraz/Alcatraz Island ENTAIL/ENTAIL, Alcatraz/Corteiz CONTR/CONTR, Corteiz/UNIVERSAL CONTR/NEUTR, Obama/Barack Obama ENTAIL/ENTAIL, US/United States ENTAIL/ENTAIL, Paris/Paris Hilton NEUTR/ENTAIL, Giusewa/Giusewa Pueblo NEUTR/ENTAIL. Strict mode therefore merges Alcatraz/Alcatraz Island, Obama/Barack Obama and US/United States, and splits the rest, including "X" versus "X Pueblo". In runs, strict mode still splits some elaborated variants ("Snapchat" vs "Snap Inc. (Snapchat)"). `se_strict: false` gives the paper's looser rule but would wrongly merge Paris with Paris Hilton. An LLM-judge entailment variant was tried earlier and removed (it split "Alcatraz" from "Alcatraz Island").

### 5.3 `se_w`, the paper's probability-weighted estimator (implemented, source verified only loosely)
Read from the PMC full text (PMC11186750) through a summarizing fetch tool, so **check Equations 2 to 5 yourself**. As read: sequence probability P(s|x) = exp(sum of token log-probs); cluster mass P(c|x) = sum of member sequence probabilities; normalize across sampled clusters; SE = −Σ P(C) log P(C). Discrete SE (count/M) is the paper's fallback. **Open point:** the fetched text was ambiguous about length normalization. The code uses the raw sum. `se_ntok` is logged so a length-normalized variant can be computed offline. Verified: the server's returned token log-probs **include the stop token** (2 entries for the one-token answer "Blue": the word and the stop token, id 248046). Log-prob entries only have `id` and `logprob` (no token text).

Other paper-aligned settings adopted: 10 samples, T=1, top-p 0.9, top-k 50, question in the entailment context, best answer at T=0.1.

### 5.4 Removed
FLARE (min/mean token prob, low tokens, the FLARE hint), P(IK) (the "is this correct?" call), and their config keys and stop-rule signals. The general trajectory logging (per-call root log-prob stats, evidence tracking, calls, trajectories) was left untouched; the user may want it dropped too. Ask.

## 6. Smoke-test results so far (only 2 questions, Q737 gold "Alcatraz", Q1155 gold "Giusewa")

| Run | Setup | Q737 | Q1155 |
|---|---|---|---|
| A | plain, old sampler | 7 iters, 359 s, answer contained Alcatraz (`em` 0, extra words) | 20 iters, 588 s, Giusewa (`em` 1) |
| B | SE θ=0, old sampler, string clustering | stopped turn 4, "Corteiz" wrong | stopped turn 3, "Guswinton" wrong |
| C | SE θ=0, patched sampler, LLM-judge clustering, 5 samples | stopped turn 9, Alcatraz, correct, 166 s | stopped turn 6, Giusewa, correct, 57 s |
| D | SE θ=0, DeBERTa strict, 5 samples | stopped turn 8, "Alcatraz logo", judged incorrect | stopped turn 3, "Santa Fe Village", wrong |
| E | SE θ=0, DeBERTa strict, 10 samples, `se_w` logged | stopped turn 3, "ZARA", wrong | finished itself at iteration 5, Giusewa, correct, 137 s; `se` never reached 0 |

**Reading (small sample, do not over-claim):** with working sampling, SE varies sensibly (0 to ~1.05). The recurring failure is a **confidently wrong early guess** (Corteiz, Santa Fe Village, Zara) where all samples agree, so SE = 0 and the θ=0 rule stops wrongly at turn 3 or 4. SE cannot detect consistent errors. `se` and `se_w` differ a lot (for example Q1155 turn 3: `se` 0.94 vs `se_w` 0.07), so thresholds are not interchangeable. Three of the four SE-stop runs (B, D, E) ended with at least one wrong early stop; run-to-run randomness means 2 questions prove nothing.

## 7. Current state on the server (verify, do not assume)

- The trimmed `bcp_rlm_runner.py` (0 references to FLARE or P(IK)) and the new `run_rlm100.py` were uploaded at about 16:53. The user uploads files by hand through the JupyterLab UI (the browser file picker cannot be driven by automation). An "Overwrite?" prompt once made an upload silently not apply, so **check with `grep -c -i flare bcp_rlm_runner.py`** (expect 0).
- A **stray smoke run** was started at about 16:49 with the old runner code and `run_tag=rlm100_final_smoke`. It has P(IK) fields and is not representative. At 16:56 it was at 1 of 2 questions. Q737 took 8 iterations; the parsed answer was a full sentence containing "Alcatraz" (`em` 0) because the model skipped the `Exact Answer:` label and the runner's fallback took the last line. Let it finish or stop it (`pkill -f bcp_rlm_runner` stops only runners, not the model servers).
- The patched root and worker servers may still be running from earlier runs; the runner reuses them if healthy and tagged `fresh-key`.

## 8. What to do next

1. Confirm the server files match this zip (`grep` counts, file sizes) and that no stray runner is running (`pgrep -f bcp_rlm_runner`).
2. **Smoke test the exact benchmark config:** `python run_rlm100.py 0 --dry` (regenerates lane configs), copy `bcp_rlm_rlm100_set1_r0_config.json` to a new file with a **new `run_tag`** (finished questions are skipped per `run_tag`), then `python bcp_rlm_runner.py <config> --smoke 2`. Launch pattern that works: `(python bcp_rlm_runner.py CFG --smoke 2 < /dev/null > runs/x.log 2>&1 &)`. Check that probes have no `p_ik`/`min_p`, that `se`, `se_w`, `se_logp`, `se_ntok` are present, and read the judged file.
3. Show the user the per-iteration `se`/`se_w` next to the samples, the final answers and the judge verdicts, then get their go-ahead.
4. **Full run, set 1:** `mkdir -p runs; (python run_rlm100.py 0 < /dev/null > runs/rlm100_orchestrator.log 2>&1 &)`. Monitor with `tail`, `results/*status.json` (`done/total`, `em`, `eta_h`, errors) and `runs/shard0/runner_*.log`. Resumable: rerun the same command. Rough estimate: 3 to 6 hours per set (about 8 min per question plain, 1 to 2 min of probes, 3 lanes sharing servers on a shared machine); refine from `eta_h`.
5. **Offline analysis** after the run (build these tables from `results/*_judged.jsonl`; never invent numbers):
   - Headline per set: `judge_acc`, `em`, `contains`, `answer_found`, mean iterations, mean wall time, errors/timeouts.
   - Signal quality: AUROC of `se` and `se_w` for "the draft at this turn is correct" (per-turn draft verdicts are in the judged rows; confirm their field names, they were not inspected).
   - Stop-rule comparison: rows are rules (signal, θ, min_turn, optionally "same draft on 2 consecutive turns"), columns are % stopped early, mean turns at stop, turns saved, accuracy at stop, and delta accuracy versus plain. Note `se` takes few values with 10 samples (0, 0.33, 0.50 ...), `se_w` is continuous and rarely exactly 0.
   - Optionally re-cluster the saved `se_answers` with other rules (string, DeBERTa loose, LLM judge) offline for a fair clustering comparison.
   - Only then decide, with the user, whether to run an online stop-rule condition.

## 9. Open questions for the user

- Which signal (`se` or `se_w`) and which stopping variant to test online; whether to add `min_turn` >= 4 or a two-consecutive-turns condition (the wrong stops all came at turns 3 to 4).
- Strict versus loose entailment (`se_strict`).
- Length-normalized `se_w`: confirm against the paper.
- Save raw SE sample text? Drop the remaining general logging (per-call root log-probs, evidence)? Run set 2?

## 10. Caveats and gotchas

- **Judge:** the local 35B model grades answers and is strict and noisy (it marked "Alcatraz logo" incorrect). `em` is very strict; `contains` is looser. Prefer `judge_acc`, but spot-check borderline verdicts.
- **Answer parsing:** if the model omits `Exact Answer:`, the fallback takes the last non-code line, which can be a whole sentence (hurts `em`).
- **`norm` quirk:** it removes the word "a", so the single-letter answer "A" normalizes to empty and counts as UNKNOWN.
- **Shared machine:** timings vary; do not kill other users' processes.
- **Driving the Jupyter terminal through Chrome automation:** click into the **terminal body** right before typing, or the keystrokes go to the file browser and open files as tabs. Use `browser_batch` for multi-line input (embedded newlines work there; a direct `type` call typed a literal `\n`). The `wait` action is capped at 10 s. `nohup` fails in this terminal, use the `( ... < /dev/null > log 2>&1 &)` pattern. Long commands echo before finishing; wait rather than retype. An auto-mode safety classifier sometimes blocks terminal actions (especially process kills and long jobs); do not work around it, ask the user (`/permissions`, or leave auto mode with Shift+Tab).
- **Files:** the user uploads through the JupyterLab file browser; verify each upload landed (mtime, `grep`).
- **Stale test files on the server:** `rng_test.py`, `nli_test.py`, `nli`/`wrapper` logs, and configs for earlier smoke `run_tag`s (`rlm100_se_smoke`, `rlm100_se_ent_smoke`, `rlm100_se_deb_smoke`, `rlm100_sew_smoke`, `rlm100_final_smoke`) are harmless.
