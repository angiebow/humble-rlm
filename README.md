# GATE-RLM: Gated Adaptive Termination for Recursive Language Models

Target venue: Knowledge and Natural Language Processing (KNLP) track, ACM SAC 2027 (submission due **October 2, 2026**).

**Title options**
1. *Token Confidence and Evidence Sufficiency in Recursive Language Models*
2. *Adaptive Termination for Cost-Efficient Knowledge Retrieval with Recursive Language Models*

## Motivation

Recursive Language Models (RLMs; Zhang, Kraska and Khattab, 2025) store a long prompt as a
variable in a Python REPL. The model writes code that splits it up and calls itself
recursively on the pieces. That removes the context-length ceiling, but the framework never
answers one question: **when has the model read enough?**

In the RLM loop, the only stopping rule is the root model deciding to emit its final answer.
Nothing checks whether a retrieved chunk is relevant to the task, or whether the evidence
gathered so far is sufficient. The original authors list exploding sub-call costs as an
open problem. Our pilot runs showed:

- **(a)** RLMs beat a conventional LLM on long, complex queries but lose on short, simple
  ones, where direct completion is more accurate and cheaper;
- **(b)** RLMs retrieve a wide range of information, much of it not relevant to the task;
- **(c)** inference cost and latency are noticeably higher.

An LLM's native confidence is a next-token probability: how sure it is of the next word.
It says nothing about whether a chunk is relevant or whether the evidence is sufficient.
GATE-RLM bridges those quantities.

## Research question

**Which runtime signal best predicts that enough evidence has been gathered: embedding
similarity, reranker score, or model confidence?**

The pipeline also answers two supporting questions: does gating cut cost without cutting
accuracy (vs. vanilla RLM), and how does the benefit change with query length and complexity?

## Method

| Stage | What it decides | How |
|---|---|---|
| 0 Router | answer directly or recurse? | P(IK)-inspired self-assessment: context fits the window + simple query (optional LLM self-rating on a preview) |
| 1 Relevance gate | is this chunk worth passing to the root? | cosine prefilter (bge-small) → cross-encoder reranker (bge-reranker); irrelevant sub-call outputs are hidden from the root |
| 2 Stopping rule | has enough been read? | stop on **sufficiency** (confident, relevant, agreeing worker answers; FLARE / DeepConf style) **or** the **knee** of the relevant-evidence gain curve (Cormack & Grossman) |

After a stop, later sub-calls return immediately at zero cost. If the root keeps going past
a grace turn, `FINAL(best_answer)` is injected.

**Models:** GPT-5 (root) + GPT-5-mini (sub-calls), the original paper's configuration.
GPT-5 models generally don't return token log-probs, so worker confidence is *verbalized*
by default (structured JSON reply). A log-prob signal is available through a separate
non-reasoning scorer model (`confidence.source: logprob`).

Built on [grishahq/recursive-llm](https://github.com/grishahq/recursive-llm), pinned to a
fixed commit in `pyproject.toml`. `GateRLM` subclasses its `RLM` and hooks the worker call,
the root turn, and the REPL event stream (`src/gate_rlm/gate.py`).

## Data

| Set | Role | Ground truth used |
|---|---|---|
| [BABILong](https://huggingface.co/datasets/RMT-team/babilong) qa1, qa2 at 4k-512k | main, controllable (RQ1-3) | answer + supporting facts, recovered by aligning each sample with its `0k` version (`scripts/prepare_babilong.py`) |
| [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) subset | realistic confirmation | answer + human-verified *gold* (sufficient) and *evidence* (relevant) documents |

Validation/test splits are fixed by a hash of the question id. Thresholds are chosen on
validation only, then frozen.

## Metrics

| Group | Metrics |
|---|---|
| Effectiveness | accuracy (BABILong contains-match; EM/F1), evidence precision / recall of what the gate passed |
| Efficiency | total tokens, sub-calls, cost (USD), latency p50 / p95 |
| Trade-off | cost per correct answer, sub-call productivity, accuracy-cost Pareto curve, oracle-stop gap |
| Signal quality (RQ) | AUROC / AUPRC per signal for relevance and for sufficiency, premature-stop rate |
| Headline | **over-read ratio**: share of tokens spent after the evidence had already been read |
| Significance | McNemar (accuracy), paired bootstrap + Wilcoxon (cost), 3 seeds |

Baselines: direct LLM, vanilla RLM, RLM with hard budget caps, oracle stop (computed from
logs), plus ablations removing the router, the relevance filter, or the stopping rule.

## Quick start

```bash
git clone <this repo> && cd gate-rlm
python -m venv .venv && source .venv/bin/activate
make setup                     # pip install -e ".[dev,tokens]"
cp .env.example .env           # add OPENAI_API_KEY
make test                      # no API calls; includes a fake-model end-to-end run
make data                      # BABILong subset -> data/processed/babilong.jsonl
make smoke                     # 3 real examples on the cheap dev pair
```

Full pipeline (each step writes what the next one reads):

```bash
make observe-val   # log every signal on validation (no gating)
make sweep         # replay logs offline under a threshold grid -> results/thresholds.json
make signals       # RQ table: AUROC / AUPRC per signal
make test-runs     # baselines + GATE-RLM on test, thresholds frozen
make ablations
make eval          # results/table_main.csv, table_rq3.csv, paired_tests.json, figures
```

`make sweep` replays the observe-mode logs under every threshold setting instead of paying
for a live run per setting. The live test runs are the numbers we report.

## Repository layout

```
configs/default.yaml        all knobs; [SWEEP] marks thresholds chosen on validation
configs/experiments/        one file per condition (baselines, gate, ablations, dev)
src/gate_rlm/
  gate.py                   GateRLM: hooks into recursive-llm (worker, root, events)
  router.py                 Stage 0
  relevance.py              Stage 1 (cosine prefilter + cross-encoder)
  stopping.py               Stage 2 (sufficiency + knee), shared by live runs and replay
  confidence.py             structured worker reply, optional log-prob scorer
  pipeline.py               one example -> one result record
  data.py                   JSONL, splits, gold-evidence fingerprints, BABILong rules
scripts/                    dataset preparation
experiments/run.py          run a config over a split (resumable, parallel)
experiments/sweep.py        threshold selection on validation
eval/                       metrics, aggregation, RQ signal analysis, stats, figures
docs/PLAN.md                roles, timeline, sync points, cut order
docs/RESULT_SCHEMA.md       every field in a result record
paper/                      ACM template notes, final figures and tables
```

## Rules for the team

- Gold annotations are **logging only**. Nothing in `src/gate_rlm` may decide using them.
- Never tune on test. Thresholds come from `results/thresholds.json`, written once.
- Debug with `dev_cheap`; never report its numbers.
- Keep this repository private until review ends (double-blind), or share an anonymized mirror.

## Known limitations

- `GateRLM` supports `max_depth=1`, the paper's default (sub-calls are plain LM calls).
- The root can also read the document with REPL prints and regex; those reads count toward
  cost and gold detection but not toward the gate's checkpoints.
- If BABILong's `0k` and long configs don't align for a task, `prepare_babilong.py` skips
  those samples and reports how many. Check the counts before running.
- Replay-based threshold selection assumes the root would issue the same sub-calls; live
  gating changes what the root sees, so only live test runs are reported.

## Built on

Zhang, Kraska & Khattab, *Recursive Language Models* (arXiv:2512.24601) · FLARE (Jiang et al., 2023) ·
P(IK) / P(True) (Kadavath et al., 2022) · DeepConf (Fu et al., 2025) · CRAG (Yan et al., 2024) ·
Sufficient Context (Joren et al., 2025) · CALM (Schuster et al., 2022) · knee stopping method
(Cormack & Grossman, 2016) · BABILong (Kuratov et al., 2024) · BrowseComp-Plus (Chen et al., 2025).
