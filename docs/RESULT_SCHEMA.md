# Result record schema

`experiments/run.py` appends one JSON object per (example, seed) to
`results/runs/<config>__<dataset>__<split>.jsonl`. Every condition writes the same
fields, so evaluation code never branches on mode. Fields that don't apply are `null`.

| Field | Meaning |
|---|---|
| `config`, `mode`, `seed` | experiment name, `direct` / `vanilla` / `observe` / `gate`, repeat index |
| `qid`, `dataset`, `split` | example id, `babilong` / `browsecomp_plus`, `val` / `test` |
| `length_bucket`, `complexity` | `short`/`long`, `simple`/`complex` (RQ3 breakdown) |
| `query`, `gold_answer`, `answer` | question, reference, model output |
| `route`, `route_reason`, `p_direct` | Stage 0 decision |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | all models plus router tokens |
| `cost_usd`, `priced_calls` | LiteLLM price estimate; compare `priced_calls` with `llm_calls` |
| `latency_s` | wall-clock seconds |
| `llm_calls`, `leaf_calls`, `root_iterations`, `by_model` | from recursive-llm stats |
| `stop_reason` | `sufficient`, `knee`, or `null` (never stopped) |
| `stop_at_tokens` | cumulative tokens when the stop fired |
| `forced_final` | FINAL was injected after the grace turns |
| `worker_calls_skipped` | sub-calls answered by the gate at zero cost after a stop |
| `worker_parse_failures` | worker replies that ignored the JSON format |
| `gold_seen_at_tokens` | cumulative tokens when all required gold evidence had been read (logging only) |
| `checkpoints` | list, one per worker sub-call (below) |
| `error`, `traceback` | set if the run failed; failed runs are kept as data |

Checkpoint fields: `idx`, `cum_tokens`, `cosine`, `rerank`, `conf`, `found`, `answer`,
`relevant` (the gate's judgement), `contains_gold`, `contains_evidence` (both
logging only).

**Leakage rule:** `contains_gold`, `contains_evidence` and `gold_seen_at_tokens` come from
dataset annotations and are written for evaluation only. No decision in `src/gate_rlm`
reads them. Keep it that way.
