# Plan to submission (KNLP track, ACM SAC 2027, due Oct 2)

## Roles

| Owner | Track | Owns |
|---|---|---|
| Lead | Method and story | method spec, threshold decisions, scope cuts, Introduction, Method, abstract |
| Engineering | Pipeline | `src/gate_rlm/`, `experiments/`, running every configuration |
| Evaluation | Data, metrics, writing | `scripts/`, `eval/`, figures, Related Work, Setup, ACM template, anonymization |

## Timeline

| Day | Lead | Engineering | Evaluation |
|---|---|---|---|
| Sep 28 | freeze method spec | `make setup test`, confirm API key | `make data`, check skip counts |
| Sep 29 | draft Intro + Method | `make smoke` by early afternoon, then `make observe-val` | `make sweep signals`, start Related Work |
| Sep 30 | pick operating point from Pareto plot, draft Results | `make test-runs`, then `make ablations` | `make eval`, write Setup |
| Oct 1 | Discussion, Limitations, abstract, full edit | re-run broken configs, implementation details | check every number against result files, template, page count |
| Oct 2 | final read and **submit** | standby | proofread references |

## Sync points (15 min each)

1. **Sep 28 night:** method spec and result schema agreed (`docs/RESULT_SCHEMA.md`).
2. **Sep 29 early afternoon:** smoke test passes? If not, cut D2 immediately.
3. **Sep 29 night:** thresholds frozen in `results/thresholds.json`. No changes after this.
4. **Sep 30 evening:** results locked; writing only.
5. **Oct 1 evening:** full draft read by all three.

## Cut order if time runs short

1. BrowseComp-Plus (D2)
2. router self-assessment (keep the rule-based router)
3. ablations beyond `abl_no_relevance`
4. lengths above 128k

Never cut: vanilla RLM vs. GATE-RLM on BABILong, with the accuracy vs. cost plot.

## Cost control

- Validation: `observe` only, 1 seed. Thresholds come from offline replay, not live sweeps.
- Test: 3 seeds for `b1_direct`, `b2_vanilla`, `b3_budget`, `gate_full`; 1 seed for ablations.
- Use `dev_cheap` for every debugging run. Never report its numbers.
- `b1_direct` fails on contexts beyond GPT-5's window. That failure is a result; keep it.
