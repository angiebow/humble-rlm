# GATE-RLM pipeline. Run `make help` for targets.
PY      ?= python
DATA    ?= data/processed/babilong.jsonl
WORKERS ?= 4
TH      := results/thresholds.json
RUNS    := results/runs

help:
	@grep -E '^[a-z0-9-]+:.*## ' Makefile | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

# --- Current focus: P(IK)-RLM validated against a 50-question BrowseComp-Plus
# sample, on the MLX bridge (scripts/start_mlx_local.sh). Run in order:
#   1. browsecomp-data      (needs data/raw/*decrypted.jsonl + qrel_* first --
#                             see scripts/prepare_browsecomp.py docstring)
#   2. browsecomp-sample50
#   3. start the MLX bridge yourself (model pulls need manual confirmation),
#      then: export OPENAI_API_BASE=http://localhost:4000/v1 OPENAI_API_KEY=not-needed
#   4. pik-sample50
#   5. pik-sample50-eval
BC_DATA   := data/processed/browsecomp.jsonl
BC_SAMPLE := data/processed/browsecomp_sample50.jsonl

browsecomp-data: ## build the full BrowseComp-Plus set (n=800) from data/raw
	$(PY) scripts/prepare_browsecomp.py --n 800 --max-docs 50 --max-chars-per-doc 12000 --out $(BC_DATA)

browsecomp-sample50: ## draw a reproducible 50-question random sample (seed=42, test split)
	$(PY) scripts/sample_dataset.py --data $(BC_DATA) --n 50 --seed 42 --out $(BC_SAMPLE)

pik-sample50: ## run P(IK)-RLM against the 50-question sample (needs MLX bridge + env vars set)
	$(PY) experiments/run.py --config configs/experiments/pik_rlm.yaml --data $(BC_SAMPLE) \
	  --split all --workers $(WORKERS) --out $(RUNS)/pik_rlm__browsecomp_sample50.jsonl

pik-sample50-eval: ## accuracy + inference-time table for the sample50 run
	$(PY) eval/aggregate.py --runs $(RUNS)/pik_rlm__browsecomp_sample50.jsonl --out results/table_pik_sample50.csv

pik-sample50-per-question: ## per-question accuracy + latency (not aggregated) for the pik sample50 run
	$(PY) eval/per_question.py --runs $(RUNS)/pik_rlm__browsecomp_sample50.jsonl --out results/table_pik_per_question.csv --semantic

# --- DRAGIN-RLM (src/dragin_rlm/): separate construction, needs direct mlx-lm
# access on the machine with the model weights (see experiments/run_dragin.py).
# Not part of the litellm/MLX-bridge path above -- run it there directly.
# theta gets swept (see DRAGIN_RLM_TEST_RESULTS.md), so every run/eval output
# below is theta-tagged, read live from the config each time, rather than a
# static name that silently goes stale the next time theta changes.
DRAGIN_THETA := $(shell grep -oE 'theta: [0-9.]+' configs/experiments/dragin_rlm.yaml | grep -oE '[0-9.]+')

dragin-sample50: ## run DRAGIN-RLM against the 50-question sample (loads the model directly; slow, sequential)
	$(PY) experiments/run_dragin.py --config configs/experiments/dragin_rlm.yaml --data $(BC_SAMPLE) \
	  --split all --out $(RUNS)/dragin_rlm__browsecomp_sample50_theta$(DRAGIN_THETA).jsonl

dragin-sample50-eval: ## accuracy + inference-time table for a dragin_rlm sample50 run
	$(PY) eval/aggregate.py --runs $(RUNS)/dragin_rlm__browsecomp_sample50_theta$(DRAGIN_THETA).jsonl \
	  --out results/table_dragin_sample50_theta$(DRAGIN_THETA).csv

dragin-sample50-per-question: ## per-question accuracy + latency (not aggregated) for the dragin sample50 run
	$(PY) eval/per_question.py --runs $(RUNS)/dragin_rlm__browsecomp_sample50_theta$(DRAGIN_THETA).jsonl \
	  --out results/table_dragin_sample50_theta$(DRAGIN_THETA)_per_question.csv --semantic

setup: ## install the package and dependencies
	$(PY) -m pip install -e ".[dev,tokens]"

test: ## unit + fake-model integration tests (no API calls)
	$(PY) -m pytest -q

data: ## build the BABILong subset (D1)
	$(PY) scripts/prepare_babilong.py --tasks qa1 qa2 --lengths 4k 32k 128k 512k --n 30

smoke: ## 3 real examples with the cheap dev pair (costs cents)
	$(PY) experiments/run.py --config configs/experiments/dev_cheap.yaml --data $(DATA) --split val --limit 3 --workers 1

observe-val: ## log all signals on validation (input to sweep + RQ2)
	$(PY) experiments/run.py --config configs/experiments/observe.yaml --data $(DATA) --split val --seeds 0 --workers $(WORKERS)

sweep: ## choose + freeze thresholds from the validation logs
	$(PY) experiments/sweep.py --logs $(RUNS)/observe__*__val.jsonl --out $(TH)

signals: ## RQ2 table: AUROC/AUPRC per signal
	$(PY) eval/signals.py --logs $(RUNS)/observe__*__val.jsonl

test-runs: ## baselines + GATE-RLM on the test split (3 seeds)
	for c in b1_direct b2_vanilla b3_budget gate_full; do \
	  $(PY) experiments/run.py --config configs/experiments/$$c.yaml --data $(DATA) --split test --thresholds $(TH) --workers $(WORKERS); \
	done

ablations: ## ablations on the test split (1 seed to save cost)
	for c in abl_no_router abl_no_relevance abl_no_stopping; do \
	  $(PY) experiments/run.py --config configs/experiments/$$c.yaml --data $(DATA) --split test --thresholds $(TH) --seeds 0 --workers $(WORKERS); \
	done

eval: ## main table, RQ3 breakdown, paired tests, figures
	$(PY) eval/aggregate.py --runs $(RUNS)/*__test.jsonl --reference b2_vanilla --method gate_full
	$(PY) eval/plots.py

.PHONY: help setup test data smoke observe-val sweep signals test-runs ablations eval \
	browsecomp-data browsecomp-sample50 pik-sample50 pik-sample50-eval pik-sample50-per-question \
	dragin-sample50 dragin-sample50-eval dragin-sample50-per-question
