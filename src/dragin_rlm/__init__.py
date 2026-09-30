"""DRAGIN-RLM: a RIND/QFS-gated generation mechanism (Su et al. 2024, DRAGIN,
arXiv:2403.10081) adapted to this project's RLM setting.

This is a separate construction from gate_rlm/, not a GateRLM subclass -- see
harness.py's module docstring for why. It shares gate_rlm's data schema
(gate_rlm.data) and result-record conventions (gate_rlm.pipeline) so it plugs
into the same eval/aggregate.py, but the generation path is entirely its own:
direct mlx-lm access for the root model (attention_probe.py), not litellm/REST,
because DRAGIN's RIND signal needs per-token self-attention that an OpenAI-
compatible API does not expose (see the paper's own Limitations, §7).

Modules:
  stopwords.py       static English stopword list (substitutes the paper's spaCy
                      en_core_web_sm choice; see that file for why)
  rind.py             Real-time Information Needs Detection (paper §3.1, Eq 1-5):
                      pure math + an online accumulator for streaming generation
  qfs.py              Query Formulation based on Self-attention (paper §3.2)
  retrieval.py        BM25-lite slicing of the already-provided document, standing
                      in for the paper's BM25-over-Wikipedia (see module docstring)
  attention_probe.py  direct mlx-lm generation with per-token entropy + attention,
                      via an instance-level monkey-patch of the last full-attention
                      layer (Qwen3.5 is a hybrid linear/full-attention model)
  harness.py          the RIND-gated generation loop: early-stop-if-confident /
                      trigger-sub-call-if-uncertain, QFS-sliced worker sub-calls
  pipeline.py         run_example() producing gate_rlm-compatible result records
"""
