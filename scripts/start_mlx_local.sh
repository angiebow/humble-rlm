#!/usr/bin/env bash
# Starts the MLX serving bridge: two mlx_lm.server backends (root, worker)
# plus the litellm proxy that routes between them. Run from the repo root
# with the venv active. Each ollama pull/model download happens on first
# request to a not-yet-cached model -- expect a large one-time download for
# the root model the first time this runs.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
# enable_thinking:false matches the team's agreed setting (see teammate's
# mlx_lm.server invocation observed on the shared host); worker is bf16
# (not quantized) to match their precision too.
mlx_lm.server --model mlx-community/Qwen3.5-35B-A3B-4bit --port 8011 \
  --chat-template-args '{"enable_thinking": false}' \
  > logs/mlx_root.log 2>&1 &
echo "root server pid $!"

mlx_lm.server --model mlx-community/Qwen3.5-2B-bf16 --port 8002 \
  --chat-template-args '{"enable_thinking": false}' \
  > logs/mlx_worker.log 2>&1 &
echo "worker server pid $!"

sleep 2
litellm --config configs/litellm_proxy.yaml --port 4000 \
  > logs/litellm_proxy.log 2>&1 &
echo "litellm proxy pid $!"

echo "export OPENAI_API_BASE=http://localhost:4000/v1"
echo "export OPENAI_API_KEY=not-needed"
echo "set both env vars in your shell before running experiments/run.py"
