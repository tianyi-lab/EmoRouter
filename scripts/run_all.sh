#!/bin/bash
# Serve one Omni-LLM with vLLM, run per-tier inference on all datasets, then route.
#
#   MODEL_PATH=/path/to/Qwen2.5-Omni-7B SERVED_NAME=qwen2.5-omni TAG=qwen25 bash scripts/run_all.sh
#
# Expects the datasets under data/ (see README) and ffmpeg on PATH.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
SERVED_NAME=${SERVED_NAME:-omni}
TAG=${TAG:-model}
NGPUS=${NGPUS:-1}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-32}
DATASETS=${DATASETS:-"meld iemocap mosei mosi"}

vllm serve "$MODEL_PATH" --served-model-name "$SERVED_NAME" --port "$PORT" \
  --trust-remote-code --tensor-parallel-size "$NGPUS" \
  --gpu-memory-utilization 0.95 --max-model-len 4096 > "vllm_${TAG}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -s "http://localhost:${PORT}/v1/models" | grep -q "$SERVED_NAME"; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited, see vllm_${TAG}.log"; exit 1; }
  sleep 10
done

for ds in $DATASETS; do
  python inference/run_${ds}.py --base_url "http://localhost:${PORT}/v1" \
    --model "$SERVED_NAME" --output_dir "outputs/${ds}_${TAG}" --max_workers "$WORKERS"
done

for ds in $DATASETS; do
  echo "=== ${ds} (${TAG}) ==="
  python router/emorouter.py --pred_dir "outputs/${ds}_${TAG}" --dataset "$ds"
done
