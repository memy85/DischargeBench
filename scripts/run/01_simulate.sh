#!/bin/bash
# Simulate one educator model on the 320-case benchmark, 5 unseeded runs.
# Usage: bash scripts/run/01_simulate.sh <educator served name>   e.g. Qwen3-32B
# GPUs (A100-80GB): patient TP=2 (GPUs 0,1), Environment Agent (GPU 2), local educator (GPUs 3,4).
# Output: data/processed/output/<educator>__run<r>/simulation_results.jsonl (resumable)
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL="$1"; MODEL_HF="${EDUCATOR_HF[$MODEL]}"
PIDS=(); trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$VP_HF" --served-model-name llama3 --port $VP_PORT \
    --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.95 \
    > "$LOGS/vllm_patient.log" 2>&1 & PIDS+=($!)
CUDA_VISIBLE_DEVICES=2 vllm serve "$ENV_HF" --served-model-name Qwen3.5-9B --port $ENV_PORT \
    --max-model-len 32768 --default-chat-template-kwargs '{"enable_thinking": false}' \
    > "$LOGS/vllm_env_agent.log" 2>&1 & PIDS+=($!)
if [[ -n "$MODEL_HF" ]]; then
    CUDA_VISIBLE_DEVICES=3,4 vllm serve "$MODEL_HF" --served-model-name "$MODEL" --port $EDUCATOR_PORT \
        --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.95 \
        > "$LOGS/vllm_educator.log" 2>&1 & PIDS+=($!)
    wait_for_server $EDUCATOR_PORT "educator"
fi
wait_for_server $VP_PORT "patient"; wait_for_server $ENV_PORT "environment agent"

for RUN in "${RUNS[@]}"; do
    python "$ROOT/src/discharge_bench.py" \
        "discharge_bench.educator.backend=$MODEL" "discharge_bench.educator.port=$EDUCATOR_PORT" \
        "discharge_bench.patient.port=$VP_PORT" "discharge_bench.env_agent.port=$ENV_PORT" \
        "discharge_bench.run_tag=$RUN"
done
