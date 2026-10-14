#!/bin/bash
# Simulate one educator model on the benchmark, once per run tag in config/config.yaml (runs).
# Usage: bash scripts/run/01_simulate.sh <educator name>   e.g. Qwen3-32B
# Servers, ports, vLLM arguments and GPU assignments: config/config.yaml (servers).
# Output: data/processed/output/<educator>__<run>/simulation_results.jsonl (resumable)
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL="$1"; MODEL_HF="${EDUCATOR_HF[$MODEL]}"
PIDS=(); trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=$SIM_GPUS_PATIENT vllm serve "$VP_HF" --served-model-name "$VP_NAME" \
    --port "$VP_PORT" "${VP_ARGS[@]}" > "$LOGS/vllm_patient.log" 2>&1 & PIDS+=($!)
CUDA_VISIBLE_DEVICES=$SIM_GPUS_ENV vllm serve "$ENV_HF" --served-model-name "$ENV_NAME" \
    --port "$ENV_PORT" "${ENV_ARGS[@]}" > "$LOGS/vllm_env_agent.log" 2>&1 & PIDS+=($!)
if [[ -n "$MODEL_HF" ]]; then
    CUDA_VISIBLE_DEVICES=$SIM_GPUS_EDU vllm serve "$MODEL_HF" --served-model-name "$MODEL" \
        --port "$EDU_PORT" "${EDU_ARGS[@]}" > "$LOGS/vllm_educator.log" 2>&1 & PIDS+=($!)
    wait_for_server "$EDU_PORT" "educator"
fi
wait_for_server "$VP_PORT" "patient"; wait_for_server "$ENV_PORT" "simulation orchestrator"

for RUN in "${RUNS[@]}"; do
    python "$ROOT/src/discharge_bench.py" "discharge_bench.educator.backend=$MODEL" "discharge_bench.run_tag=$RUN"
done
