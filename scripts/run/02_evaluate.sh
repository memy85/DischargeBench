#!/bin/bash
# LLM-judge evaluation of one educator model's runs (judge and settings: config/config.yaml).
# Usage: bash scripts/run/02_evaluate.sh <educator name>
# Serves the judge for TCS / Conversation Quality / CLSA, and the patient for SCS (the
# patient re-answers the six questions from the conversation only).
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL="$1"; EVAL="$ROOT/src/evaluation"
PIDS=(); trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=$EVAL_GPUS_JUDGE vllm serve "$JUDGE_HF" --served-model-name "$JUDGE_NAME" \
    --revision "$JUDGE_REVISION" --port "$JUDGE_PORT" "${JUDGE_ARGS[@]}" \
    > "$LOGS/vllm_judge.log" 2>&1 & PIDS+=($!)
CUDA_VISIBLE_DEVICES=$EVAL_GPUS_PATIENT vllm serve "$VP_HF" --served-model-name "$VP_NAME" \
    --port "$VP_PORT" "${VP_ARGS[@]}" > "$LOGS/vllm_patient.log" 2>&1 & PIDS+=($!)
wait_for_server "$JUDGE_PORT" "judge"; wait_for_server "$VP_PORT" "patient"

for RUN in "${RUNS[@]}"; do
    ARGS=("evaluation.models=['${MODEL}__${RUN}']" "evaluation.eval_port=$JUDGE_PORT"
          "evaluation.comprehension.vp_port=$VP_PORT")
    python "$EVAL/topic_checklist_raw_score.py"        "${ARGS[@]}"   # TCS judgments
    python "$EVAL/conversation_quality.py"             "${ARGS[@]}"   # Conversation Quality
    python "$EVAL/safety_commission_raw_score.py"      "${ARGS[@]}"   # CLSA claims + verdicts
    python "$EVAL/comprehension_score.py"              "${ARGS[@]}"   # SCS answers + grades
    # CPU scoring of the raw judgments
    python "$EVAL/topic_checklist_calculate_score.py"      "${ARGS[@]}"
    python "$EVAL/conversation_quality_calculate_score.py" "${ARGS[@]}"
    python "$EVAL/safety_commission_calculate_score.py"    "${ARGS[@]}"
done
