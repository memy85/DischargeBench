#!/bin/bash
# LLM-judge evaluation of one educator model's 5 runs (gemma-4-31B-it, temperature 0).
# Usage: bash scripts/run/02_evaluate.sh <educator served name>
# GPUs (A100-80GB): judge TP=2 (GPUs 0,1) for TCS / CQ / CLSA; for the reported Comprehension
# score the persona-ablated simulated patient is served as well (GPUs 2,3).
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL="$1"; EVAL="$ROOT/src/evaluation"
PIDS=(); trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$JUDGE_HF" --served-model-name gemma-4-31B-it \
    --revision "$JUDGE_REVISION" --port $JUDGE_PORT --tensor-parallel-size 2 \
    --max-model-len 65536 --gpu-memory-utilization 0.95 > "$LOGS/vllm_judge.log" 2>&1 & PIDS+=($!)
CUDA_VISIBLE_DEVICES=2,3 vllm serve "$VP_HF" --served-model-name llama3 --port $VP_PORT \
    --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.95 \
    > "$LOGS/vllm_patient.log" 2>&1 & PIDS+=($!)
wait_for_server $JUDGE_PORT "judge"; wait_for_server $VP_PORT "patient"

for RUN in "${RUNS[@]}"; do
    ARGS=("evaluation.models=['${MODEL}__${RUN}']" "evaluation.eval_model=gemma-4-31B-it"
          "evaluation.decompose_model=gemma-4-31B-it" "evaluation.eval_port=$JUDGE_PORT"
          "evaluation.concurrency=16")
    python "$EVAL/topic_checklist_raw_score.py"        "${ARGS[@]}"   # TCS judgments
    python "$EVAL/conversation_quality.py"             "${ARGS[@]}"   # CQ (appendix)
    python "$EVAL/safety_commission_raw_score.py"      "${ARGS[@]}"   # CLSA claims + verdicts
    # Comprehension: patient re-answers the six questions from the conversation only
    COMPREHENSION_STRIP_CLINICAL=1 python "$EVAL/comprehension_score.py" "${ARGS[@]}" "evaluation.vp_port=$VP_PORT"
    # CPU scoring of the raw judgments
    python "$EVAL/topic_checklist_calculate_score.py"      "${ARGS[@]}"
    python "$EVAL/conversation_quality_calculate_score.py" "${ARGS[@]}"
    python "$EVAL/safety_commission_calculate_score.py"    "${ARGS[@]}"
done
