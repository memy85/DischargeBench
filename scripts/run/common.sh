# Shared helpers for the run scripts. Source, do not execute.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS="$ROOT/logs"; mkdir -p "$LOGS"

# Models (Hugging Face ids). The served names must match config/config.yaml.
VP_HF="meta-llama/Llama-3.3-70B-Instruct"      # simulated patient, served as "llama3"
ENV_HF="Qwen/Qwen3.5-9B"                        # in-loop Environment Agent, served as "Qwen3.5-9B"
JUDGE_HF="google/gemma-4-31B-it"                # evaluation judge, served as "gemma-4-31B-it"
JUDGE_REVISION="439edf5652646a0d1bd8b46bfdc1d3645761a445"
VP_PORT=5001; ENV_PORT=5002; EDUCATOR_PORT=5003; JUDGE_PORT=5002

# Educator models: served name -> HF id ("" = OpenAI API, needs OPENAI_API_KEY).
declare -A EDUCATOR_HF=(
  [Llama-3.3-70B-Instruct]="meta-llama/Llama-3.3-70B-Instruct"
  [Qwen3-32B]="Qwen/Qwen3-32B"
  [Qwen3-4B]="Qwen/Qwen3-4B"
  [medgemma-27b-text-it]="google/medgemma-27b-text-it"
  [medgemma-4b-it]="google/medgemma-4b-it"
  [gpt-5.5]="" [gpt-5.4-mini]="" [gpt-5.4-nano]=""
)
EDUCATORS=(gpt-5.5 gpt-5.4-mini gpt-5.4-nano Qwen3-32B Qwen3-4B medgemma-27b-text-it Llama-3.3-70B-Instruct medgemma-4b-it)
RUNS=(run1 run2 run3 run4 run5)

wait_for_server() {   # port name
    local n=0
    until curl -sf "http://localhost:$1/health" > /dev/null 2>&1; do
        sleep 10; n=$((n + 1))
        if (( n > 180 )); then echo "ERROR: $2 did not start within 30 min"; exit 1; fi
    done
    echo "$2 ready on port $1"
}
