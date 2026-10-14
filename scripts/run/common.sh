# Shared helpers for the run scripts. Source, do not execute.
# Models, ports, vLLM arguments and GPU assignments come from config/config.yaml.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGS="$ROOT/logs"; mkdir -p "$LOGS"
eval "$(python "$ROOT/scripts/run/config.py")"

wait_for_server() {   # port name
    local n=0
    until curl -sf "http://localhost:$1/health" > /dev/null 2>&1; do
        sleep 10; n=$((n + 1))
        if (( n > 180 )); then echo "ERROR: $2 did not start within 30 min"; exit 1; fi
    done
    echo "$2 ready on port $1"
}
