"""
config.py — config/config.yaml values as shell assignments for scripts/run/*.sh.

common.sh runs `eval "$(python scripts/run/config.py)"`. vLLM arguments become bash
arrays (use as "${VP_ARGS[@]}") so quoted values such as a JSON chat-template argument
survive word splitting.
"""
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils import CONFIG   # noqa: E402

s = CONFIG["servers"]
scalars = {
    "VP_NAME": s["patient"]["served_name"], "VP_HF": s["patient"]["hf_id"], "VP_PORT": s["patient"]["port"],
    "ENV_NAME": s["env_agent"]["served_name"], "ENV_HF": s["env_agent"]["hf_id"], "ENV_PORT": s["env_agent"]["port"],
    "EDU_PORT": s["educator"]["port"],
    "JUDGE_NAME": s["judge"]["served_name"], "JUDGE_HF": s["judge"]["hf_id"],
    "JUDGE_REVISION": s["judge"]["revision"], "JUDGE_PORT": s["judge"]["port"],
    "SIM_GPUS_PATIENT": s["gpus"]["simulate"]["patient"], "SIM_GPUS_ENV": s["gpus"]["simulate"]["env_agent"],
    "SIM_GPUS_EDU": s["gpus"]["simulate"]["educator"],
    "EVAL_GPUS_JUDGE": s["gpus"]["evaluate"]["judge"], "EVAL_GPUS_PATIENT": s["gpus"]["evaluate"]["patient"],
}
arrays = {
    "VP_ARGS": shlex.split(s["patient"]["vllm_args"]),
    "ENV_ARGS": shlex.split(s["env_agent"]["vllm_args"]),
    "EDU_ARGS": shlex.split(s["educator"]["vllm_args"]),
    "JUDGE_ARGS": shlex.split(s["judge"]["vllm_args"]),
    "EDUCATORS": [e["name"] for e in CONFIG["educators"]],
    "RUNS": CONFIG["runs"],
}

for k, v in scalars.items():
    print(f"{k}={shlex.quote(str(v))}")
for k, v in arrays.items():
    print(f"{k}=({' '.join(shlex.quote(str(x)) for x in v)})")
# educator served name -> HF id ("" = OpenAI API)
print("declare -A EDUCATOR_HF=(" + " ".join(
    f"[{shlex.quote(e['name'])}]={shlex.quote(e['hf_id'] or '')}" for e in CONFIG["educators"]) + ")")
