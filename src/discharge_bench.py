"""
discharge_bench.py

Runs all cases from the discharge bench dataset in parallel.
Each case simulates a full patient-educator conversation under
environment agent supervision.

Output: JSONL file, one record per case, readable with:
    pd.read_json("simulation_results.jsonl", lines=True)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent))

from case_session import CaseSession
from educator_agent import EducatorAgent
from utils import CONFIG, PROJECT_PATH, PROMPTS_PATH

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

DATA_PATH = PROJECT_PATH / "data/processed"

EDUCATOR_PROMPT_PATH = PROMPTS_PATH / CONFIG["prompts"]["educator"]


# ---------------------------------------------------------------------------
# Single-case simulation (automated AI educator)
# ---------------------------------------------------------------------------

def simulate_case(
    scenario: pd.Series,
    patient_backend:  str,
    patient_port:     int,
    educator_backend: str,
    educator_port:    int,
    educator_host:    str,
    eval_model:       str,
    env_agent_port:   int | None,
    env_agent_host:   str,
    max_retries:      int,
    max_turns:        int,
    min_turns_before_complete: int,
    conversation_starter: str,
) -> dict:
    note_id = scenario["note_id"]

    try:
        session = CaseSession(
            scenario=scenario,
            patient_backend=patient_backend,
            patient_port=patient_port,
            eval_model=eval_model,
            env_agent_port=env_agent_port,
            env_agent_host=env_agent_host,
            max_retries=max_retries,
            max_turns=max_turns,
            min_turns_before_complete=min_turns_before_complete,
        )

        educator = EducatorAgent(
            backend_str=str(educator_backend),
            backend_port=educator_port,
            discharge_note=session.discharge_note_context,
            prompt_path=str(EDUCATOR_PROMPT_PATH),
            client_params={"host": educator_host},
        )

        # the first user message, which triggers the educator's opening turn
        educator_message = educator.generate_response(conversation_starter)
        while not session.is_done:
            patient_reply = session.submit_educator_turn(
                educator_message,
                agent_prompt=educator.system_prompt,
            )
            if patient_reply is None or session.is_done:
                break
            educator_message = educator.generate_response(patient_reply)

        return session.to_record()

    except Exception as exc:
        return {
            "note_id":           note_id,
            "conversation":      [],
            "discarded":         [],
            "n_accepted":        0,
            "n_discarded":       0,
            "early_termination": False,
            "stop_reason":       None,
            "session_complete":  False,
            "error":             str(exc),
        }


# ---------------------------------------------------------------------------
# Parallel benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    test_file:        str,
    output_file:      str,
    patient_backend:  str       = "llama3",
    patient_port:     int       = 5001,
    educator_backend: str       = "llama3",
    educator_port:    int       = 5002,
    educator_host:    str       = "localhost",
    eval_model:       str       = "gpt-4o-mini",
    env_agent_port:   int | None = None,
    env_agent_host:   str       = "localhost",
    max_retries:      int       = 2,
    max_turns:        int       = 100,
    n_workers:        int       = 20,
    min_turns_before_complete: int = 10,
    conversation_starter: str = "Please begin the discharge education session.",
) -> None:
    testset = pd.read_json(test_file, lines=True, orient="records")
    total   = len(testset)
    print(f"Loaded {total} cases from {test_file}")
    print(f"Workers: {n_workers}  |  max_turns: {max_turns}  |  eval_model: {eval_model}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: completed cases are skipped. Errored records are removed from the file so the
    # re-run replaces them instead of appearing beside them.
    done_ids: set[str] = set()
    if output_path.exists():
        kept, n_error = [], 0
        with open(output_path) as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{output_path}:{i}: malformed record; fix or remove it before resuming") from e
                if rec.get("error") is not None:
                    n_error += 1
                    continue
                if rec["note_id"] in done_ids:
                    raise ValueError(f"{output_path}:{i}: duplicate completed record for {rec['note_id']}")
                done_ids.add(rec["note_id"])
                kept.append(line if line.endswith("\n") else line + "\n")
        if n_error:
            tmp = output_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as f:
                f.writelines(kept)
            os.replace(tmp, output_path)
            print(f"Resuming: removed {n_error} errored records; those cases will be re-run.")
        if done_ids:
            print(f"Resuming: {len(done_ids)} cases already recorded, will be skipped.")

    lock      = threading.Lock()
    completed = 0

    def _write(record: dict) -> None:
        with lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    shared_kwargs = dict(
        patient_backend=patient_backend,
        patient_port=patient_port,
        educator_backend=educator_backend,
        educator_port=educator_port,
        educator_host=educator_host,
        eval_model=eval_model,
        env_agent_port=env_agent_port,
        env_agent_host=env_agent_host,
        max_retries=max_retries,
        max_turns=max_turns,
        min_turns_before_complete=min_turns_before_complete,
        conversation_starter=conversation_starter,
    )

    pending = [row for _, row in testset.iterrows() if row["note_id"] not in done_ids]
    skipped = total - len(pending)
    if skipped:
        print(f"Skipping {skipped} already-completed cases; running {len(pending)}.")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(simulate_case, row, **shared_kwargs): row["note_id"]
            for row in pending
        }

        for future in as_completed(futures):
            record    = future.result()
            completed += 1
            _write(record)

            if record["error"]:
                status = f"ERROR: {record['error']}"
            elif record["early_termination"]:
                status = f"STOPPED ({record['stop_reason']})"
            elif record["session_complete"]:
                status = "COMPLETE"
            else:
                status = f"MAX_TURNS ({record['n_accepted']} accepted)"

            print(f"[{completed:>3}/{len(pending)}] {record['note_id']} -- {status}")

    print(f"\nDone. Results written to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path=PROJECT_PATH.joinpath("config").as_posix(), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    db  = cfg.discharge_bench
    sim = db.simulation

    test_file = PROJECT_PATH / "data/processed" / db.test_file

    # run_tag suffixes ONLY the output directory (e.g. "gpt-5.5__run1"); the
    # educator model actually served/called below is db.educator.backend,
    # unchanged. Lets repeated variance-study runs land in separate
    # pseudo-model dirs that src/evaluation/*.py scores unmodified via
    # evaluation.models=[gpt-5.5__run1, ...].
    output_dir = f"{db.educator.backend}__{db.run_tag}" if db.run_tag else db.educator.backend
    output_file = PROJECT_PATH / "data/processed/output" / output_dir / "simulation_results.jsonl"

    run_benchmark(
        test_file=str(test_file),
        output_file=str(output_file),
        patient_backend=db.patient.backend,
        patient_port=db.patient.port,
        educator_backend=db.educator.backend,
        educator_port=db.educator.port,
        educator_host=db.educator.host,
        eval_model=db.env_agent.eval_model,
        env_agent_port=db.env_agent.port,
        env_agent_host=db.env_agent.host,
        max_retries=db.env_agent.max_retries,
        max_turns=sim.max_turns,
        n_workers=sim.n_workers,
        min_turns_before_complete=sim.min_turns_before_complete,
        conversation_starter=sim.conversation_starter,
    )


if __name__ == "__main__":
    main()
