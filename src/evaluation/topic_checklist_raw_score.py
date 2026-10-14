"""
topic_checklist_raw_score.py

For each model in cfg.evaluation.models, loads simulation results and evaluates
each conversation against the discharge note using the topic checklist prompt.
Raw LLM responses (yes/no + reason per question) are saved to:
    data/processed/output/{model}/topic_checklist_raw__{eval_model}.jsonl
(tagged by judge model; topic_checklist_calculate_score.py reads it).

Usage:
    # judge served on eval_port
    python src/evaluation/topic_checklist_raw_score.py "evaluation.eval_model=gemma-4-31B-it"

    # single model override:
    python src/evaluation/topic_checklist_raw_score.py "evaluation.models=[medgemma-4b-text-it]"
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

import hydra
import jinja2
import pandas as pd
from omegaconf import DictConfig
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_PATH, PROMPTS_PATH

sys.path.insert(0, str(Path(__file__).parent))
import judge

OUTPUT_ROOT  = PROJECT_PATH / "data/processed/output"
DATA_PATH    = PROJECT_PATH / "data/processed"


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class ChecklistAnswer(BaseModel):
    answer: Literal["yes", "no"]
    reason: str

class TopicChecklistResponse(BaseModel):
    Q1:   ChecklistAnswer
    Q2:   ChecklistAnswer
    Q2_1: ChecklistAnswer
    Q2_2: ChecklistAnswer
    Q2_3: ChecklistAnswer
    Q2_4: ChecklistAnswer
    Q3:   ChecklistAnswer
    Q3_1: ChecklistAnswer
    Q3_2: ChecklistAnswer
    Q3_3: ChecklistAnswer
    Q4:   ChecklistAnswer
    Q4_1: ChecklistAnswer
    Q4_2: ChecklistAnswer
    Q4_3: ChecklistAnswer
    Q5:   ChecklistAnswer
    Q5_1: ChecklistAnswer
    Q5_2: ChecklistAnswer
    Q5_3: ChecklistAnswer
    Q6:   ChecklistAnswer
    Q6_1: ChecklistAnswer
    Q6_2: ChecklistAnswer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(model: str) -> list[dict]:
    path = OUTPUT_ROOT / model / "simulation_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No simulation results found for model '{model}' at {path}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    valid   = [r for r in records if not r.get("error")]
    skipped = len(records) - len(valid)
    if skipped:
        print(f"  [{model}] Skipped {skipped} errored records.")
    return valid


def load_note_lookup(test_file: str) -> dict[str, str]:
    """Returns a dict mapping note_id -> discharge note text."""
    df = pd.read_json(test_file, lines=True, orient="records")
    return dict(zip(df["note_id"], df["text"]))


def load_profile_lookup(test_file: str) -> dict[str, dict]:
    """Returns a dict mapping note_id -> patient_profile."""
    df = pd.read_json(test_file, lines=True, orient="records")
    return dict(zip(df["note_id"], df["profile"]))


def format_conversation(conversation: list[dict]) -> str:
    return "\n".join(f"{turn['role'].capitalize()}: {turn['message']}" for turn in conversation)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate_one(
    semaphore: asyncio.Semaphore,
    record: dict,
    patient_note: str,
    patient_profile: dict,
    template: jinja2.Template,
    client: AsyncOpenAI,
    eval_model: str,
    temperature: float,
) -> dict:
    conversation_history = format_conversation(record["conversation"])
    prompt = template.render(
        patient_note=patient_note,
        conversation_history=conversation_history,
    )

    async with semaphore:
        try:
            parsed = await judge.structured_call(
                client, eval_model, prompt, TopicChecklistResponse,
                temperature=temperature,
            )
        except openai.BadRequestError as e:
            # The judge's context window can't hold this case's note + conversation.
            # Abstain visibly instead of crashing the stage.
            if "maximum context length" in str(e):
                print(f"  [ABSTAIN] {eval_model} context overflow on note "
                      f"{record['note_id']} — skipping this case.")
                return None
            raise

    if parsed is None:
        print(f"  [ABSTAIN] {eval_model} schema-invalid on note "
              f"{record['note_id']} — skipping this case.")
        return None

    return {
        "note_id":         record["note_id"],
        "patient_profile": patient_profile,
        "n_accepted":      record["n_accepted"],
        "n_discarded":     record["n_discarded"],
        "early_termination": record["early_termination"],
        "session_complete":  record["session_complete"],
        "tcs_raw":         parsed.model_dump(),
    }


async def evaluate_model(
    model: str,
    note_lookup: dict[str, str],
    profile_lookup: dict[str, dict],
    template: jinja2.Template,
    client: AsyncOpenAI,
    eval_model: str,
    temperature: float,
    concurrency: int,
) -> None:
    print(f"\n[{model}] Loading records...")
    records = load_records(model)
    filtered = [r for r in records if r["note_id"] in note_lookup]
    dropped  = len(records) - len(filtered)
    if dropped:
        print(f"  [{model}] Dropped {dropped} records whose note_id is not in the curated test_file.")
    print(f"[{model}] Evaluating {len(filtered)} records...")

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        evaluate_one(
            semaphore, r,
            note_lookup[r["note_id"]],
            profile_lookup[r["note_id"]],
            template, client, eval_model, temperature,
        )
        for r in filtered
    ]
    results = [r for r in await asyncio.gather(*tasks) if r is not None]
    abstained = len(tasks) - len(results)

    output_path = OUTPUT_ROOT / model / f"topic_checklist_raw__{eval_model}.jsonl"
    pd.DataFrame(results).to_json(output_path, lines=True, orient="records")
    msg = f"[{model}] Saved {len(results)} evaluations to {output_path}"
    if abstained:
        msg += f"  ({abstained} abstained)"
    print(msg)


async def async_main(cfg: DictConfig) -> None:
    ev        = cfg.evaluation
    models    = list(ev.models)
    test_file = str(PROJECT_PATH / "data/processed" / cfg.discharge_bench.test_file)

    note_lookup    = load_note_lookup(test_file)
    profile_lookup = load_profile_lookup(test_file)

    loader   = jinja2.FileSystemLoader(str(PROMPTS_PATH))
    env      = jinja2.Environment(loader=loader)
    template = env.get_template(cfg.prompts.topic_checklist)
    eval_port = ev.get("eval_port", None)
    client = judge.make_judge_client(eval_port)
    temperature = ev.get("eval_temperature", 0)

    for model in models:
        await evaluate_model(
            model, note_lookup, profile_lookup,
            template, client, ev.eval_model, temperature, ev.concurrency,
        )

    print("\nAll models evaluated.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path=PROJECT_PATH.joinpath("config").as_posix(), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
