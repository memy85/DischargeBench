"""
conversation_quality.py

Evaluates conversation quality across all models listed in cfg.evaluation.models.
For each model, loads data/processed/output/{model}/simulation_results.jsonl and
saves raw ratings to
    data/processed/output/{model}/conversation_quality__{eval_model}.jsonl
(tagged by judge model; conversation_quality_calculate_score.py reads it).

The judge rates four Likert dimensions (naturalness, responsiveness, clarity,
clinical_relevance). `readability` is a deterministic textstat computation
(Flesch-Kincaid grade of the educator turns), not a judgement.

Usage:
    # judge served on eval_port
    python src/evaluation/conversation_quality.py "evaluation.eval_model=gemma-4-31B-it"

    # single model override:
    python src/evaluation/conversation_quality.py evaluation.models=[medgemma-4b-text-it]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import hydra
import jinja2
import pandas as pd
from omegaconf import DictConfig, ListConfig
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel
import numpy as np
import textstat

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_PATH, PROMPTS_PATH

sys.path.insert(0, str(Path(__file__).parent))
import judge

OUTPUT_ROOT   = PROJECT_PATH / "data/processed/output"


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class Rating(BaseModel):
    score: int
    reason: str

class ConversationQuality(BaseModel):
    naturalness:        Rating
    responsiveness:     Rating
    clarity:            Rating
    clinical_relevance: Rating


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


def format_conversation(conversation: list[dict]) -> list[str]:
    return [f"{turn['role'].capitalize()}: {turn['message']}" for turn in conversation]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_readability(
    record: dict,
        ) -> list : 

    educator_sentences = [turn["message"] for turn in record["conversation"] if turn["role"] == "educator"]

    read_grade = list(map(textstat.flesch_kincaid_grade, educator_sentences))
    return read_grade

async def evaluate_one(
    semaphore: asyncio.Semaphore,
    record: dict,
    template: jinja2.Template,
    client: AsyncOpenAI,
    eval_model: str,
    temperature: float,
) -> dict:

    # calculate readability
    readability_grades = evaluate_readability(record)

    conversation_history = format_conversation(record["conversation"])
    prompt = template.render(conversation_history=conversation_history)

    async with semaphore:
        try:
            parsed = await judge.structured_call(
                client, eval_model, prompt, ConversationQuality,
                temperature=temperature,
            )
        except openai.BadRequestError as e:
            # The judge's context window can't hold this case's conversation.
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
        "note_id":              record["note_id"],
        "n_accepted":           record["n_accepted"],
        "n_discarded":          record["n_discarded"],
        "early_termination":    record["early_termination"],
        "session_complete":     record["session_complete"],
        "conversation_quality": parsed.model_dump(),
        "readability": {"raw" : readability_grades, "score" : float(np.mean(readability_grades))}
    }


async def evaluate_model(
    model: str,
    template: jinja2.Template,
    client: AsyncOpenAI,
    eval_model: str,
    temperature: float,
    concurrency: int,
) -> None:
    print(f"\n[{model}] Loading records...")
    records = load_records(model)
    print(f"[{model}] Evaluating {len(records)} records...")

    semaphore = asyncio.Semaphore(concurrency)
    tasks     = [evaluate_one(semaphore, r, template, client, eval_model, temperature)
                 for r in records]
    results   = [r for r in await asyncio.gather(*tasks) if r is not None]
    abstained = len(tasks) - len(results)

    output_path = OUTPUT_ROOT / model / f"conversation_quality__{eval_model}.jsonl"
    pd.DataFrame(results).to_json(output_path, lines=True, orient="records")
    msg = f"[{model}] Saved {len(results)} evaluations to {output_path}"
    if abstained:
        msg += f"  ({abstained} abstained)"
    print(msg)


async def async_main(cfg: DictConfig) -> None:
    ev      = cfg.evaluation
    models  = list(ev.models)

    loader   = jinja2.FileSystemLoader(str(PROMPTS_PATH))
    env      = jinja2.Environment(loader=loader)
    template = env.get_template(cfg.prompts.conversation_quality)
    eval_port = ev.get("eval_port", None)
    client = judge.make_judge_client(eval_port)
    temperature = ev.get("eval_temperature", 0)

    for model in models:
        await evaluate_model(model, template, client, ev.eval_model, temperature, ev.concurrency)

    print("\nAll models evaluated.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path=PROJECT_PATH.joinpath("config").as_posix(), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
