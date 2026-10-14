"""
comprehension_score.py

Evaluates patient comprehension after the simulated conversation.
Six questions are asked one at a time to both the discharge note (reference)
and the patient agent (conditioned on system prompt + conversation history).
Each answer is judged as correct (1.0), partially correct (0.5), or incorrect (0.0).

Reference answers are pre-computed once across all models since they only depend
on the discharge note.

Output: data/processed/output/{model}/comprehension_score.jsonl

Usage:
    python src/evaluation/comprehension_score.py

    # single model override:
    python src/evaluation/comprehension_score.py "evaluation.models=[medgemma-4b-text-it]"
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from enum import Enum
from pathlib import Path

import hydra
import jinja2
import pandas as pd
from omegaconf import DictConfig
from openai import AsyncOpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_PATH

sys.path.insert(0, str(Path(__file__).parent))
import jury
from virtual_patient import (
    EDUCATION_LEVEL,
    HEALTH_LITERACY,
    PATIENT_PERSONAS,
    PAST_MEDICAL_HISTORY_RECALL,
)

PROMPTS_PATH = PROJECT_PATH / "prompts"
OUTPUT_ROOT  = PROJECT_PATH / "data/processed/output"
logger = logging.getLogger("comprehension_score")

# STRICT/ablation mode: COMPREHENSION_STRIP_CLINICAL=1 loads a persona that omits all
# note-derived clinical content (diagnosis, treatments, medications, history), so the
# patient can only answer comprehension questions from the education conversation.
# Used to quantify how much comprehension is persona-carried vs educator-driven.
import os as _os
_STRIP_CLINICAL = bool(_os.environ.get("COMPREHENSION_STRIP_CLINICAL"))
_PATIENT_PROMPT_FILE = "patient_prompt_conversation_only.jinja" if _STRIP_CLINICAL else "patient_prompt.jinja"
PATIENT_PROMPT = (PROMPTS_PATH / _PATIENT_PROMPT_FILE).read_text()
if _STRIP_CLINICAL:
    logger.warning("[comprehension] STRIP_CLINICAL mode: patient persona omits note-derived "
                   "clinical fields (%s)", _PATIENT_PROMPT_FILE)

COMPREHENSION_QUESTIONS = [
    "What was your discharge diagnosis?",
    "What treatments did you receive during your hospital stay?",
    "What post-discharge treatments were recommended to you?",
    "Which doctors or clinics do you need to follow up with after discharge?",
    "What symptoms or changes should prompt you to return to the ED or hospital?",
    "What medications should you take? Please include their name, dosage, and route.",
]

SCORE_MAP = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ReferenceAnswer(BaseModel):
    answer: str

class CorrectnessLevel(str, Enum):
    CORRECT           = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT         = "incorrect"

class JudgeResult(BaseModel):
    correctness: CorrectnessLevel
    reason:      str


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(model: str) -> list[dict]:
    path = OUTPUT_ROOT / model / "simulation_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No simulation results found for '{model}' at {path}")
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


def load_dataset(test_file: str) -> tuple[dict[str, str], dict[str, dict]]:
    df = pd.read_json(test_file, lines=True, orient="records")
    note_lookup    = dict(zip(df["note_id"], df["text"]))
    profile_lookup = dict(zip(df["note_id"], df["profile"]))
    return note_lookup, profile_lookup


# ---------------------------------------------------------------------------
# Patient system prompt + message builder
# ---------------------------------------------------------------------------

def render_patient_system_prompt(patient_profile: dict) -> str:
    template = jinja2.Template(PATIENT_PROMPT)
    return template.render(
        patient_age=patient_profile["age"],
        patient_gender=patient_profile["gender"],
        patient_race=patient_profile["race"],
        personality_description=PATIENT_PERSONAS[patient_profile["personality"]],
        education_level=patient_profile["education_level"],
        education_description=EDUCATION_LEVEL[patient_profile["education_level"]],
        health_literacy=patient_profile["health_literacy"],
        health_literacy_description=HEALTH_LITERACY[patient_profile["health_literacy"]],
        chief_complaint=patient_profile["chief_complaint"],
        main_diagnosis=patient_profile["main_diagnosis"],
        reason_for_admission=patient_profile["reason_for_admission"],
        medication_on_admission=patient_profile["medication_on_admission"],
        allergy=patient_profile["allergy"],
        medical_history=patient_profile["medical_history"],
        family_history=patient_profile["family_medical_history"],
        past_medical_history_recall_level=patient_profile["past_medical_history_recall"],
        past_medical_history_recall_description=PAST_MEDICAL_HISTORY_RECALL[patient_profile["past_medical_history_recall"]],
    )


def build_patient_messages(
    patient_system_prompt: str,
    conversation: list[dict],
    question_prompt: str,
) -> list[dict]:
    """
    Constructs messages from the patient's perspective:
      system    : patient persona
      user      : educator turns
      assistant : patient turns
      user      : comprehension question (final)
    """
    messages = [{"role": "system", "content": patient_system_prompt}]
    for turn in conversation:
        role = "user" if turn["role"] == "educator" else "assistant"
        messages.append({"role": role, "content": turn["message"]})
    messages.append({"role": "user", "content": question_prompt})
    return messages


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

async def extract_reference_answer(
    note_text: str,
    question: str,
    template: jinja2.Template,
    client: AsyncOpenAI,
    eval_model: str,
    api_style: str,
    semaphore: asyncio.Semaphore,
    *,
    temperature: float = 0,
) -> str:
    prompt = template.render(discharge_note=note_text, question=question)
    async with semaphore:
        answer = await jury.structured_call(
            client, eval_model, api_style, prompt, ReferenceAnswer,
            temperature=temperature, max_output_tokens=800,
        )
    if answer is None:
        logger.warning("[comprehension] reference extraction failed (schema-invalid); empty reference")
        return ""
    return answer.answer


async def ask_patient(
    messages: list[dict],
    vp_client: AsyncOpenAI,
    vp_model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        for attempt in range(3):
            try:
                output = await vp_client.chat.completions.create(
                    model=vp_model,
                    messages=messages,
                    max_tokens=512,
                )
                return (output.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    logger.warning("[comprehension] patient answer failed (%s); empty", type(e).__name__)
                    return ""


async def judge_answer(
    question: str,
    reference_answer: str,
    patient_answer: str,
    client: AsyncOpenAI,
    eval_model: str,
    api_style: str,
    semaphore: asyncio.Semaphore,
    *,
    temperature: float = 0,
) -> JudgeResult:
    prompt = (
        f"You are evaluating a patient's comprehension of their discharge information.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {reference_answer}\n"
        f"Patient's answer: {patient_answer}\n\n"
        f"Rate the patient's answer:\n"
        f"- correct: fully captures the key information in the reference answer\n"
        f"- partially_correct: captures some but not all key information\n"
        f"- incorrect: missing, wrong, or the patient said they don't know\n\n"
        f"Provide your rating and a brief reason."
    )
    async with semaphore:
        result = await jury.structured_call(
            client, eval_model, api_style, prompt, JudgeResult,
            temperature=temperature, max_output_tokens=400,
        )
    if result is None:
        logger.warning("[comprehension] judge failed (schema-invalid); marking incorrect")
        return JudgeResult(correctness=CorrectnessLevel.INCORRECT, reason="eval_error")
    return result


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

async def evaluate_question(
    question: str,
    note_text: str,
    conversation: list[dict],
    patient_system_prompt: str,
    ref_template: jinja2.Template,
    q_template: jinja2.Template,
    client: AsyncOpenAI,
    eval_model: str,
    api_style: str,
    vp_client: AsyncOpenAI,
    vp_model: str,
    semaphore: asyncio.Semaphore,
    *,
    temperature: float = 0,
) -> dict:
    """Evaluates a single question pre- and post-education."""
    question_prompt = q_template.render(question=question)

    # Reference answer from discharge note
    reference_answer = await extract_reference_answer(
        note_text, question, ref_template, client, eval_model, api_style, semaphore,
        temperature=temperature,
    )

    # Pre-education: patient conditioned on system prompt only (no conversation)
    pre_messages   = build_patient_messages(patient_system_prompt, [], question_prompt)
    pre_answer     = await ask_patient(pre_messages, vp_client, vp_model, semaphore)
    pre_judgment   = await judge_answer(question, reference_answer, pre_answer, client, eval_model,
                                        api_style, semaphore, temperature=temperature)

    # Post-education: patient conditioned on system prompt + full conversation
    post_messages  = build_patient_messages(patient_system_prompt, conversation, question_prompt)
    post_answer    = await ask_patient(post_messages, vp_client, vp_model, semaphore)
    post_judgment  = await judge_answer(question, reference_answer, post_answer, client, eval_model,
                                        api_style, semaphore, temperature=temperature)

    return {
        "question":         question,
        "reference_answer": reference_answer,
        "pre": {
            "patient_answer": pre_answer,
            "correctness":    pre_judgment.correctness,
            "reason":         pre_judgment.reason,
            "score":          SCORE_MAP[pre_judgment.correctness],
        },
        "post": {
            "patient_answer": post_answer,
            "correctness":    post_judgment.correctness,
            "reason":         post_judgment.reason,
            "score":          SCORE_MAP[post_judgment.correctness],
        },
    }


async def evaluate_case(
    record: dict,
    patient_profile: dict,
    note_text: str,
    ref_template: jinja2.Template,
    q_template: jinja2.Template,
    client: AsyncOpenAI,
    eval_model: str,
    api_style: str,
    vp_client: AsyncOpenAI,
    vp_model: str,
    semaphore: asyncio.Semaphore,
    *,
    temperature: float = 0,
) -> dict:
    patient_system_prompt = render_patient_system_prompt(patient_profile)
    conversation          = record["conversation"]

    question_results = []
    for question in COMPREHENSION_QUESTIONS:
        result = await evaluate_question(
            question, note_text, conversation, patient_system_prompt,
            ref_template, q_template, client, eval_model, api_style, vp_client, vp_model, semaphore,
            temperature=temperature,
        )
        question_results.append(result)

    pre_score  = sum(r["pre"]["score"]  for r in question_results) / len(question_results)
    post_score = sum(r["post"]["score"] for r in question_results) / len(question_results)

    return {
        "note_id":             record["note_id"],
        "n_accepted":          record["n_accepted"],
        "early_termination":   record["early_termination"],
        "session_complete":    record["session_complete"],
        "question_results":    question_results,
        "pre_education_score": pre_score,
        "post_education_score": post_score,
        "score_shift":         post_score - pre_score,
    }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

async def evaluate_model(
    model: str,
    note_lookup: dict[str, str],
    profile_lookup: dict[str, dict],
    ref_template: jinja2.Template,
    q_template: jinja2.Template,
    client: AsyncOpenAI,
    eval_model: str,
    api_style: str,
    vp_client: AsyncOpenAI,
    vp_model: str,
    concurrency: int,
    *,
    temperature: float = 0,
) -> None:
    print(f"\n[{model}] Loading records...")
    records = load_records(model)
    print(f"[{model}] Evaluating {len(records)} records...")

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        evaluate_case(
            r,
            profile_lookup[r["note_id"]],
            note_lookup[r["note_id"]],
            ref_template, q_template,
            client, eval_model, api_style, vp_client, vp_model, semaphore,
            temperature=temperature,
        )
        for r in records
        if r["note_id"] in profile_lookup
    ]
    results = await asyncio.gather(*tasks)

    df          = pd.DataFrame(results)
    # STRIP_CLINICAL mode writes to a separate file so the real results are untouched.
    _score_name = "comprehension_score_stripped.jsonl" if _STRIP_CLINICAL else "comprehension_score.jsonl"
    output_path = OUTPUT_ROOT / model / _score_name
    df.to_json(output_path, lines=True, orient="records")
    print(f"[{model}] Saved {len(results)} evaluations to {output_path}  "
          f"(pre: {df['pre_education_score'].mean():.3f}  "
          f"post: {df['post_education_score'].mean():.3f}  "
          f"shift: {df['score_shift'].mean():+.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(cfg: DictConfig) -> None:
    ev        = cfg.evaluation
    models    = list(ev.models)
    test_file = str(PROJECT_PATH / "data/processed" / cfg.discharge_bench.test_file)

    note_lookup, profile_lookup = load_dataset(test_file)
    eval_port = ev.get("eval_port", None)
    client, api_style = jury.make_judge_client(ev.eval_model, eval_port)
    temperature = ev.get("eval_temperature", 0)

    vp_port = ev.get("vp_port", None)
    if vp_port is not None:
        vp_client = AsyncOpenAI(base_url=f"http://localhost:{vp_port}/v1", api_key="EMPTY")
        vp_model  = cfg.discharge_bench.patient.backend
    else:
        vp_client = client
        vp_model  = ev.eval_model

    loader      = jinja2.FileSystemLoader(str(PROMPTS_PATH))
    env         = jinja2.Environment(loader=loader)
    ref_template = env.get_template("comprehension_reference.jinja")
    q_template   = env.get_template("comprehension_question.jinja")

    for model in models:
        await evaluate_model(
            model, note_lookup, profile_lookup,
            ref_template, q_template,
            client, ev.eval_model, api_style, vp_client, vp_model, ev.concurrency,
            temperature=temperature,
        )

    print("\nAll models evaluated.")


@hydra.main(config_path=PROJECT_PATH.joinpath("config").as_posix(), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
