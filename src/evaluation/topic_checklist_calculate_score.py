"""
topic_checklist_calculate_score.py

Loads the judge's topic-checklist raw file for each model and computes a weighted
coverage score per case. Questions are skipped when the corresponding topic is absent
from the patient profile (None), so the denominator reflects only applicable topics.

Input:  data/processed/output/{model}/topic_checklist_raw__{eval_model}.jsonl
Output: data/processed/output/{model}/topic_checklist_score.jsonl

Usage:
    python src/evaluation/topic_checklist_calculate_score.py

    # single model override:
    python src/evaluation/topic_checklist_calculate_score.py "evaluation.models=[medgemma-4b-it]"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_PATH

OUTPUT_ROOT = PROJECT_PATH / "data/processed/output"
DATA_ROOT   = PROJECT_PATH / "data/processed"
RAW_STEM    = "topic_checklist_raw"

# ---------------------------------------------------------------------------
# Score book: question -> (patient_profile key, weight)
# Parent questions (Q2, Q3, ...) are scored only if sub-questions exist.
# Sub-questions carry half weight; parent carries full weight.
# ---------------------------------------------------------------------------

SCORE_BOOK = {
    "Q1":   {"qtype": "discharge_diagnosis",                        "score": 1.0},
    "Q2":   {"qtype": "new_medication",                             "score": 1.0},
    "Q2_1": {"qtype": "new_medication",                             "score": 0.5},
    "Q2_2": {"qtype": "new_medication",                             "score": 0.5},
    "Q2_3": {"qtype": "new_medication",                             "score": 0.5},
    "Q2_4": {"qtype": "new_medication",                             "score": 0.5},
    "Q3":   {"qtype": "treatment_during_stay",                      "score": 1.0},
    "Q3_1": {"qtype": "treatment_during_stay",                      "score": 0.5},
    "Q3_2": {"qtype": "treatment_during_stay",                      "score": 0.5},
    "Q3_3": {"qtype": "treatment_during_stay",                      "score": 0.5},
    "Q4":   {"qtype": "post_discharge_treatment",                   "score": 1.0},
    "Q4_1": {"qtype": "post_discharge_treatment",                   "score": 0.5},
    "Q4_2": {"qtype": "post_discharge_treatment",                   "score": 0.5},
    "Q4_3": {"qtype": "post_discharge_treatment",                   "score": 0.5},
    "Q5":   {"qtype": "return_to_hospital_sign_and_symptoms_and_how", "score": 1.0},
    "Q5_1": {"qtype": "return_to_hospital_sign_and_symptoms_and_how", "score": 0.5},
    "Q5_2": {"qtype": "return_to_hospital_sign_and_symptoms_and_how", "score": 0.5},
    "Q5_3": {"qtype": "return_to_hospital_sign_and_symptoms_and_how", "score": 0.5},
    "Q6":   {"qtype": "follow_up_appointment",                      "score": 1.0},
    "Q6_1": {"qtype": "follow_up_appointment",                      "score": 0.5},
    "Q6_2": {"qtype": "follow_up_appointment",                      "score": 0.5},
}

QUESTIONS = list(SCORE_BOOK.keys())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def calculate_tcs(topic_fields: dict, tcs_raw: dict) -> float:
    total_score = 0.0
    model_score = 0.0

    for q, book in SCORE_BOOK.items():
        qtype = book["qtype"]
        weight = book["score"]

        if not topic_fields.get(qtype):
            continue

        answer = tcs_raw.get(q, {}).get("answer", "no")
        total_score += weight
        if answer == "yes":
            model_score += weight

    return model_score / total_score if total_score > 0 else 0.0


def load_raw(model: str, eval_model: str) -> list[dict]:
    """The judge's raw records, one per case (a repeated note_id keeps its last record,
    in first-seen order). Each record's tcs_raw is normalised to the 21 checklist
    questions, a missing question carrying answer None."""
    path = OUTPUT_ROOT / model / f"{RAW_STEM}__{eval_model}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No topic-checklist raw file for '{model}': expected {path}")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    by_note = {}
    for r in rows:
        by_note[r["note_id"]] = r
    records = []
    for r in by_note.values():
        rec = dict(r)
        raw = r["tcs_raw"]
        rec["tcs_raw"] = {q: {**raw.get(q, {}), "answer": raw.get(q, {}).get("answer")}
                          for q in QUESTIONS}
        records.append(rec)
    return records


def load_topic_fields_lookup(test_file: str) -> dict[str, dict]:
    df = pd.read_json(DATA_ROOT / test_file, lines=True, orient="records")
    if "topic_checklist_fields" not in df.columns:
        raise KeyError(
            f"'topic_checklist_fields' column missing from {test_file}. "
            "Re-run the dataset-enrichment step that pulls applicability fields from "
            "dischargebench_dataset.json."
        )
    return dict(zip(df["note_id"], df["topic_checklist_fields"]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path=PROJECT_PATH.joinpath("config").as_posix(), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ev = cfg.evaluation
    models = list(ev.models)
    print(f"Judge: {ev.eval_model}")

    topic_fields_lookup = load_topic_fields_lookup(cfg.discharge_bench.test_file)

    for model in models:
        print(f"\n[{model}] Calculating scores...")
        df = pd.DataFrame(load_raw(model, str(ev.eval_model)))

        before = len(df)
        df = df[df["note_id"].isin(topic_fields_lookup)].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"  [{model}] Dropped {dropped} records whose note_id is not in the curated test_file.")

        scores = []
        for _, row in df.iterrows():
            tf = topic_fields_lookup[row["note_id"]]
            scores.append(calculate_tcs(tf, row["tcs_raw"]))
        df["tcs_score"] = scores

        output_path = OUTPUT_ROOT / model / "topic_checklist_score.jsonl"
        df.to_json(output_path, lines=True, orient="records")
        print(f"[{model}] Saved scores to {output_path}  (mean: {df['tcs_score'].mean():.3f})")

    print("\nAll models scored.")


if __name__ == "__main__":
    main()
