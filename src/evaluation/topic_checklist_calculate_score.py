"""
topic_checklist_calculate_score.py

Loads the per-juror topic-checklist raw files for each model and computes a
weighted coverage score per case. Questions are skipped when the corresponding
topic is absent from the patient profile (None), so the denominator reflects only
applicable topics.

**LLM-as-a-Jury.** Each juror answered the SAME fixed 21-question
checklist about the SAME (note, conversation), so answers align by question key. For
each question we take a strict MAJORITY VOTE of the binary yes/no answer across
jurors (odd panel -> always resolves, no ties). The consensus answers then flow
through the exact same `calculate_tcs` as the single-judge design, so a one-juror
panel reproduces the old numbers bit-for-bit.

Output: data/processed/output/{model}/topic_checklist_score.jsonl
        data/processed/output/tcs_jury_agreement.csv   (>1 juror only)

Usage:
    python src/evaluation/topic_checklist_calculate_score.py

    # single model override:
    python src/evaluation/topic_checklist_calculate_score.py "evaluation.models=[medgemma-4b-text-it]"
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_PATH

sys.path.insert(0, str(Path(__file__).parent))
import jury

OUTPUT_ROOT = PROJECT_PATH / "data/processed/output"
DATA_ROOT   = PROJECT_PATH / "data/processed"
RAW_STEM    = "topic_checklist_raw"

ANSWERS = ["yes", "no"]

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
# Scoring (unchanged from the single-judge design)
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


# ---------------------------------------------------------------------------
# Jury consensus
# ---------------------------------------------------------------------------

def _index_by_note(records: list[dict]) -> dict[str, dict]:
    return {r["note_id"]: r for r in records}


def build_consensus(panel_raw: dict[str, list[dict]], panel: list[str],
                    primary_index: int, agr: dict) -> list[dict]:
    """Majority-vote each checklist question across jurors -> consensus records.

    Every juror answered the same fixed 21 questions, so answers align by question
    key. The consensus record keeps the primary juror's full row (note_id, profile,
    metadata, and the tcs_raw sub-dicts with their reasons) and overwrites each
    question's `answer` with the voted label, so a one-juror panel is a no-op.
    `agr` accumulates per-question juror-answer lists for the agreement report.
    """
    by_juror = {j: _index_by_note(panel_raw[j]) for j in panel}

    # case set = UNION over jurors, so one juror's transient failure on a case does
    # not silently drop it; it is still scored from the jurors that judged it.
    note_ids, seen = [], set()
    for j in panel:
        for r in panel_raw[j]:
            if r["note_id"] not in seen:
                seen.add(r["note_id"]); note_ids.append(r["note_id"])

    primary = panel[primary_index]
    consensus_records = []
    for nid in note_ids:
        jurors_here = [j for j in panel if nid in by_juror[j]]
        # metadata + reasons come from the primary juror when it judged this case,
        # else the first juror that did (voted answers overwrite the answers below).
        src = primary if nid in by_juror[primary] else jurors_here[0]
        base = dict(by_juror[src][nid])
        base_raw = dict(base["tcs_raw"])

        cons_raw = {}
        for q in QUESTIONS:
            labels = []
            for j in panel:
                rec = by_juror[j].get(nid)
                ans = rec["tcs_raw"].get(q, {}).get("answer") if rec else None
                labels.append(ans)
            voted, _ = jury.resolve(labels, primary_index=primary_index)
            agr.setdefault(q, []).append(labels)
            q_entry = dict(base_raw.get(q, {}))
            q_entry["answer"] = voted
            cons_raw[q] = q_entry

        base["tcs_raw"] = cons_raw
        consensus_records.append(base)
    return consensus_records


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
    panel = jury.panel_from_cfg(ev)
    primary = panel.index(str(ev.eval_model)) if str(ev.eval_model) in panel else 0
    print(f"Jury panel ({len(panel)}): {panel}  | primary/tie-break: {panel[primary]}")

    topic_fields_lookup = load_topic_fields_lookup(cfg.discharge_bench.test_file)

    agr_rows = []
    for model in models:
        print(f"\n[{model}] Calculating scores...")
        agr = {}
        panel_raw = jury.load_panel_raw(OUTPUT_ROOT, model, RAW_STEM, panel)
        records = build_consensus(panel_raw, panel, primary, agr)
        df = pd.DataFrame(records)

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

        if len(panel) > 1:
            # per-question agreement, then pooled over all questions
            all_units = [u for q in QUESTIONS for u in agr.get(q, [])]
            a = jury.agreement(all_units, categories=ANSWERS)
            row = {"model": model, "n_question_units": len(all_units),
                   "kappa": a.fleiss_kappa, "percent_agreement": a.percent_agreement,
                   "unanimous_rate": a.unanimous_rate}
            for q in QUESTIONS:
                aq = jury.agreement(agr.get(q, []), categories=ANSWERS)
                row[f"{q}_pa"] = aq.percent_agreement
            agr_rows.append(row)
            print(f"  agreement: kappa={a.fleiss_kappa}, pa={a.percent_agreement:.4f}, "
                  f"unanimous={a.unanimous_rate:.4f} over {len(all_units)} question-units")

    if agr_rows:
        out = OUTPUT_ROOT / "tcs_jury_agreement.csv"
        pd.DataFrame(agr_rows).to_csv(out, index=False)
        print(f"\nSaved inter-juror agreement -> {out}")

    print("\nAll models scored.")


if __name__ == "__main__":
    main()
