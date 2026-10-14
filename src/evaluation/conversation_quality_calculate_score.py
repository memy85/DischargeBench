"""
conversation_quality_calculate_score.py

Turns the judge's conversation-quality ratings (conversation_quality__<eval_model>.jsonl)
into per-case scores. Pure CPU. Each of the four Likert dimensions keeps the judge's
1-5 score (stored as a float); `readability` is carried through unchanged.

Output: data/processed/output/{model}/conversation_quality_score.jsonl

Usage:
    python src/evaluation/conversation_quality_calculate_score.py
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
RAW_STEM = "conversation_quality"

DIMENSIONS = ["naturalness", "responsiveness", "clarity", "clinical_relevance"]

# session metadata carried through from the raw file
CARRY = ["n_accepted", "n_discarded", "early_termination", "session_complete"]


def load_raw(model: str, eval_model: str) -> list[dict]:
    """The judge's ratings, one record per case (a repeated note_id keeps its last
    record, in first-seen order)."""
    path = OUTPUT_ROOT / model / f"{RAW_STEM}__{eval_model}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No conversation-quality raw file for '{model}': expected {path}")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    by_note = {}
    for r in rows:
        by_note[r["note_id"]] = r
    records = []
    for nid, base in by_note.items():
        cq = {}
        for dim in DIMENSIONS:
            sc = base["conversation_quality"][dim]["score"]
            cq[dim] = {**base["conversation_quality"].get(dim, {}),
                       "score": float(sc) if sc is not None else None}
        rec = {"note_id": nid}
        for k in CARRY:
            rec[k] = base.get(k)
        rec["conversation_quality"] = cq
        rec["readability"] = base.get("readability")
        records.append(rec)
    return records


def score_model(model: str, eval_model: str) -> pd.DataFrame:
    df = pd.DataFrame(load_raw(model, eval_model))
    df.to_json(OUTPUT_ROOT / model / "conversation_quality_score.jsonl",
               lines=True, orient="records")
    return df


def dim_mean(df: pd.DataFrame, dim: str) -> float:
    return df["conversation_quality"].map(lambda cq: cq[dim]["score"]).mean()


@hydra.main(config_path=PROJECT_PATH.joinpath("config").as_posix(),
            config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ev = cfg.evaluation
    print(f"Judge: {ev.eval_model}")

    for model in list(ev.models):
        print(f"\n[{model}] Scoring...")
        df = score_model(model, str(ev.eval_model))
        means = {d: dim_mean(df, d) for d in DIMENSIONS}
        read_mean = df["readability"].map(lambda r: r["score"]).mean()
        print("  " + " | ".join(f"{d[:4]} {means[d]:.3f}" for d in DIMENSIONS)
              + f" | readability {read_mean:.2f}")

    print("\nAll models scored.")


if __name__ == "__main__":
    main()
