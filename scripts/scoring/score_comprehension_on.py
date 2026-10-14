"""
score_comprehension_on.py — the reported Comprehension score (SCS).

Input  : data/processed/output/<model>__run<r>/comprehension_score_stripped.jsonl, produced by
         src/evaluation/comprehension_score.py with COMPREHENSION_STRIP_CLINICAL=1 (the persona-
         ablated patient answers the six questions using only the education conversation).
Rule   : for each record, drop every question whose stored reference answer is "not applicable"
         (the reference extractor's null form: the note contains nothing for that question; such
         items cannot be graded meaningfully). The record's score is the mean of the remaining
         post-conversation question scores (correct 1, partially correct 0.5, incorrect 0).
Output : data/processed/output/<model>__run<r>/comprehension_score_orig_na.jsonl
         one row per record: {"note_id", "post_education_score"}.

Usage  : python scripts/scoring/score_comprehension_on.py            # write outputs
         python scripts/scoring/score_comprehension_on.py --check    # compare with existing files, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from utils import CONFIG   # noqa: E402
OUT = PROJECT / "data/processed/output"
DATASET = PROJECT / "data/processed" / CONFIG["discharge_bench"]["test_file"]
MODELS = [e["name"] for e in CONFIG["educators"]]
RUNS = CONFIG["runs"]
IN_FILE, OUT_FILE = "comprehension_score_stripped.jsonl", "comprehension_score_orig_na.jsonl"


def is_na(ref: str) -> bool:
    return ref.strip().lower().startswith("not applicable")


def score_record(rec: dict) -> float:
    kept = [qr["post"]["score"] for qr in rec["question_results"] if not is_na(qr["reference_answer"])]
    if not kept:
        raise ValueError(f"{rec['note_id']}: no scorable question (all references not applicable)")
    return float(np.mean(kept))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with existing outputs; write nothing")
    args = ap.parse_args()
    ids = {json.loads(l)["note_id"] for l in open(DATASET) if l.strip()}
    n_mismatch = 0
    for m in MODELS:
        for r in RUNS:
            d = OUT / f"{m}__{r}"
            recs = [json.loads(l) for l in open(d / IN_FILE) if l.strip()]
            rows = [{"note_id": x["note_id"], "post_education_score": score_record(x)}
                    for x in recs if x["note_id"] in ids]
            if args.check:
                old = {json.loads(l)["note_id"]: json.loads(l)["post_education_score"]
                       for l in open(d / OUT_FILE) if l.strip()}
                bad = [x for x in rows if abs(old.get(x["note_id"], np.nan) - x["post_education_score"]) > 1e-9   # existing files store 10 decimals
                       or x["note_id"] not in old]
                n_mismatch += len(bad) + len(set(old) - {x["note_id"] for x in rows})
                print(f"{m}__{r}: {len(rows)} records, {len(bad)} mismatches")
            else:
                with open(d / OUT_FILE, "w") as f:
                    for x in rows:
                        f.write(json.dumps(x) + "\n")
                print(f"{m}__{r}: wrote {len(rows)} records")
    if args.check:
        print("CHECK", "PASSED" if n_mismatch == 0 else f"FAILED ({n_mismatch} mismatches)")


if __name__ == "__main__":
    main()
