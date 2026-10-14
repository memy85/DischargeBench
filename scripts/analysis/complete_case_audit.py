"""
complete_case_audit.py — descriptive audit of completed simulation sessions (appendix table).

Per educator model, counted per SESSION (320 cases × 5 runs = 1,600 per model):
  complete       the Environment Agent closed the session normally (no error, no early termination,
                 session_complete = true)
  short          complete sessions with ≤ 15 accepted turns
  char_break     complete sessions in which a patient turn matches an AI self-disclosure pattern
                 ("as an AI", "language model", "AI assistant", ...) — the simulated patient broke character
  lowTCS_short   short sessions whose TCS is below 0.5 (at least half the checklist weight missed)
  retained       complete minus (char_break ∪ lowTCS_short)
Percentages are relative to the model's complete sessions. This audit is descriptive: all reported
metrics use every session.

Usage:  python scripts/analysis/complete_case_audit.py
Output: outputs/clean_complete_audit.csv
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from utils import CONFIG   # noqa: E402
OUT = PROJECT / "data/processed/output"
DATASET = PROJECT / "data/processed" / CONFIG["discharge_bench"]["test_file"]
_DISPLAY = {e["name"]: e["display"] for e in CONFIG["educators"]}
MODELS = [(_DISPLAY[m], m) for m in CONFIG["analysis"]["display_order"]]   # (display, output dir)
RUNS = CONFIG["runs"]
SHORT_MAX_TURNS = CONFIG["analysis"]["audit"]["short_max_turns"]
LOW_TCS = CONFIG["analysis"]["audit"]["low_tcs"]
AI_DISCLOSURE = re.compile(
    r"\b(as an? AI|language model|I am not a real|I'?m just a|I cannot pretend|I can'?t actually|"
    r"in reality I'?m|as a chatbot|i am an? (AI|virtual) assistant|AI assistant|trained by|my training|"
    r"i don'?t have (a body|real feelings|consciousness)|as an? (language |text )?model|as a simulation)\b", re.I)


def main() -> None:
    ids = {json.loads(l)["note_id"] for l in open(DATASET) if l.strip()}
    rows = []
    for disp, d in MODELS:
        for r in RUNS:
            run = OUT / f"{d}__{r}"
            tcs = {json.loads(l)["note_id"]: json.loads(l)["tcs_score"]
                   for l in open(run / "topic_checklist_score.jsonl") if l.strip()}
            for line in open(run / "simulation_results.jsonl"):
                x = json.loads(line)
                if x["note_id"] not in ids:
                    continue
                rows.append({
                    "model": disp,
                    "complete": not x.get("error") and not x.get("early_termination") and bool(x.get("session_complete")),
                    "short": x.get("n_accepted", 0) <= SHORT_MAX_TURNS,
                    "char_break": any(t.get("role") == "patient" and AI_DISCLOSURE.search(t.get("message") or "")
                                      for t in x.get("conversation", [])),
                    "tcs": tcs.get(x["note_id"])})
    t = pd.DataFrame(rows)
    c = t[t.complete].copy()
    c["short"] = c["short"].astype(bool)
    c["lowTCS_short"] = c["short"] & (c["tcs"] < LOW_TCS)
    g = c.groupby("model")
    s = pd.DataFrame({"sessions": t.groupby("model").size(), "complete": g.size(),
                      "short": g["short"].sum(), "char_break": g["char_break"].sum(),
                      "lowTCS_short": g["lowTCS_short"].sum()})
    s["retained"] = c.assign(drop=c["char_break"] | c["lowTCS_short"]).groupby("model")["drop"].apply(lambda v: (~v).sum())
    for k in ["short", "char_break", "lowTCS_short"]:
        s[f"{k}_%"] = (s[k] / s["complete"] * 100).round(1)
    s = s.loc[[m for m, _ in MODELS]]
    (PROJECT / "outputs").mkdir(exist_ok=True)
    s.to_csv(PROJECT / "outputs/clean_complete_audit.csv")
    print(s.to_string())


if __name__ == "__main__":
    main()
