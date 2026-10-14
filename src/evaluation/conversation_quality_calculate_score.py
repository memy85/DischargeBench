"""
conversation_quality_calculate_score.py

Aggregates the per-juror conversation-quality raw files into consensus ratings.
Pure CPU.

**LLM-as-a-Jury.** The four Likert dimensions are ordinal, so
they are combined by MEAN POOLING, not majority
vote. For each case and dimension we average the jurors' 1-5 scores. `readability`
is a deterministic textstat computation (identical across jurors), carried through
from the primary juror. A one-juror panel makes the pool a no-op and reproduces the
old ratings bit-for-bit (4 -> 4.0).

Output: data/processed/output/{model}/conversation_quality_score.jsonl
        data/processed/output/conversation_quality_jury_agreement.csv   (>1 juror only)

Usage:
    python src/evaluation/conversation_quality_calculate_score.py
"""

from __future__ import annotations

import statistics
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
RAW_STEM = "conversation_quality"

DIMENSIONS = ["naturalness", "responsiveness", "clarity", "clinical_relevance"]
LIKERT = ["1", "2", "3", "4", "5"]

# metadata carried verbatim from the primary juror (not judged, identical across jurors)
CARRY = ["n_accepted", "n_discarded", "early_termination", "session_complete"]


def _index_by_note(records: list[dict]) -> dict[str, dict]:
    return {r["note_id"]: r for r in records}


def build_consensus(panel_raw: dict[str, list[dict]], panel: list[str],
                    primary_index: int, agr: dict) -> list[dict]:
    """Mean-pool each Likert dimension across jurors -> consensus records.

    `agr` accumulates, per dimension, the per-case list of juror scores (as strings)
    for the agreement report and a numeric spread.
    """
    by_juror = {j: _index_by_note(panel_raw[j]) for j in panel}
    primary = panel[primary_index]

    # case set = UNION over jurors (a juror's transient failure on a case must not
    # silently drop it; it is still pooled from the jurors that rated it).
    note_ids, seen = [], set()
    for j in panel:
        for r in panel_raw[j]:
            if r["note_id"] not in seen:
                seen.add(r["note_id"]); note_ids.append(r["note_id"])

    records = []
    for nid in note_ids:
        base = by_juror[primary].get(nid) or by_juror[
            next(j for j in panel if nid in by_juror[j])][nid]

        cons_cq = {}
        for dim in DIMENSIONS:
            scores, str_labels = [], []
            for j in panel:
                rec = by_juror[j].get(nid)
                sc = rec["conversation_quality"][dim]["score"] if rec else None
                scores.append(sc)
                str_labels.append(str(sc) if sc is not None else None)
            pooled = jury.mean_pool(scores)
            base_dim = dict(base["conversation_quality"].get(dim, {}))
            base_dim["score"] = pooled
            cons_cq[dim] = base_dim
            agr[dim].append(str_labels)
            present = [s for s in scores if s is not None]
            if len(present) >= 2:
                agr["_spread"][dim].append(statistics.pstdev(present))

        rec_out = {"note_id": nid}
        for k in CARRY:
            rec_out[k] = base.get(k)
        rec_out["conversation_quality"] = cons_cq
        rec_out["readability"] = base.get("readability")  # deterministic, from primary
        records.append(rec_out)
    return records


def score_model(model: str, panel: list[str], primary_index: int, agr: dict) -> pd.DataFrame:
    panel_raw = jury.load_panel_raw(OUTPUT_ROOT, model, RAW_STEM, panel)
    records = build_consensus(panel_raw, panel, primary_index, agr)
    df = pd.DataFrame(records)
    df.to_json(OUTPUT_ROOT / model / "conversation_quality_score.jsonl",
               lines=True, orient="records")
    return df


def dim_mean(df: pd.DataFrame, dim: str) -> float:
    return df["conversation_quality"].map(lambda cq: cq[dim]["score"]).mean()


@hydra.main(config_path=PROJECT_PATH.joinpath("config").as_posix(),
            config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ev = cfg.evaluation
    panel = jury.panel_from_cfg(ev)
    primary = panel.index(str(ev.eval_model)) if str(ev.eval_model) in panel else 0
    print(f"Jury panel ({len(panel)}): {panel}  | primary/tie-break: {panel[primary]}")

    agr_rows = []
    for model in list(ev.models):
        print(f"\n[{model}] Scoring...")
        agr = {d: [] for d in DIMENSIONS}
        agr["_spread"] = {d: [] for d in DIMENSIONS}
        df = score_model(model, panel, primary, agr)
        means = {d: dim_mean(df, d) for d in DIMENSIONS}
        read_mean = df["readability"].map(lambda r: r["score"]).mean()
        print("  " + " | ".join(f"{d[:4]} {means[d]:.3f}" for d in DIMENSIONS)
              + f" | readability {read_mean:.2f}")

        if len(panel) > 1:
            row = {"model": model, "n": len(df)}
            for d in DIMENSIONS:
                a = jury.agreement(agr[d], categories=LIKERT)
                sp = agr["_spread"][d]
                row[f"{d}_exact_pa"] = a.percent_agreement
                row[f"{d}_kappa"] = a.fleiss_kappa
                row[f"{d}_mean_juror_std"] = (sum(sp) / len(sp)) if sp else None
            agr_rows.append(row)
            print("  agreement (exact-match PA): "
                  + " | ".join(f"{d[:4]} {row[f'{d}_exact_pa']:.3f}" for d in DIMENSIONS))

    if agr_rows:
        out = OUTPUT_ROOT / "conversation_quality_jury_agreement.csv"
        pd.DataFrame(agr_rows).to_csv(out, index=False)
        print(f"\nSaved inter-juror agreement -> {out}")

    print("\nAll models scored.")


if __name__ == "__main__":
    main()
