"""
icd_chapter_analysis.py — why TCS varies by ICD chapter while Comprehension does not.

Pools the 5 variance runs x 8 educator models per case (Table 1 basis), on the
canonical v3 320-case set, and produces:

  1. Per-chapter table: TCS and strict Comprehension with across-case bootstrap 95% CIs,
     the Comprehension-TCS gap, the share of notes whose "Major Surgical or Invasive
     Procedure" header is empty, and TCS split by that header.
  2. Variance decomposition: chapter vs procedure-header as predictors of case TCS
     (nested-model F-test for chapter | procedure).
  3. TCS Q3 ("procedures during the hospital stay") yes-rate by procedure header.

Mechanism: TCS gates Q3-Q3.3 (weight 2.5) on `treatment_during_stay`, which is non-empty
for essentially every note (medications, monitoring), but Q3 asks about *procedures*. On
notes whose header reads "None", the judge answers Q3 "no" — often citing that header —
even when the educator correctly says no procedure was done. Chapters dominated by
non-procedural admissions (Pregnancy, Blood, Genitourinary, Infectious) lose TCS for a
reason unrelated to teaching; strict Comprehension is identical across the two groups.

Outputs:
  outputs/icd_chapter_table.csv
  figures/icd_chapter_tcs_comp_v3.{pdf,png}
  figures/icd_chapter_procedure_v3.{pdf,png}

Usage: python scripts/analysis/icd_chapter_analysis.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from utils import CONFIG   # noqa: E402
OUT = PROJECT / "data/processed/output"
FIG = PROJECT / "figures"
RUNS = CONFIG["runs"]
B = CONFIG["analysis"]["icd_bootstrap"]["B"]
SEED = CONFIG["analysis"]["icd_bootstrap"]["seed"]
JUDGE_RAW = f"topic_checklist_raw__{CONFIG['evaluation']['eval_model']}.jsonl"
COMP = "comprehension_score_orig_na.jsonl"   # reported SCS: strict grades, 'not applicable' items dropped
MODELS = CONFIG["analysis"]["display_order"]   # row order feeds the per-chapter bootstrap
C_TCS, C_COMP = "#3B5F91", "#A15340"          # unified manuscript palette

# Mirrors SCORE_BOOK in src/evaluation/topic_checklist_calculate_score.py.
SCORE_BOOK = {
    "Q1": ("discharge_diagnosis", 1.0),
    "Q2": ("new_medication", 1.0), "Q2_1": ("new_medication", 0.5), "Q2_2": ("new_medication", 0.5),
    "Q2_3": ("new_medication", 0.5), "Q2_4": ("new_medication", 0.5),
    "Q3": ("treatment_during_stay", 1.0), "Q3_1": ("treatment_during_stay", 0.5),
    "Q3_2": ("treatment_during_stay", 0.5), "Q3_3": ("treatment_during_stay", 0.5),
    "Q4": ("post_discharge_treatment", 1.0), "Q4_1": ("post_discharge_treatment", 0.5),
    "Q4_2": ("post_discharge_treatment", 0.5), "Q4_3": ("post_discharge_treatment", 0.5),
    "Q5": ("return_to_hospital_sign_and_symptoms_and_how", 1.0),
    "Q5_1": ("return_to_hospital_sign_and_symptoms_and_how", 0.5),
    "Q5_2": ("return_to_hospital_sign_and_symptoms_and_how", 0.5),
    "Q5_3": ("return_to_hospital_sign_and_symptoms_and_how", 0.5),
    "Q6": ("follow_up_appointment", 1.0), "Q6_1": ("follow_up_appointment", 0.5),
    "Q6_2": ("follow_up_appointment", 0.5),
}

ICD_SHORT = {
    "Pregnancy, Childbirth and the Puerperium": "Pregnancy",
    "Diseases of the Blood and Blood-Forming Organs": "Blood",
    "Factors Influencing Health Status": "Health-status factors",
    "Infectious and Parasitic Diseases": "Infectious",
    "Diseases of the Genitourinary System": "Genitourinary",
    "Symptoms, Signs and Ill-Defined Conditions": "Symptoms & signs",
    "Endocrine, Nutritional and Metabolic Diseases": "Endocrine",
    "Diseases of the Nervous System": "Nervous",
    "Diseases of the Respiratory System": "Respiratory",
    "Diseases of the Skin and Subcutaneous Tissue": "Skin",
    "Diseases of the Digestive System": "Digestive",
    "Diseases of the Circulatory System": "Circulatory",
    "Injury and Poisoning": "Injury & poisoning",
    "Neoplasms": "Neoplasms",
    "Diseases of the Musculoskeletal System and Connective Tissue": "Musculoskeletal",
    "Congenital Anomalies": "Congenital",
}

NO_PROC = {"none", "n/a", "none this hospitalization", "--"}


def procedure_status(text: str) -> str:
    """'none' / 'listed' / 'unknown' from the MIMIC 'Major Surgical or Invasive Procedure' header."""
    m = re.search(r"Major Surgical or Invasive Procedure:\s*(.*?)\n\s*\n", text, re.S | re.I)
    if m is None:
        return "unknown"
    val = m.group(1).strip().lower().strip(" .")
    if val in NO_PROC:
        return "none"
    if val == "___":                       # de-identified, content unknowable
        return "unknown"
    return "listed"


def calc_tcs(fields: dict, raw: dict) -> float:
    got = tot = 0.0
    for q, (field, w) in SCORE_BOOK.items():
        if not fields.get(field):
            continue
        tot += w
        if raw.get(q, {}).get("answer", "no") == "yes":
            got += w
    if tot == 0:
        raise ValueError("case with no applicable TCS fields")
    return got / tot


def load():
    ds = pd.read_json(PROJECT / "data/processed" / CONFIG["discharge_bench"]["test_file"], lines=True)
    ds["procedure"] = ds.text.map(procedure_status)
    fields = dict(zip(ds.note_id, ds.topic_checklist_fields))
    tcs, comp, q3 = {}, {}, []
    for m in MODELS:
        for r in RUNS:
            d_run = OUT / f"{m}__{r}"
            for line in open(d_run / JUDGE_RAW):
                d = json.loads(line)
                nid = d["note_id"]
                if nid not in fields:
                    continue
                tcs.setdefault(nid, []).append(calc_tcs(fields[nid], d["tcs_raw"]))
                q3.append((nid, d["tcs_raw"].get("Q3", {}).get("answer") == "yes"))
            for line in open(d_run / COMP):
                d = json.loads(line)
                if d["note_id"] in fields and d.get("post_education_score") is not None:
                    comp.setdefault(d["note_id"], []).append(d["post_education_score"])
    ds["tcs"] = ds.note_id.map(lambda n: np.mean(tcs[n]))
    ds["comp"] = ds.note_id.map(lambda n: np.mean(comp[n]))
    ds["chapter"] = ds.icd_chapter.map(ICD_SHORT)
    if ds.chapter.isna().any():
        raise KeyError(f"unmapped chapters: {ds.loc[ds.chapter.isna(), 'icd_chapter'].unique()}")
    q3 = pd.DataFrame(q3, columns=["note_id", "q3_yes"]).merge(ds[["note_id", "procedure"]])
    return ds, q3


def boot_ci(v, rng):
    v = np.asarray(v)
    return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)], [2.5, 97.5])


def main():
    rng = np.random.default_rng(SEED)
    ds, q3 = load()

    rows = []
    for ch, g in ds.groupby("chapter"):
        tl, th = boot_ci(g.tcs, rng)
        cl, chi = boot_ci(g.comp, rng)
        rows.append(dict(
            chapter=ch, n=len(g),
            tcs=g.tcs.mean(), tcs_lo=tl, tcs_hi=th,
            comp=g.comp.mean(), comp_lo=cl, comp_hi=chi,
            gap=g.comp.mean() - g.tcs.mean(),
            pct_no_procedure=(g.procedure == "none").mean(),
            tcs_no_procedure=g.loc[g.procedure == "none", "tcs"].mean(),
            tcs_procedure=g.loc[g.procedure == "listed", "tcs"].mean(),
        ))
    tab = pd.DataFrame(rows).sort_values("tcs", ascending=False).reset_index(drop=True)
    tab.to_csv(PROJECT / "outputs/icd_chapter_table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.round(3).to_string(index=False))

    # --- does chapter explain TCS beyond the procedure header? ---
    k = ds[ds.procedure != "unknown"]
    print(f"\nprocedure header: {ds.procedure.value_counts().to_dict()}")
    for f in ["tcs ~ C(chapter)", "tcs ~ C(procedure)", "tcs ~ C(procedure) + C(chapter)"]:
        r = smf.ols(f, k).fit()
        print(f"  {f:34s} R2={r.rsquared:.3f}  adjR2={r.rsquared_adj:.3f}")
    for y in ["tcs", "comp"]:
        a = smf.ols(f"{y} ~ C(procedure)", k).fit()
        b = smf.ols(f"{y} ~ C(procedure) + C(chapter)", k).fit()
        c0 = smf.ols(f"{y} ~ C(chapter)", k).fit()
        F, p, _ = b.compare_f_test(a)
        print(f"  {y}: chapter alone p={c0.f_pvalue:.2g}; chapter | procedure F={F:.2f} p={p:.3f}; "
              f"procedure effect (none - listed) = {a.params['C(procedure)[T.none]']:+.3f} (p={a.pvalues['C(procedure)[T.none]']:.1g})")
    print(f"  chapter-level corr(TCS, %no-procedure) = {np.corrcoef(tab.tcs, tab.pct_no_procedure)[0, 1]:.2f}")
    print(f"  chapter-level corr(Comp, %no-procedure) = {np.corrcoef(tab.comp, tab.pct_no_procedure)[0, 1]:.2f}")
    print("\nTCS Q3 yes-rate (all models x runs) by procedure header:")
    print(q3.groupby("procedure").q3_yes.agg(["mean", "count"]).round(3).to_string())

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    FIG.mkdir(exist_ok=True)

    # --- Figure 1: forest plot, TCS vs strict Comprehension by chapter ---
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(tab))[::-1]
    for col, c, lab, off in [("tcs", C_TCS, "TCS", 0.17), ("comp", C_COMP, "SCS", -0.17)]:
        ax.errorbar(tab[col], y + off, xerr=[tab[col] - tab[f"{col}_lo"], tab[f"{col}_hi"] - tab[col]],
                    fmt="o", color=c, ms=5.5, capsize=2.5, lw=1.2, label=lab, zorder=3)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  ({p:.0%})" for c, p in zip(tab.chapter, tab.pct_no_procedure)])
    ax.set_xlabel("Score across models")
    ax.text(1.0, -0.20, "(%) = share of notes with no procedure listed", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="0.35")
    ax.grid(True, color="0.85", lw=0.7)     # gridlines at both x and y ticks
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"icd_chapter_tcs_comp_v3.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: chapter TCS / Comprehension vs share of non-procedural notes ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, col, c, lab in [(axes[0], "tcs", C_TCS, "TCS"), (axes[1], "comp", C_COMP, "SCS")]:
        x = tab.pct_no_procedure.values * 100
        ax.scatter(x, tab[col], color=c, s=38, zorder=3)
        slope, icpt = np.polyfit(x, tab[col], 1)
        xs = np.linspace(0, 80, 2)
        ax.plot(xs, icpt + slope * xs, color=c, lw=1.2, alpha=0.7)
        r = np.corrcoef(x, tab[col])[0, 1]
        for _, row in tab.iterrows():
            if row.chapter in {"Pregnancy", "Blood", "Genitourinary", "Congenital", "Musculoskeletal", "Neoplasms"}:
                ax.annotate(row.chapter, (row.pct_no_procedure * 100, row[col]), fontsize=9,
                            xytext=(4, 3), textcoords="offset points", color="0.25")
        ax.set_title(f"{lab}   (r = {r:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("% of chapter's notes with no procedure listed")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Chapter mean score")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"icd_chapter_procedure_v3.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote outputs/icd_chapter_table.csv and figures/icd_chapter_{{tcs_comp,procedure}}_v3.{{pdf,png}}")


if __name__ == "__main__":
    main()
