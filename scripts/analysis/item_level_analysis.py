"""
item_level_analysis.py — exploratory item-level breakdown of TCS and Comprehension:
where models earn and lose points, stratified by personality and ICD chapter.
Descriptive statistics only. Existing outputs only (CPU).

Basis: 8 educator models x 5 variance runs x 320 v3 cases, pooled over records.
  TCS            per-item judge answers (topic_checklist_raw__gemma-4-31B-it.jsonl), gated exactly
                 like topic_checklist_calculate_score.py (item applies iff its qtype field in
                 topic_checklist_fields is non-empty). Topic roll-ups use the scorer's weights.
  Comprehension  reported SCS: persona-ablated (strict) grades (comprehension_score_stripped.jsonl),
                 items whose stored reference is "not applicable" dropped.

Output: outputs/item_analysis/*.csv, figures/item_*.{pdf,png}; summary statistics printed to stdout
Usage:  python scripts/analysis/item_level_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "src/evaluation"))
sys.path.insert(0, str(PROJECT / "scripts/analysis"))
from topic_checklist_calculate_score import SCORE_BOOK          # noqa: E402
from icd_chapter_analysis import ICD_SHORT, procedure_status   # noqa: E402
from utils import CONFIG                                        # noqa: E402

DATA = PROJECT / "data/processed"
OUT = DATA / "output"
TAB = PROJECT / "outputs/item_analysis"
FIG = PROJECT / "figures"
RUNS = CONFIG["runs"]
_DISPLAY = {e["name"]: e["display"] for e in CONFIG["educators"]}
MODELS = {m: _DISPLAY[m] for m in CONFIG["analysis"]["display_order"]}   # output dir -> display name
PERS = {"distrustful": "Distrustful", "anxious": "Anxious", "minimiser": "Minimizer",
        "high_conscientiousness": "Conscientious", "neutral": "Neutral"}
TOPIC = {"discharge_diagnosis": "Diagnosis", "new_medication": "New medication",
         "treatment_during_stay": "Treatment during stay", "post_discharge_treatment": "Post-discharge",
         "return_to_hospital_sign_and_symptoms_and_how": "Return-to-hospital",
         "follow_up_appointment": "Follow-up"}
TCS_LABEL = {
    "Q1": "Q1 Diagnosis", "Q2": "Q2 New medications", "Q2_1": "Q2.1 Dose", "Q2_2": "Q2.2 Route",
    "Q2_3": "Q2.3 Frequency", "Q2_4": "Q2.4 Side effects/precautions",
    "Q3": "Q3 Procedures during stay", "Q3_1": "Q3.1 Procedure names", "Q3_2": "Q3.2 Indications",
    "Q3_3": "Q3.3 Outcomes", "Q4": "Q4 Post-discharge treatment", "Q4_1": "Q4.1 Timing",
    "Q4_2": "Q4.2 Preparation", "Q4_3": "Q4.3 Related follow-up care",
    "Q5": "Q5 Warning signs + actions", "Q5_1": "Q5.1 Warning signs", "Q5_2": "Q5.2 Actions",
    "Q5_3": "Q5.3 When to seek care", "Q6": "Q6 Follow-up appointments", "Q6_1": "Q6.1 Timing",
    "Q6_2": "Q6.2 What to expect"}
COMP_Q = {
    "What was your discharge diagnosis?": "Q1 Diagnosis",
    "What treatments did you receive during your hospital stay?": "Q2 Treatments during stay",
    "What post-discharge treatments were recommended to you?": "Q3 Post-discharge treatments",
    "Which doctors or clinics do you need to follow up with after discharge?": "Q4 Follow-up",
    "What symptoms or changes should prompt you to return to the ED or hospital?": "Q5 Return signs",
    "What medications should you take? Please include their name, dosage, and route.": "Q6 Medications"}
PAIR = {"Diagnosis": "Q1 Diagnosis", "Treatment during stay": "Q2 Treatments during stay",
        "Post-discharge": "Q3 Post-discharge treatments", "Follow-up": "Q4 Follow-up",
        "Return-to-hospital": "Q5 Return signs", "New medication": "Q6 Medications"}
SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def read_jsonl(p: Path):
    return [json.loads(l) for l in open(p) if l.strip()]


def load():
    ds = pd.read_json(DATA / CONFIG["discharge_bench"]["test_file"], lines=True)
    ds["personality"] = ds.profile.map(lambda p: PERS[p["personality"]])
    ds["chapter"] = ds.icd_chapter.map(ICD_SHORT)
    ds["procedure"] = ds.text.map(procedure_status)
    meta = ds.set_index("note_id")[["personality", "chapter", "procedure"]]
    tcf = dict(zip(ds.note_id, ds.topic_checklist_fields))
    tcs, comp = [], []
    for mdir, disp in MODELS.items():
        for r in RUNS:
            d = OUT / f"{mdir}__{r}"
            for rec in read_jsonl(d / f"topic_checklist_raw__{CONFIG['evaluation']['eval_model']}.jsonl"):
                nid = rec["note_id"]
                if nid not in tcf:
                    continue
                for q, book in SCORE_BOOK.items():
                    if not tcf[nid].get(book["qtype"]):
                        continue                         # scorer's gate
                    tcs.append((disp, r, nid, q, TOPIC[book["qtype"]], book["score"],
                                rec["tcs_raw"][q]["answer"] == "yes"))
            for rec in read_jsonl(d / "comprehension_score_stripped.jsonl"):
                nid = rec["note_id"]
                if nid not in tcf:
                    continue
                for qr in rec["question_results"]:
                    if qr["reference_answer"].strip().lower().startswith("not applicable"):
                        continue                         # 'not applicable' reference: dropped (SCS rule)
                    comp.append((disp, r, nid, COMP_Q[qr["question"]], qr["post"]["correctness"],
                                 SCORE[qr["post"]["correctness"]]))
    tcs = pd.DataFrame(tcs, columns=["model", "run", "note_id", "item", "topic", "w", "yes"]).join(meta, on="note_id")
    comp = pd.DataFrame(comp, columns=["model", "run", "note_id", "q", "grade", "score"]).join(meta, on="note_id")
    tcs["item"] = tcs["item"].map(TCS_LABEL)
    return tcs, comp


def heatmap(tbl: pd.DataFrame, title: str, path: str, fmt=".2f", width=None, cbar_label=""):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Fira Sans", "DejaVu Sans"]})
    w = width or max(5.5, 0.62 * tbl.shape[1] + 3.2)
    fig, ax = plt.subplots(figsize=(w, 0.36 * tbl.shape[0] + 1.9))
    M = tbl.to_numpy(float)
    im = ax.imshow(M, cmap="cividis", aspect="auto")
    ax.set_xticks(range(tbl.shape[1])); ax.set_xticklabels(tbl.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(tbl.shape[0])); ax.set_yticklabels(tbl.index, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    thr = np.nanmin(M) + 0.55 * (np.nanmax(M) - np.nanmin(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < thr else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(cbar_label, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def topic_cov(df: pd.DataFrame, by: list[str]) -> pd.Series:
    """Weighted topic coverage like the scorer: sum(w*yes)/sum(w)."""
    g = df.assign(wy=df.w * df.yes).groupby(by)
    return g.wy.sum() / g.w.sum()


def main() -> None:
    for f in fm.findSystemFonts(str(PROJECT / "fonts")):
        fm.fontManager.addfont(f)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    tcs, comp = load()
    items = list(TCS_LABEL.values())
    qs = list(COMP_Q.values())
    pers = list(PERS.values())
    # personality order = SCS hardest -> easiest, as in Figure 7
    pers = list(comp.groupby(["note_id", "personality"]).score.mean().groupby("personality").mean()
                .sort_values().index)
    chap = list(comp.groupby("chapter").score.mean().sort_values().index)   # as strat_icd_chapter_v3

    # ---------------- TCS item tables ----------------
    yes = lambda by: tcs.groupby(by).yes.agg(yes_rate="mean", n="size")
    t_all = yes(["item"]).reindex(items)
    t_model = yes(["item", "model"]).unstack("model")
    t_pers = yes(["item", "personality"]).unstack("personality")
    t_chap = yes(["item", "chapter"]).unstack("chapter")
    for name, t in [("tcs_item_overall", t_all), ("tcs_item_by_model", t_model),
                    ("tcs_item_by_personality", t_pers), ("tcs_item_by_chapter", t_chap)]:
        t.to_csv(TAB / f"{name}.csv")
    topics = list(TOPIC.values())
    tp_all = topic_cov(tcs, ["topic"]).reindex(topics)
    tp_pers = topic_cov(tcs, ["topic", "personality"]).unstack()[pers].reindex(topics)
    tp_chap = topic_cov(tcs, ["topic", "chapter"]).unstack()[chap].reindex(topics)
    tp_model = topic_cov(tcs, ["topic", "model"]).unstack()[list(MODELS.values())].reindex(topics)
    pd.DataFrame({"coverage": tp_all}).to_csv(TAB / "tcs_topic_overall.csv")
    tp_pers.to_csv(TAB / "tcs_topic_by_personality.csv")
    tp_chap.to_csv(TAB / "tcs_topic_by_chapter.csv")
    tp_model.to_csv(TAB / "tcs_topic_by_model.csv")
    for t in topics:   # cross-tabs
        sub = tcs[tcs.topic == t]
        topic_cov(sub, ["model", "personality"]).unstack()[pers].to_csv(TAB / f"xtab_tcs_{t.replace(' ', '_').replace('-', '_').lower()}_model_x_personality.csv")
        topic_cov(sub, ["model", "chapter"]).unstack()[chap].to_csv(TAB / f"xtab_tcs_{t.replace(' ', '_').replace('-', '_').lower()}_model_x_chapter.csv")

    # ---------------- Comprehension question tables ----------------
    def cstats(by):
        g = comp.groupby(by)
        out = g.score.agg(mean="mean", n="size")
        for k in SCORE:
            out[k] = g.grade.apply(lambda s, k=k: (s == k).mean())
        return out
    c_all = cstats(["q"]).reindex(qs)
    c_model = cstats(["q", "model"]); c_pers = cstats(["q", "personality"]); c_chap = cstats(["q", "chapter"])
    c_all.to_csv(TAB / "comp_q_overall.csv"); c_model.to_csv(TAB / "comp_q_by_model.csv")
    c_pers.to_csv(TAB / "comp_q_by_personality.csv"); c_chap.to_csv(TAB / "comp_q_by_chapter.csv")
    for q in qs:
        sub = comp[comp.q == q]
        tag = q.split()[0].lower()
        sub.groupby(["model", "personality"]).score.mean().unstack()[pers].to_csv(TAB / f"xtab_comp_{tag}_model_x_personality.csv")
        sub.groupby(["model", "chapter"]).score.mean().unstack()[chap].to_csv(TAB / f"xtab_comp_{tag}_model_x_chapter.csv")

    # ---------------- heatmaps ----------------
    heatmap(t_pers["yes_rate"][pers].reindex(items), "TCS item coverage by personality",
            "item_tcs_by_personality", cbar_label="yes-rate")
    heatmap(t_chap["yes_rate"][chap].reindex(items), "TCS item coverage by ICD chapter",
            "item_tcs_by_chapter", cbar_label="yes-rate")
    heatmap(c_pers["mean"].unstack()[pers].reindex(qs), "Comprehension by question and personality",
            "item_comp_by_personality", cbar_label="mean score")
    heatmap(c_chap["mean"].unstack()[chap].reindex(qs), "Comprehension by question and ICD chapter",
            "item_comp_by_chapter", cbar_label="mean score")

    # ---------------- summary statistics (printed) ----------------
    F = []
    F.append("TCS items, yes-rate overall (applicable records):")
    F += [f"  {i:34s} {r.yes_rate:.3f}  (n={int(r.n)})" for i, r in t_all.iterrows()]
    F.append("\nTCS topic coverage (scorer weights): " + ", ".join(f"{k} {v:.3f}" for k, v in tp_all.items()))
    F.append("\nComprehension (SCS) per question: mean | correct/partial/incorrect | n")
    F += [f"  {q:30s} {r['mean']:.3f} | {r.correct:.2f}/{r.partially_correct:.2f}/{r.incorrect:.2f} | {int(r.n)}"
          for q, r in c_all.iterrows()]
    ty = t_pers["yes_rate"][pers]
    F.append("\nPersonality effect per TCS item (yes-rate; range across personalities; distrustful − mean of others):")
    eff = pd.DataFrame({"range": ty.max(1) - ty.min(1),
                        "distrustful_minus_others": ty["Distrustful"] - ty.drop(columns="Distrustful").mean(1)})
    F += [f"  {i:34s} range {r.range:.3f}  D−others {r.distrustful_minus_others:+.3f}" for i, r in eff.loc[items].iterrows()]
    cp = c_pers["mean"].unstack()[pers].reindex(qs)
    F.append("\nPersonality effect per Comprehension question (mean score):")
    F += [f"  {q:30s} " + "  ".join(f"{p[:5]} {cp.loc[q, p]:.3f}" for p in pers) +
          f"  | range {cp.loc[q].max() - cp.loc[q].min():.3f}" for q in qs]
    tc = t_chap["yes_rate"][chap]
    F.append("\nICD-chapter spread per TCS item (yes-rate range across chapters; highest / lowest chapter):")
    F += [f"  {i:34s} range {tc.loc[i].max() - tc.loc[i].min():.3f}  hi {tc.loc[i].idxmax()} {tc.loc[i].max():.2f} / lo {tc.loc[i].idxmin()} {tc.loc[i].min():.2f}" for i in items]
    cc = c_chap["mean"].unstack()[chap].reindex(qs)
    F.append("\nICD-chapter spread per Comprehension question:")
    F += [f"  {q:30s} range {cc.loc[q].max() - cc.loc[q].min():.3f}  hi {cc.loc[q].idxmax()} {cc.loc[q].max():.2f} / lo {cc.loc[q].idxmin()} {cc.loc[q].min():.2f}" for q in qs]
    # procedure split for the procedure-sensitive items
    pr = tcs[tcs.procedure != "unknown"].groupby(["item", "procedure"]).yes.mean().unstack()
    F.append("\nTCS item yes-rate by procedure header (listed / none):")
    F += [f"  {i:34s} {pr.loc[i, 'listed']:.3f} / {pr.loc[i, 'none']:.3f}" for i in items]
    # TCS topic coverage <-> matching Comprehension question, record level (same conversation)
    cov = topic_cov(tcs, ["model", "run", "note_id", "topic"]).rename("cov").reset_index()
    sc = comp[["model", "run", "note_id", "q", "score"]]
    F.append("\nTCS topic coverage vs matching Comprehension question (same conversation, record level):")
    F.append("  mean Comprehension score by coverage band  [coverage = 0 | 0<c<1 | 1]   Pearson r   n")
    for topic, q in PAIR.items():
        m = cov[cov.topic == topic].merge(sc[sc.q == q], on=["model", "run", "note_id"])
        band = pd.cut(m["cov"], [-0.01, 0.0001, 0.9999, 1.0], labels=["0", "partial", "1"])
        b = m.groupby(band, observed=False).score.mean()
        F.append(f"  {topic:22s} ↔ {q:28s} {b.get('0', np.nan):.3f} | {b.get('partial', np.nan):.3f} | "
                 f"{b.get('1', np.nan):.3f}   r={m['cov'].corr(m.score):.3f}   n={len(m)}")
    print("\n".join(F))


if __name__ == "__main__":
    main()
