"""
bootstrap_case_ci.py — across-CASE bootstrap confidence intervals for the
headline DischargeBench metrics.

This is the sampling CI a reader expects from "score ± x" on a clinical
benchmark: it answers "if we had drawn a different sample of 320 discharge
cases, how much would the mean score move?". It is a DIFFERENT quantity from the
run-to-run SD in variance_summary.csv (which reflects the 5 pipeline reruns with the
judge fixed). Report both, clearly labelled.

Method
------
For each model and metric we pool the 5 variance runs (run1..run5) so the point
estimate matches the reported 5-run headline mean:
  1. per case, average the metric over the runs in which it appears
     (per-case expected score; run noise is ~0.003-0.009, small vs case spread);
  2. point estimate = mean over the v3 cases of that per-case mean;
  3. bootstrap: resample the cases with replacement (B=10000, seeded), recompute
     the mean each time, take the 2.5 / 97.5 percentiles.

Cases are filtered to the canonical v3 set (CURATED_IDS). Metrics:
tcs_overall (tcs_score), comprehension_post (post_education_score),
harm_rate, unsafe_advice_rate, wrong_instruction_rate (per-case CLSA rates).

Output: data/processed/output/case_bootstrap_ci.csv  (+ a markdown table on stdout)
Usage:  python scripts/analysis/bootstrap_case_ci.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from utils import CONFIG   # noqa: E402
OUTPUT_ROOT = PROJECT / "data/processed/output"
V3 = PROJECT / "data/processed" / CONFIG["discharge_bench"]["test_file"]

B = CONFIG["analysis"]["case_bootstrap"]["B"]
SEED = CONFIG["analysis"]["case_bootstrap"]["seed"]
RUN_TAGS = CONFIG["runs"]
MODELS = [e["name"] for e in CONFIG["educators"]]   # config order (one RNG runs through them)

# metric -> (score file, per-case field OR list of fields to sum)
# unsafe_claims = the headline CLSA safety metric = the two educator-attributable
# harmful grid cells, neutral x harmful (unsafe_advice) + contradicted x harmful
# (wrong_instruction). It EXCLUDES entailed x harmful (relayed_bad_note = faithful
# relay of a harmful note, a note-quality property). harm_rate (all harmful) and
# relayed_bad_note are kept below as context, not headline.
METRICS = {
    "tcs_overall":          ("topic_checklist_score.jsonl", "tcs_score"),
    "comprehension_post":   ("comprehension_score_orig_na.jsonl", "post_education_score"),  # reported SCS (Comprehension; 'not applicable' items dropped)
    "unsafe_claims":        ("safety_commission_score.jsonl",
                             ["unsafe_advice_rate", "wrong_instruction_rate"]),
    "unsafe_advice_rate":   ("safety_commission_score.jsonl", "unsafe_advice_rate"),
    "wrong_instruction_rate": ("safety_commission_score.jsonl", "wrong_instruction_rate"),
    "relayed_bad_note_rate": ("safety_commission_score.jsonl", "relayed_bad_note_rate"),
    "harm_rate":            ("safety_commission_score.jsonl", "harm_rate"),
}


def curated_ids() -> set[str]:
    with open(V3) as f:
        return {json.loads(line)["note_id"] for line in f if line.strip()}


def load_case_field(path: Path, field) -> dict[str, float]:
    """note_id -> value for one run's score file (skips missing/None).
    `field` may be a str, or a list of fields to SUM (all must be present)."""
    fields = [field] if isinstance(field, str) else list(field)
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            vals = [rec.get(k) for k in fields]
            if all(v is not None for v in vals):
                out[rec["note_id"]] = float(sum(vals))
    return out


def per_case_run_mean(model: str, fname: str, field: str,
                      ids: set[str]) -> dict[str, float]:
    """Average each case's metric across the runs in which it appears."""
    acc: dict[str, list[float]] = {}
    for tag in RUN_TAGS:
        d = load_case_field(OUTPUT_ROOT / f"{model}__{tag}" / fname, field)
        for nid, v in d.items():
            if nid in ids:
                acc.setdefault(nid, []).append(v)
    return {nid: float(np.mean(vs)) for nid, vs in acc.items()}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 b: int = B, alpha: float = 0.05) -> tuple[float, float]:
    n = len(values)
    boots = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    return tuple(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def main() -> None:
    ids = curated_ids()
    print(f"v3 canonical cases: {len(ids)}")
    rng = np.random.default_rng(SEED)
    rows = []
    for metric, (fname, field) in METRICS.items():
        for model in MODELS:
            case_means = per_case_run_mean(model, fname, field, ids)
            if not case_means:
                print(f"  [warn] no data for {model} / {metric}")
                continue
            vals = np.array(list(case_means.values()), dtype=float)
            mean = float(vals.mean())
            lo, hi = bootstrap_ci(vals, rng)
            rows.append({
                "metric": metric, "model": model, "n_cases": len(vals),
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
                "half_width": (hi - lo) / 2,
            })
    df = pd.DataFrame(rows)
    out_csv = OUTPUT_ROOT / "case_bootstrap_ci.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}\n")

    # markdown, ranked within each metric
    for metric in METRICS:
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            continue
        higher_better = metric in ("tcs_overall", "comprehension_post")
        sub = sub.sort_values("mean", ascending=not higher_better)
        print(f"### {metric}  (95% across-case bootstrap CI, B={B})\n")
        print("| model | n | mean | 95% CI |")
        print("|---|---:|---:|---|")
        for _, r in sub.iterrows():
            print(f"| {r['model']} | {int(r['n_cases'])} | {r['mean']:.4f} "
                  f"| [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] |")
        print()


if __name__ == "__main__":
    main()
