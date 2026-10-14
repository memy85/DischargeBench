"""
aggregate_variance_results.py

Aggregates the 40 (educator x run) pseudo-model directories written by
scripts/run/01_simulate.sh (5 unseeded runs per educator) and scored by
scripts/run/02_evaluate.sh (single judge, evaluation.models=[<educator>__run1, ...])
into the variance-study statistics: mean +/- SD across runs, a sanity panel, and a comparison against the historical single-run scores.

Design: full 320-case set, 5 unseeded simulation repeats, judge held fixed.

Assumes the evaluation pipeline has ALREADY been run against all 40
<educator>__run<N> directories; this script does no LLM calls of its own.

Usage:
    python scripts/analysis/aggregate_variance_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from utils import CONFIG   # noqa: E402
DATA_ROOT   = PROJECT_DIR / "data/processed"
OUTPUT_ROOT = DATA_ROOT / "output"
CURATED_PATH = DATA_ROOT / CONFIG["discharge_bench"]["test_file"]

EDUCATORS = [e["name"] for e in CONFIG["educators"]]   # config order (shared bootstrap RNG)
RUN_TAGS = CONFIG["runs"]

N_BOOTSTRAP = CONFIG["analysis"]["run_sd_bootstrap"]["B"]
BOOT_SEED = CONFIG["analysis"]["run_sd_bootstrap"]["seed"]  # aggregation-side bootstrap only; unrelated to the (unseeded) sim runs

HEADLINE_METRICS = ["tcs_overall", "comprehension_post"]  # post = final understanding (shift kept as secondary)

CQ_DIMS = {
    "naturalness":       "cq_naturalness",
    "responsiveness":    "cq_responsiveness",
    "clarity":            "cq_clarity",
    "clinical_relevance": "cq_clinical_relevance",
}
# Written by conversation_quality_calculate_score.py (nested {dim: {"score": ...}}).
CQ_FILE = "conversation_quality_score.jsonl"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def curated_ids() -> set[str]:
    df = pd.read_json(CURATED_PATH, lines=True)
    return set(df["note_id"])


def _read_filtered(path: Path, ids: set[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_json(path, lines=True, orient="records")
    if "note_id" not in df.columns or not len(df):
        return None
    return df[df["note_id"].isin(ids)].reset_index(drop=True)


def _long(educator: str, run_tag: str, note_ids: pd.Series, metric: str,
          values: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "educator": educator, "run_tag": run_tag,
        "note_id": note_ids.to_numpy(), "metric": metric, "value": values.to_numpy(),
    })


def load_cq(model_dir: Path, ids: set[str]) -> pd.DataFrame | None:
    df = _read_filtered(model_dir / CQ_FILE, ids)
    if df is not None and "conversation_quality" in df.columns:
        return df
    return None


def load_run_long(educator: str, run_tag: str, ids: set[str]) -> pd.DataFrame:
    model_dir = OUTPUT_ROOT / f"{educator}__{run_tag}"
    parts: list[pd.DataFrame] = []

    if not model_dir.exists():
        print(f"  [WARN] missing directory: {model_dir}")
        return pd.DataFrame(columns=["educator", "run_tag", "note_id", "metric", "value"])

    tc = _read_filtered(model_dir / "topic_checklist_score.jsonl", ids)
    if tc is not None:
        parts.append(_long(educator, run_tag, tc["note_id"], "tcs_overall", tc["tcs_score"]))
    else:
        print(f"  [WARN] {model_dir}/topic_checklist_score.jsonl missing/empty")

    co = _read_filtered(model_dir / "comprehension_score_orig_na.jsonl", ids)
    if co is not None:
        parts.append(_long(educator, run_tag, co["note_id"], "comprehension_post", co["post_education_score"]))
    else:
        print(f"  [WARN] {model_dir}/comprehension_score_orig_na.jsonl missing/empty")


    cq = load_cq(model_dir, ids)
    if cq is not None:
        for dim, metric_name in CQ_DIMS.items():
            values = cq["conversation_quality"].apply(lambda x, d=dim: x[d]["score"])
            parts.append(_long(educator, run_tag, cq["note_id"], metric_name, values))
    else:
        print(f"  [WARN] {model_dir}/conversation_quality*.jsonl missing/empty")

    sc = _read_filtered(model_dir / "safety_commission_score.jsonl", ids)
    if sc is not None:
        for col in ["harm_rate", "unsafe_advice_rate", "wrong_instruction_rate"]:
            parts.append(_long(educator, run_tag, sc["note_id"], col, sc[col]))
    else:
        print(f"  [WARN] {model_dir}/safety_commission_score.jsonl missing/empty")

    if not parts:
        return pd.DataFrame(columns=["educator", "run_tag", "note_id", "metric", "value"])
    return pd.concat(parts, ignore_index=True)


def build_long_df(ids: set[str]) -> pd.DataFrame:
    parts = []
    for educator in EDUCATORS:
        for run_tag in RUN_TAGS:
            print(f"Loading {educator} / {run_tag} ...")
            parts.append(load_run_long(educator, run_tag, ids))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-run means, cross-run mean/SD
# ---------------------------------------------------------------------------

def per_run_means(long_df: pd.DataFrame) -> pd.DataFrame:
    """Case-averaged score per (educator, run_tag, metric)."""
    return (
        long_df.dropna(subset=["value"])
        .groupby(["educator", "run_tag", "metric"], as_index=False)["value"]
        .mean()
    )


def cross_run_summary(run_means: pd.DataFrame) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        vals = g["value"].to_numpy()
        return pd.Series({
            "n_runs": len(vals),
            "mean": np.mean(vals) if len(vals) else np.nan,
            "sd": np.std(vals, ddof=1) if len(vals) >= 2 else np.nan,
        })
    return (
        run_means.groupby(["educator", "metric"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )


def bootstrap_sd_ci(values: np.ndarray, rng: np.random.Generator,
                     B: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.std(values[idx], ddof=1)
    return tuple(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def add_sd_bootstrap_ci(run_means: pd.DataFrame, summary: pd.DataFrame,
                         metrics: list[str], seed: int = BOOT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lo_col, hi_col = [], []
    for _, row in summary.iterrows():
        if row["metric"] not in metrics:
            lo_col.append(np.nan); hi_col.append(np.nan)
            continue
        vals = run_means[(run_means["educator"] == row["educator"])
                          & (run_means["metric"] == row["metric"])]["value"].to_numpy()
        lo, hi = bootstrap_sd_ci(vals, rng)
        lo_col.append(lo); hi_col.append(hi)
    summary = summary.copy()
    summary["sd_ci_lo"] = lo_col
    summary["sd_ci_hi"] = hi_col
    return summary


# ---------------------------------------------------------------------------
# Sanity panel
# ---------------------------------------------------------------------------

def sanity_row(educator: str, run_tag: str) -> dict:
    path = OUTPUT_ROOT / f"{educator}__{run_tag}" / "simulation_results.jsonl"
    if not path.exists():
        return {"educator": educator, "run_tag": run_tag, "n_cases": 0,
                "error_rate": None, "early_termination_rate": None,
                "discard_rate": None, "mean_turns": None}

    df = pd.read_json(path, lines=True, orient="records")
    n = len(df)
    if not n:
        return {"educator": educator, "run_tag": run_tag, "n_cases": 0,
                "error_rate": None, "early_termination_rate": None,
                "discard_rate": None, "mean_turns": None}

    is_error = df["error"].notna()
    n_accepted_sum = df["n_accepted"].sum()
    n_discarded_sum = df["n_discarded"].sum()
    denom = n_accepted_sum + n_discarded_sum
    ok = df[~is_error]

    return {
        "educator": educator, "run_tag": run_tag, "n_cases": n,
        "error_rate": float(is_error.mean()),
        "early_termination_rate": float(df["early_termination"].mean()),
        "discard_rate": float(n_discarded_sum / denom) if denom else None,
        "mean_turns": float(ok["n_accepted"].mean()) if len(ok) else None,
    }


def build_sanity_panel() -> pd.DataFrame:
    rows = [sanity_row(e, r) for e in EDUCATORS for r in RUN_TAGS]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison against the existing single-run (unseeded) headline scores
# ---------------------------------------------------------------------------

def historical_comparison(summary: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    rows = []
    for educator in EDUCATORS:
        model_dir = OUTPUT_ROOT / educator
        tc = _read_filtered(model_dir / "topic_checklist_score.jsonl", ids)
        co = _read_filtered(model_dir / "comprehension_score_orig_na.jsonl", ids)
        hist = {
            "tcs_overall": float(tc["tcs_score"].mean()) if tc is not None else None,
            "comprehension_post": float(co["post_education_score"].mean()) if co is not None else None,
        }
        for metric, hist_val in hist.items():
            run_row = summary[(summary["educator"] == educator) & (summary["metric"] == metric)]
            run_mean = float(run_row["mean"].iloc[0]) if len(run_row) else None
            rows.append({
                "educator": educator, "metric": metric,
                "run_mean_5runs": run_mean, "historical_single_run": hist_val,
                "diff": (run_mean - hist_val) if run_mean is not None and hist_val is not None else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ids = curated_ids()
    print(f"Canonical curated set: {len(ids)} note_ids ({CURATED_PATH.name})\n")

    long_df = build_long_df(ids)
    if not len(long_df):
        print("\nNo score files found under any <educator>__runN directory. "
              "Run the evaluation pipeline against evaluation.models=[<educator>__run1, ...] "
              "for all 40 directories first.")
        return

    run_means = per_run_means(long_df)
    summary = cross_run_summary(run_means)
    summary = add_sd_bootstrap_ci(run_means, summary, metrics=HEADLINE_METRICS)

    sanity = build_sanity_panel()
    hist = historical_comparison(summary, ids)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "variance_summary.csv"
    sanity_path = OUTPUT_ROOT / "variance_sanity_panel.csv"
    hist_path = OUTPUT_ROOT / "variance_historical_comparison.csv"

    summary.to_csv(summary_path, index=False)
    sanity.to_csv(sanity_path, index=False)
    hist.to_csv(hist_path, index=False)

    print(f"\nSaved:\n  {summary_path}\n  {sanity_path}\n  {hist_path}")


if __name__ == "__main__":
    main()
