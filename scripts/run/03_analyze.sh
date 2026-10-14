#!/bin/bash
# CPU-only: reported Comprehension score, headline statistics, stratified and audit analyses.
# Run after 02_evaluate.sh has completed for all eight educators.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT"
python scripts/scoring/score_comprehension_on.py       # Comprehension (N/A-reference items dropped)
python scripts/analysis/aggregate_variance_results.py  # Table 1 means ± SD
python scripts/analysis/bootstrap_case_ci.py           # case-bootstrap 95% CIs
python scripts/analysis/icd_chapter_analysis.py        # ICD-chapter stratification
python scripts/analysis/item_level_analysis.py         # item-level breakdown
python scripts/analysis/complete_case_audit.py         # completed-session audit
