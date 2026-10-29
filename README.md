# DischargeBench

Code for **DischargeBench: Pre-Deployment Evaluation of LLM Discharge Educators through Persona-Grounded Patient Simulation** (Won Seok Jang, Youngbin Jang, Chanyoung Ko, Hyunwook Kim, Zonghai Yao, Hong Yu).

[![arXiv](https://img.shields.io/badge/arXiv-2609.20827-b31b1b.svg)](https://arxiv.org/abs/2609.20827)

DischargeBench is a pre-deployment benchmark for LLMs as discharge educators. A candidate Educator is given a real discharge note and teaches a persona-grounded Virtual Patient about their discharge in an open-ended, multi-turn conversation: diagnosis, treatment during the stay, new medications, follow-up and the warning signs that mean coming back to hospital. The Virtual Patient varies in personality, education level, health literacy and recall of their own medical history. A Simulation Orchestrator regulates the patient side only, so the Educator's behaviour is never altered. Each conversation is then scored by an LLM judge validated against physician annotations.

DischargeBench is intended as a screening step before human evaluation, not as evidence that a model is ready to educate patients.

This repository contains the simulation harness, every prompt, the evaluation code and the analysis scripts. The benchmark cases are derived from MIMIC-IV-Note and are distributed separately through PhysioNet (see [Data](#data)).

## Changes in v1.0.1

The paper's results and the simulation records released on PhysioNet were produced with v1.0.0. v1.0.1 changes how failures are handled in new runs; it does not change how any metric is defined.

**Simulation**
- The Simulation Orchestrator no longer substitutes a default verdict when its judge call fails. In v1.0.0 a failed call let the turn through unjudged and could never end the session; once a session outgrew the Orchestrator's context, every later call failed this way. The call is now retried (3 attempts for invalid output, 5 with backoff for connection or server errors); if it still fails, the case is recorded with an `error` starting `env_agent_error:`.
- The Orchestrator's `severity` is restricted to `minor`, `moderate` or `severe`. In v1.0.0 it was free text, and any value other than exactly `minor` or `moderate` was treated as severe.
- An empty or truncated educator reply ends the case with an `error` instead of being passed to the patient.
- Line breaks in educator replies are kept. In v1.0.0 they were removed from every reply containing a `</think>` tag (the Qwen3 educators).
- On resume, errored records are removed from the output file before their cases are re-run, so each case appears once. A malformed line in the output file stops the resume instead of being skipped.

**Evaluation**
- SCS no longer asks the comprehension questions before the conversation; only the post-conversation score (`post_education_score`) is computed. The reported SCS never used the pre-conversation score.
- A comprehension question whose patient answer or judge call still fails after retries is recorded with an `error` and no score, instead of being graded incorrect. The case's `post_education_score` is then null, and `score_comprehension_on.py` stops until the case is re-run.
- `safety_commission_calculate_score.py` no longer writes `data/processed/output/commission_summary.csv`, which each run overwrote; per-case CLSA scores are unchanged.

## DischargeBench Overview

<p align="center"><img src="assets/fig_overview.png" alt="DischargeBench overview: Stage I simulates the dialogue between the Educator and the Virtual Patient under the Simulation Orchestrator; Stage II scores it with an LLM judge on Conversation Quality, TCS, SCS and CLSA." width="440"></p>

- **Virtual Patient:** `meta-llama/Llama-3.3-70B-Instruct`. Five personalities (neutral, anxious, distrustful, high conscientiousness, minimizer), three education levels (elementary, high school, college), two health-literacy levels (low, high) and three levels of past-medical-history recall (poor, partial, accurate).
- **Simulation Orchestrator:** `Qwen/Qwen3.5-9B`, thinking disabled (implemented in `src/environment_agent.py`). It checks each patient turn, corrects out-of-character turns, and decides when the session is complete.
- **Judge:** `google/gemma-4-31B-it` at temperature 0 for all metrics.

## Metrics

| Metric | What it measures | Code |
|---|---|---|
| **TCS**, Topic Checklist Score | Coverage of a 21-question discharge checklist, gated to the topics the note actually contains | `src/evaluation/topic_checklist_*.py` |
| **SCS**, Simulated Comprehension Score | The Virtual Patient answers six questions after the session; answers are graded against references from the note | `src/evaluation/comprehension_score.py`, `scripts/scoring/score_comprehension_on.py` |
| **CLSA**, Claim-Level Safety Audit | Educator turns are split into atomic claims; each claim is judged for grounding in the note and clinical safety. Reports unsafe-advice and wrong-instruction rates | `src/evaluation/safety_commission_*.py` |
| **Conversation Quality** | Naturalness, responsiveness, clarity and clinical relevance on 1–5 scales, plus Flesch–Kincaid grade | `src/evaluation/conversation_quality*.py` |

## Repository layout

```
config/config.yaml          every setting: models, vLLM servers, prompts, simulation and evaluation hyperparameters
src/                        simulation harness
  virtual_patient.py          Virtual Patient
  educator_agent.py           Educator under test
  environment_agent.py        Simulation Orchestrator
  case_session.py             single-case driver
  discharge_bench.py          parallel runner over the benchmark (resumable)
  evaluation/                 LLM-judge metrics (Conversation Quality, TCS, SCS, CLSA)
prompts/                    Jinja templates for every LLM role and judge
scripts/
  run/                        end-to-end scripts: 01 simulate → 02 evaluate → 03 analyze
  scoring/                    reported SCS
  analysis/                   mean ± SD over runs, case-bootstrap CIs, ICD-chapter, item-level and audit analyses
notebooks/                  persona / patient-factor / ICD-chapter stratification and session-outcome analysis
assets/                     README figure
```

## Setup

Python 3.12. Open-weight models are served with [vLLM](https://github.com/vllm-project/vllm) 0.19 on NVIDIA A100-80GB GPUs. GPT educators use the OpenAI API (`OPENAI_API_KEY`). Analysis runs on CPU.

```bash
git clone https://github.com/memy85/DischargeBench.git && cd DischargeBench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install vllm==0.19.0     # GPU hosts only
```

Educators evaluated in the paper: GPT-5.5, GPT-5.4-mini, GPT-5.4-nano (API); Qwen3-32B, Qwen3-4B, MedGemma-27B-text-it, MedGemma-4B-it, Llama-3.3-70B-Instruct (vLLM). Their Hugging Face ids and served names are in `config/config.yaml`.

## Configuration

All settings live in `config/config.yaml`:

| Section | What it sets |
|---|---|
| `educators`, `runs` | educators under test (served name, Hugging Face id or OpenAI API, display name) and the run tags |
| `servers` | Hugging Face ids, ports, vLLM arguments and GPU assignments for the Virtual Patient, Simulation Orchestrator, educator and judge |
| `prompts` | the template file under `prompts/` for every role and judge |
| `discharge_bench` | simulation: case file, `max_turns`, parallel workers, patient-turn retries, `min_turns_before_complete`, opening message |
| `evaluation` | judge model, temperature, concurrency, output-token caps, SCS settings |
| `analysis` | table/figure order, bootstrap sizes and seeds, audit thresholds |

The Python entry points use [Hydra](https://hydra.cc), so `discharge_bench.*` and `evaluation.*` keys can be overridden on the command line, e.g. `python src/discharge_bench.py discharge_bench.simulation.max_turns=60`. The run scripts read `servers`, `educators` and `runs` through `scripts/run/config.py`.

## Data

The benchmark has **320 MIMIC-IV discharge summaries**, 16 ICD chapters × 20 cases, balanced within each chapter on personality (5 × 4) and health literacy (10 low / 10 high). ICD-9 / ICD-10 is 10 / 10 within each of the 8 chapters whose ICD-9 and ICD-10 titles were merged.

Each case record holds the discharge note, its ICD code and chapter, the patient profile extracted from the note with the assigned persona, and the note-grounded checklist fields that gate each TCS question.

The case records derive from MIMIC-IV (v3.1) and MIMIC-IV-Note (v2.2) and cannot be redistributed here. They will be released on PhysioNet as **MIMIC-IV-Ext-DischargeBench**, together with the complete simulation records of the eight educators in the paper; the link will be added here when the release is available. Access requires credentialed PhysioNet access, a signed PhysioNet Credentialed Health Data Use Agreement (v1.5.0), and the CITI "Data or Specimens Only Research" training.

The release's `data/` folder has the same layout as this repository. Copy it into the repository root, merging with `data/`:

```
data/processed/discharge_bench_curated_v3.jsonl                     the 320 cases
data/processed/output/<educator>__run<r>/simulation_results.jsonl   our simulation records
```

`02_evaluate.sh` can then score the released simulation records without re-simulating. The simulator skips cases already in an output file, so to simulate one of the eight educators again, use a new run tag (e.g. `runs: [run6]` in `config/config.yaml`).

## Running the benchmark

Outputs go to `data/processed/output/<educator>__run<r>/`, one directory per educator and run.

```bash
# 1. Simulation (GPU): 5 runs of one educator over the 320 cases
bash scripts/run/01_simulate.sh Qwen3-32B          # any educator listed under educators in config/config.yaml

# 2. Evaluation (GPU): Conversation Quality, TCS, SCS and CLSA for each run
bash scripts/run/02_evaluate.sh Qwen3-32B

# 3. Scoring and analysis (CPU)
bash scripts/run/03_analyze.sh
```

The run scripts start the vLLM servers themselves and assume one node with up to five A100-80GB GPUs; change `servers.gpus` in the config or wrap the scripts in a scheduler job as needed. Every stage is resumable: cases already present in an output file are skipped.

To evaluate a new educator, add it under `educators` in `config/config.yaml` (`hf_id: null` routes it to the OpenAI API) and to `analysis.display_order`, then run steps 1–3.

## Where each result comes from

| Result | Script | Output |
|---|---|---|
| Main results: 5-run mean ± SD per educator | `scripts/analysis/aggregate_variance_results.py` | `data/processed/output/variance_summary.csv` |
| Case-bootstrap 95% CIs (B = 10,000) | `scripts/analysis/bootstrap_case_ci.py` | `data/processed/output/case_bootstrap_ci.csv` |
| ICD-chapter stratification | `scripts/analysis/icd_chapter_analysis.py` | `outputs/icd_chapter_table.csv` |
| Persona, patient-factor and ICD-chapter stratification | `notebooks/stratified_analysis.ipynb` | `figures/` |
| Session-outcome analysis | `notebooks/failure_analysis.ipynb` | `figures/` |
| Completed-session audit | `scripts/analysis/complete_case_audit.py` | `outputs/clean_complete_audit.csv` |
| Item-level breakdown | `scripts/analysis/item_level_analysis.py` | `outputs/item_analysis/` |

### Reproducibility notes

- The judge runs at temperature 0. Simulation is unseeded by design: the five runs measure run-to-run variance.
- Bootstrap CIs are seeded.
- `python scripts/scoring/score_comprehension_on.py --check` verifies existing SCS outputs without rewriting them.
- Figures are set in Fira Sans (SIL Open Font License, from Google Fonts). The plotting code loads any `.ttf` files placed in `fonts/`; without them matplotlib uses its default font.

## Citation

```bibtex
@article{jang2026dischargebench,
  title   = {{DischargeBench}: Pre-Deployment Evaluation of {LLM} Discharge Educators through Persona-Grounded Patient Simulation},
  author  = {Jang, Won Seok and Jang, Youngbin and Ko, Chanyoung and Kim, Hyunwook and Yao, Zonghai and Yu, Hong},
  journal = {arXiv preprint arXiv:2609.20827},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Use of the MIMIC-derived data is governed by the PhysioNet Data Use Agreement.
