"""
safety_commission_raw_score.py

Claim-Level Safety Audit (CLSA) of what the educator SAID (errors of commission), as
opposed to what it failed to say (omission — that is TCS).

Metric name: **CLSA**. This module, its sibling calculate stage, the config key
(`commission_limit`), and all on-disk outputs (`safety_commission_*.jsonl`,
`commission_summary.csv`) keep the `commission` stem for provenance; "commission" below
also means the patient-safety failure class (commission vs. omission), not a second name.

Targets unsafe medical advice and wrong medication instructions, which nothing else in the benchmark
catches:
  1. "unsafe medical advice"
  2. "wrong medication instructions"

Factual Consistency cannot: the concern is precisely that ROUGE-L and embedding
similarity cannot reliably detect these. TCS cannot either — it asks whether an item
was *covered*, so an educator saying "metoprolol 500 mg" when the note says 100 mg
scores as covered. TCS would credit a dangerous error.

Architecture: decompose output into atomic statements, then evaluate each against the
source. Harm definition anchors to the AMA Principles of Medical Ethics.

Two passes per (model, case):
  1. decompose educator speech -> atomic clinical claims   (conversation only)
  2. judge each claim on two INDEPENDENT axes               (claims + note)
       grounding       : entailed | contradicted | neutral   (NLI trichotomy)
       clinical_status : appropriate | questionable | harmful

The two axes must not collapse into each other. See the 2x3 grid in
safety_commission_calculate_score.py: `neutral` + `appropriate` is **ungrounded but
plausible action** — correct advice the note happened to omit — and is NOT a defect.

LLM-as-a-Jury
-------------
The JUDGE pass is juried; the DECOMPOSE pass is not (voting on a free-form claim list is
ill-defined). So decomposition runs ONCE and is cached per educator model at
`safety_commission_claims.jsonl`; every juror then judges that shared list, which is what
lets the calculate stage align claims by index and majority-vote each axis.

  * The first juror served populates the claim cache (it is the panel's decomposer). Set
    `evaluation.decompose_model` to name that juror explicitly; a later juror that finds
    no cache and is NOT the decomposer fails loudly rather than re-decomposing under a
    different model (which would desynchronise the claim lists).
  * The judge pass writes a juror-tagged file: `safety_commission_raw__<eval_model>.jsonl`.
    Single-judge behaviour is just a one-element panel.

Output: data/processed/output/{model}/safety_commission_claims.jsonl   (shared, once)
        data/processed/output/{model}/safety_commission_raw__{juror}.jsonl

Usage:
    # one juror (served on eval_port); repeat per juror in the SBATCH loop
    python src/evaluation/safety_commission_raw_score.py \
        "evaluation.eval_model=gemma-4-31B-it" "evaluation.eval_port=5002"
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

import hydra
import jinja2
import pandas as pd
from omegaconf import DictConfig
from openai import AsyncOpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_PATH

sys.path.insert(0, str(Path(__file__).parent))
import jury

PROMPTS_PATH = PROJECT_PATH / "prompts"
OUTPUT_ROOT  = PROJECT_PATH / "data/processed/output"
DATA_ROOT    = PROJECT_PATH / "data/processed"

CLAIMS_STEM = "safety_commission_claims"
RAW_STEM    = "safety_commission_raw"

# The judge scores ALL of a case's claims in one structured call, so its JSON output
# grows with claim count (up to ~109 claims/case in this corpus). The server's default
# output budget truncated that JSON mid-string on the largest/most-verbose cases ->
# pydantic "EOF while parsing a string" and a hard job failure (no silent drop). Cap
# generously: the worst prompt (~8.4k-token note + ~100 claims ~= 12k) + this budget
# still fits the 32k-context jurors (Qwen3-32B, Llama-3.3-70B), and 16k >> the ~4k
# real max output, so verdicts can never truncate.
MAX_OUTPUT_TOKENS = 16384


class Claim(BaseModel):
    claim: str
    verbatim: str
    category: Literal["medication", "warning_sign", "follow_up",
                      "self_care", "diagnosis", "other"]

class Claims(BaseModel):
    claims: list[Claim]


class Judgment(BaseModel):
    index: int
    grounding: Literal["entailed", "contradicted", "neutral"]
    note_quote: str
    clinical_status: Literal["appropriate", "questionable", "harmful"]
    reason: str

class Judgments(BaseModel):
    judgments: list[Judgment]


def load_records(model: str) -> list[dict]:
    path = OUTPUT_ROOT / model / "simulation_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No simulation results for '{model}' at {path}")
    records = [json.loads(l) for l in path.open() if l.strip()]
    valid = [r for r in records if not r.get("error")]
    if len(records) - len(valid):
        print(f"  [{model}] Skipped {len(records) - len(valid)} errored records.")
    return valid


def format_conversation(conversation: list[dict]) -> str:
    return "\n".join(f"{t['role'].capitalize()}: {t['message']}" for t in conversation)


async def _parse(client, model, api_style, prompt, schema, temperature):
    return await jury.structured_call(
        client, model, api_style, prompt, schema,
        temperature=temperature, max_output_tokens=MAX_OUTPUT_TOKENS,
    )


# ---------------------------------------------------------------------------
# Decompose (once, cached) — the claim set every juror shares
# ---------------------------------------------------------------------------

async def decompose_one(sem, record, templates, client, model, api_style, temperature):
    async with sem:
        decomposed = await _parse(
            client, model, api_style,
            templates["decompose"].render(
                conversation_history=format_conversation(record["conversation"])),
            Claims, temperature,
        )
    if decomposed is None:
        raise RuntimeError(
            f"Decomposer '{model}' returned schema-invalid output for note "
            f"{record['note_id']} after retries — cannot build the shared claim cache."
        )
    return {
        "note_id":          record["note_id"],
        "early_termination": record.get("early_termination"),
        "session_complete":  record.get("session_complete"),
        "claims":           [c.model_dump() for c in decomposed.claims],
    }


def load_claim_cache(model: str) -> dict[str, dict] | None:
    path = OUTPUT_ROOT / model / f"{CLAIMS_STEM}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return None
    recs = [json.loads(l) for l in path.open() if l.strip()]
    return {r["note_id"]: r for r in recs}


async def build_claim_cache(model, records, templates, client, decomposer, api_style, sem, temperature):
    print(f"[{model}] Decomposing {len(records)} cases under '{decomposer}' (cache miss)...")
    cache = await asyncio.gather(*[
        decompose_one(sem, r, templates, client, decomposer, api_style, temperature) for r in records
    ])
    out = OUTPUT_ROOT / model / f"{CLAIMS_STEM}.jsonl"
    pd.DataFrame(cache).to_json(out, lines=True, orient="records")
    print(f"[{model}] Cached {sum(len(c['claims']) for c in cache)} claims -> {out}")
    return {r["note_id"]: r for r in cache}


# ---------------------------------------------------------------------------
# Judge (per juror) — the juried pass
# ---------------------------------------------------------------------------

async def judge_one(sem, cache_rec, note_text, templates, client, juror, api_style, temperature):
    claims = cache_rec["claims"]
    if not claims:
        return {"note_id": cache_rec["note_id"],
                "early_termination": cache_rec.get("early_termination"),
                "session_complete":  cache_rec.get("session_complete"),
                "claims": []}
    judge_prompt = templates["judge"].render(claims=claims, note_text=note_text)
    # Jurors occasionally return an incomplete verdict list (drop claim indices). Retry
    # with a temperature bump to break the determinism that causes it; abstain if still
    # incomplete. Attempt 0 keeps the configured temperature so the common path matches
    # the other jurors.
    judged = None
    for attempt in range(4):
        t = temperature if attempt == 0 else 0.6
        async with sem:
            judged_raw = await _parse(client, juror, api_style, judge_prompt, Judgments, t)
        if judged_raw is None:
            print(f"  [RETRY] juror '{juror}' schema-invalid verdicts on {cache_rec['note_id']}, "
                  f"attempt {attempt + 1}/4")
            continue
        judged = {j.index: j for j in judged_raw.judgments}
        missing = set(range(len(claims))) - set(judged)
        if not missing:
            break
        print(f"  [RETRY] juror '{juror}' incomplete verdicts on {cache_rec['note_id']} "
              f"(missing {sorted(missing)} of {len(claims)}), attempt {attempt + 1}/4")
    else:
        print(f"  [ABSTAIN] juror '{juror}' still-incomplete verdicts on {cache_rec['note_id']} "
              f"after 4 tries — skipping this juror's vote for this case.")
        return None
    out = []
    for i, c in enumerate(claims):
        j = judged[i]
        out.append({
            **c,
            "grounding":       j.grounding,
            "note_quote":      j.note_quote,
            "clinical_status": j.clinical_status,
            "reason":          j.reason,
        })
    return {
        "note_id":          cache_rec["note_id"],
        "early_termination": cache_rec.get("early_termination"),
        "session_complete":  cache_rec.get("session_complete"),
        "claims":           out,
    }


async def evaluate_model(model, notes, templates, client, juror, api_style, decomposer,
                         concurrency, temperature, limit=None):
    print(f"\n[{model}] Loading records...")
    records = [r for r in load_records(model) if r["note_id"] in notes]
    if limit:
        records = records[:int(limit)]
        print(f"[{model}] LIMIT={limit} — smoke test, not a full run.")

    sem = asyncio.Semaphore(concurrency)

    cache = load_claim_cache(model)
    if cache is None:
        if str(juror) != str(decomposer):
            raise RuntimeError(
                f"[{model}] No cached claims and juror '{juror}' is not the decomposer "
                f"'{decomposer}'. Serve the decomposer first so it populates "
                f"{CLAIMS_STEM}.jsonl, then run the other jurors."
            )
        cache = await build_claim_cache(model, records, templates, client, decomposer,
                                        api_style, sem, temperature)
    else:
        print(f"[{model}] Reusing cached claims ({len(cache)} cases).")

    # judge only the cases we have claims for (intersect with requested records)
    to_judge = [cache[r["note_id"]] for r in records if r["note_id"] in cache]
    print(f"[{model}] Juror '{juror}' judging {len(to_judge)} cases...")
    results = [r for r in await asyncio.gather(*[
        judge_one(sem, cr, notes[cr["note_id"]], templates, client, juror, api_style, temperature)
        for cr in to_judge
    ]) if r is not None]
    abstained = len(to_judge) - len(results)
    out = OUTPUT_ROOT / model / f"{RAW_STEM}__{juror}.jsonl"
    pd.DataFrame(results).to_json(out, lines=True, orient="records")
    n_claims = sum(len(r["claims"]) for r in results)
    msg = f"[{model}] Saved {len(results)} cases / {n_claims} claims -> {out}"
    if abstained:
        msg += f"  ({abstained} abstained: incomplete verdicts after retries)"
    print(msg)


async def async_main(cfg: DictConfig) -> None:
    ev = cfg.evaluation
    df = pd.read_json(DATA_ROOT / cfg.discharge_bench.test_file, lines=True, orient="records")
    notes = dict(zip(df["note_id"], df["text"]))
    print(f"Loaded {len(notes)} notes from {cfg.discharge_bench.test_file}")

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(PROMPTS_PATH)))
    templates = {
        "decompose": env.get_template("claim_decompose.jinja"),
        "judge":     env.get_template("claim_safety_judge.jinja"),
    }
    eval_port = ev.get("eval_port", None)
    client, api_style = jury.make_judge_client(ev.eval_model, eval_port)
    temperature = ev.get("eval_temperature", 0)
    juror = str(ev.eval_model)
    decomposer = str(ev.get("decompose_model", None) or ev.eval_model)
    print(f"Juror (judge): {juror}  |  Decomposer: {decomposer}")

    limit = ev.get("commission_limit", None)
    for model in list(ev.models):
        await evaluate_model(model, notes, templates, client, juror, api_style, decomposer,
                             ev.concurrency, temperature, limit)
    print("\nAll models judged by this juror.")


@hydra.main(config_path=PROJECT_PATH.joinpath("config").as_posix(),
            config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
