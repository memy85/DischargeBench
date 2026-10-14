"""
jury.py — LLM-as-a-Jury aggregation primitives (Panel of LLM evaluators).

A panel of diverse models
scores independently, then votes are aggregated. Replaces the single-judge design
across the DischargeBench evaluation stages.

This module holds only the pure, metric-agnostic primitives — voting, mean
pooling, and inter-juror agreement. Each metric's calculate_score.py aligns its
own units (a TCS question, a commission claim, a Comprehension question, a Likert dimension)
and calls into here.

Design rules (match the single-judge code they replace):

  * Categorical labels  -> strict majority vote. With an odd panel a BINARY label
    always resolves. A k>=3-way label (commission grounding/clinical_status,
    comprehension correctness) CAN split 1-1-1 with no majority; that is a TIE.
    The user's aggregation choice is "odd panel + strict majority", so ties are
    NOT silently turned into a score. `resolve()` breaks them with an explicit,
    recorded policy (default: the primary/most-validated juror), and the tie is
    counted so its frequency is reported, never hidden.

  * Ordinal Likert scores -> mean pooling.

  * A juror that failed / abstained contributes no vote (None is dropped). An
    empty panel yields label=None — never a fabricated default. This mirrors the
    "empty denominator -> None, never 0.0" rule used throughout the metrics.

Nothing here reads a model or hits the network; it is CPU-only and deterministic,
so it degenerates exactly to the single-judge result when the panel has size 1
(the identity used to validate the migration).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

import openai
from openai import AsyncOpenAI
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Categorical voting
# ---------------------------------------------------------------------------

@dataclass
class VoteResult:
    """Outcome of a categorical vote over one unit.

    label      : the plurality label among present votes (None if no votes).
    counts     : label -> number of jurors that chose it (present votes only).
    n_present  : jurors that returned a usable vote.
    n_total    : jurors polled (present + abstained/failed).
    is_majority: some label was held by a strict majority of PRESENT votes.
    is_unanimous: all present votes agree.
    is_tie     : two or more labels share the top count (no strict plurality winner).
    """
    label: str | None
    counts: dict[str, int]
    n_present: int
    n_total: int
    is_majority: bool
    is_unanimous: bool
    is_tie: bool


def vote(labels: list) -> VoteResult:
    """Tally categorical votes. None entries are treated as abstentions.

    `label` is the plurality winner; on a top-count tie it is the FIRST label
    (by first appearance) among the tied leaders, and `is_tie` is set so callers
    can override via `resolve()`. `is_majority` means strict >50% of present votes.
    """
    n_total = len(labels)
    present = [x for x in labels if x is not None]
    n_present = len(present)
    if n_present == 0:
        return VoteResult(None, {}, 0, n_total, False, False, False)

    counts = Counter(present)
    top = max(counts.values())
    leaders = [lab for lab in _first_seen_order(present) if counts[lab] == top]
    is_tie = len(leaders) > 1
    label = leaders[0]
    is_majority = top * 2 > n_present
    is_unanimous = len(counts) == 1
    return VoteResult(label, dict(counts), n_present, n_total,
                      is_majority, is_unanimous, is_tie)


def _first_seen_order(seq: list) -> list:
    """Distinct elements in order of first appearance (tie-break determinism)."""
    seen, order = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            order.append(x)
    return order


def resolve(labels: list, *, tie_break: list | None = None,
            primary_index: int | None = None) -> tuple[str | None, VoteResult]:
    """Vote, then settle any tie with an explicit, recorded policy.

    Returns (final_label, VoteResult). The VoteResult still reports is_tie so the
    diagnostic can count how often the tie-break was needed.

    Tie policy, in priority order:
      1. `tie_break`: a severity/priority list; the tied leader earliest in it wins
         (e.g. ["harmful","questionable","appropriate"] for a safety-conservative
         break — NOT the default, which the user declined).
      2. `primary_index`: fall back to that juror's own label (default: the
         most-validated juror, e.g. gemma). This keeps the metric fully defined
         without inventing a value.
      3. Otherwise: first-seen order (deterministic but arbitrary).
    """
    res = vote(labels)
    if not res.is_tie or res.label is None:
        return res.label, res

    leaders = [lab for lab, c in res.counts.items()
               if c == max(res.counts.values())]
    if tie_break is not None:
        for lab in tie_break:
            if lab in leaders:
                return lab, res
    if primary_index is not None and 0 <= primary_index < len(labels):
        pl = labels[primary_index]
        if pl in leaders:
            return pl, res
    return res.label, res  # first-seen fallback


# ---------------------------------------------------------------------------
# Ordinal (Likert) pooling
# ---------------------------------------------------------------------------

def mean_pool(scores: list) -> float | None:
    """Average of present scores; None if the panel gave nothing. Mean
    pooling for Likert judgements (conversation quality, readability)."""
    present = [float(s) for s in scores if s is not None]
    return sum(present) / len(present) if present else None


def median_pool(scores: list) -> float | None:
    present = sorted(float(s) for s in scores if s is not None)
    n = len(present)
    if n == 0:
        return None
    mid = n // 2
    return present[mid] if n % 2 else (present[mid - 1] + present[mid]) / 2


# ---------------------------------------------------------------------------
# Inter-juror agreement (Fleiss/Cohen kappa)
# ---------------------------------------------------------------------------

@dataclass
class AgreementReport:
    n_units: int
    n_raters: int
    categories: list = field(default_factory=list)
    percent_agreement: float | None = None   # mean pairwise agreement per unit
    fleiss_kappa: float | None = None
    unanimous_rate: float | None = None
    tie_rate: float | None = None


def fleiss_kappa(rows: list[list[int]]) -> float | None:
    """Fleiss' kappa from a units x categories count matrix.

    Each row is one unit's per-category rater counts; rows must share the same
    rater total (units rated by fewer raters are dropped by the caller before
    this point, or excluded here). Returns None if undefined (0/1 units, or
    P_e == 1, i.e. a single category).
    """
    rows = [r for r in rows if sum(r) >= 2]
    if len(rows) < 1:
        return None
    n = sum(rows[0])
    if any(sum(r) != n for r in rows):
        # unequal rater counts per unit — Fleiss requires a fixed n; caller aligns
        return None
    N = len(rows)
    k = len(rows[0])
    if n < 2:
        return None

    # P_i: agreement within unit i
    P = []
    for r in rows:
        agree = sum(c * (c - 1) for c in r)
        P.append(agree / (n * (n - 1)))
    P_bar = sum(P) / N

    # p_j: marginal proportion for category j
    col_tot = [sum(rows[i][j] for i in range(N)) for j in range(k)]
    total = N * n
    p = [c / total for c in col_tot]
    P_e = sum(pj * pj for pj in p)

    if math.isclose(P_e, 1.0):
        return None
    return (P_bar - P_e) / (1 - P_e)


def agreement(unit_labels: list[list], categories: list | None = None) -> AgreementReport:
    """Inter-juror agreement over units. `unit_labels[i]` is the list of juror
    labels for unit i (None = abstention, dropped per unit). Reports mean pairwise
    percent agreement, Fleiss' kappa (over units with a fixed present-rater count),
    unanimous rate, and tie rate.
    """
    # infer category set
    cats = categories or _first_seen_order(
        [lab for u in unit_labels for lab in u if lab is not None])
    cat_index = {c: j for j, c in enumerate(cats)}

    n_units = len(unit_labels)
    if n_units == 0:
        return AgreementReport(0, 0, cats)

    # pairwise percent agreement per unit (present raters only)
    pair_agr, unanimous, tie = [], 0, 0
    for u in unit_labels:
        present = [x for x in u if x is not None]
        m = len(present)
        if m >= 2:
            same = sum(1 for i in range(m) for j in range(i + 1, m)
                       if present[i] == present[j])
            pair_agr.append(same / (m * (m - 1) / 2))
        r = vote(u)
        if r.label is not None:
            unanimous += int(r.is_unanimous)
            tie += int(r.is_tie)

    # Fleiss over the modal present-rater count (fixed n requirement)
    present_counts = [len([x for x in u if x is not None]) for u in unit_labels]
    fixed_n = Counter(c for c in present_counts if c >= 2).most_common(1)
    kappa = None
    n_raters = 0
    if fixed_n:
        n_raters = fixed_n[0][0]
        rows = []
        for u in unit_labels:
            present = [x for x in u if x is not None]
            if len(present) == n_raters:
                row = [0] * len(cats)
                for x in present:
                    row[cat_index[x]] += 1
                rows.append(row)
        kappa = fleiss_kappa(rows)

    scored = sum(1 for u in unit_labels if vote(u).label is not None)
    return AgreementReport(
        n_units=n_units,
        n_raters=n_raters,
        categories=cats,
        percent_agreement=(sum(pair_agr) / len(pair_agr)) if pair_agr else None,
        fleiss_kappa=kappa,
        unanimous_rate=(unanimous / scored) if scored else None,
        tie_rate=(tie / scored) if scored else None,
    )


# ---------------------------------------------------------------------------
# Panel / file resolution (shared by every juried calculate stage)
# ---------------------------------------------------------------------------

def is_reasoning_juror(model: str) -> bool:
    """Qwen3 is a hybrid reasoning model. As a structured-output juror we want it
    to emit the verdict directly rather than a <think> block first."""
    m = str(model).lower()
    return m.startswith("qwen3") or "qwen3-" in m


def no_think(model: str, prompt: str) -> str:
    """Append Qwen3's `/no_think` soft switch for reasoning-model jurors so they
    answer immediately with no chain-of-thought.

    Content-level, so it flows through any API surface (Chat Completions or the
    Responses API) regardless of whether the vLLM build forwards
    chat_template_kwargs; a plain no-op for every non-reasoning model. Qwen3 is
    trained to honour this switch, and combined with guided JSON decoding it
    yields a clean, reasoning-free verdict. No effect on gemma / Llama / GPT.
    """
    return f"{prompt}\n\n/no_think" if is_reasoning_juror(model) else prompt


def panel_from_cfg(ev) -> list[str]:
    """Juror names from evaluation config. Empty/None -> [eval_model], so an
    unconfigured jury degenerates to the single-judge run."""
    j = ev.get("jury", None) if hasattr(ev, "get") else getattr(ev, "jury", None)
    if j:
        return [str(x) for x in j]
    return [str(ev.eval_model)]


def load_panel_raw(output_root, model: str, stem: str, panel: list[str]):
    """Load one juror's raw records per juror for one educator `model`.

    Returns dict: juror -> list[record]. Path is <model>/<stem>__<juror>.jsonl.
    Back-compat: a single-juror panel may use the legacy un-tagged <stem>.jsonl
    if the tagged file is absent (this is what makes the existing gemma run load
    as a 1-juror panel for validation). Missing files raise — no silent recovery.
    """
    import json
    from pathlib import Path

    out = {}
    for juror in panel:
        tagged = Path(output_root) / model / f"{stem}__{juror}.jsonl"
        legacy = Path(output_root) / model / f"{stem}.jsonl"
        if tagged.exists():
            path = tagged
        elif len(panel) == 1 and legacy.exists():
            path = legacy
        else:
            raise FileNotFoundError(
                f"Juror '{juror}' has no raw file for '{model}': expected {tagged}"
                + (f" (or legacy {legacy})" if len(panel) == 1 else "")
                + ". Run the per-juror raw stage first."
            )
        with open(path) as f:
            out[juror] = [json.loads(line) for line in f if line.strip()]
    return out


# ---------------------------------------------------------------------------
# Judge routing — DeepSeek has no strict json_schema mode
# ---------------------------------------------------------------------------

DEEPSEEK_SYS = (
    "Reply with ONLY a JSON object conforming to this JSON schema:\n{schema}\n\n"
    "CRITICAL: every value must match its schema type EXACTLY. For any field typed "
    '"string", output a single JSON string — NEVER an array or object. If multiple '
    'items apply, write them in one string separated by "; ".'
)


def is_deepseek(model) -> bool:
    return str(model).lower().startswith("deepseek")


def is_r1_distill(model) -> bool:
    """DeepSeek-R1 distills (e.g. DeepSeek-R1-Distill-Qwen-32B) are reasoning models
    served locally. They emit a <think>...</think> block before the answer and do NOT
    honour Qwen3's /no_think, so they judge via 'chat_reason' (reason freely, strip the
    think block, loose-validate the JSON) rather than guided decoding — which would force
    JSON from token 0 and throw away the reasoning the distill depends on."""
    m = str(model).lower()
    return "r1-distill" in m or ("deepseek" in m and "distill" in m)


# Judge-call robustness (see structured_call): a long client timeout so a slow-but-not-
# hung request is not aborted; a default output cap so a runaway generation on a bounded-
# output stage cannot hang the server for minutes; transient API errors are retried.
_JUDGE_TIMEOUT = 1800.0            # seconds
_DEFAULT_MAX_OUT = 8192            # cap for bounded stages (responses/chat_json); commission passes 16384 explicitly; chat_reason stays uncapped (reasoning needs room)
_TRANSIENT = (openai.APITimeoutError, openai.APIConnectionError,
              openai.InternalServerError, openai.RateLimitError)


def make_judge_client(eval_model: str, eval_port: int | None = None):
    """Build the (client, api_style) pair a raw judged stage should use.

    Any locally-served juror uses the local vLLM port: the DeepSeek-R1 distill goes
    through 'chat_reason' (reasoning model), every other local juror keeps guided
    decoding ('responses'). A DeepSeek model with NO local port is the hosted API,
    which has no json_schema mode and routes through 'chat_json'.
    """
    if eval_port:
        client = AsyncOpenAI(base_url=f"http://localhost:{eval_port}/v1", api_key="EMPTY",
                             timeout=_JUDGE_TIMEOUT)
        return client, ("chat_reason" if is_r1_distill(eval_model) else "responses")
    if is_deepseek(eval_model):
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise RuntimeError(
                f"DEEPSEEK_API_KEY is not set; required to judge with '{eval_model}'."
            )
        return AsyncOpenAI(base_url="https://api.deepseek.com/v1", api_key=key,
                           timeout=_JUDGE_TIMEOUT), "chat_json"
    return AsyncOpenAI(timeout=_JUDGE_TIMEOUT), "responses"


def _stringify_lists(obj):
    """Recursively join scalar-lists into '; '-separated strings. Safe here because
    none of the judge schemas have a legitimately list-typed field, so any list is
    a DeepSeek type slip on a string field."""
    if isinstance(obj, list):
        return "; ".join(str(_stringify_lists(x)) for x in obj)
    if isinstance(obj, dict):
        return {k: _stringify_lists(v) for k, v in obj.items()}
    return obj


def _validate_loose(schema, content):
    """Validate; on type-slip, coerce scalar-lists to strings and retry once.
    Returns (instance, coerced: bool). Raises if still invalid."""
    try:
        return schema.model_validate_json(content), False
    except ValidationError:
        data = _stringify_lists(json.loads(content))   # json.loads may raise -> caught upstream
        return schema.model_validate(data), True


def _strip_reasoning(text: str) -> str:
    """R1 reasoning models emit '<think>...</think>' before the answer, and the
    distills additionally wrap the JSON answer in a ```json ... ``` markdown fence.
    Strip both so only the JSON remains to validate — whether or not the server's
    reasoning parser already moved the think block out of `content`. Keep what
    follows the last '</think>'; if a '<think>' is left unclosed (truncated output),
    drop from it onward; then peel a surrounding code fence, and as a last resort
    slice the outermost {...} object out of any residual prose."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        text = text.split("<think>", 1)[0]
    text = text.strip()
    # distills fence the answer: ```json\n{...}\n``` — peel the fence
    text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    # last resort: residual prose around the object -> slice out the {...}
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            text = text[i:j + 1]
    return text.strip()


async def structured_call(client, model, api_style, prompt, schema, *,
                          temperature=0, max_output_tokens=None):
    """Run one structured judge call through the client's api_style.

    'responses'  : guided decoding via client.responses.parse (local vLLM jurors,
                   OpenAI). Qwen3's /no_think switch is applied here.
    'chat_json'  : DeepSeek has no strict json_schema mode, so this uses
                   chat.completions JSON mode + local Pydantic validation
                   (_validate_loose).

    Retries up to 3 attempts total on a schema-validation failure (ValidationError /
    json.JSONDecodeError) and returns None if it still fails — callers treat that
    exactly like the pre-existing context-overflow abstain path (skip this juror's
    vote for this unit, do not crash). Auth/connection errors are NOT retried here;
    they raise on first occurrence so failures stay visible.
    """
    for attempt in range(5):
        try:
            if api_style == "responses":
                p = no_think(model, prompt)
                out = await client.responses.parse(
                    model=model,
                    input=[{"role": "user", "content": p}],
                    text_format=schema,
                    temperature=temperature,
                    max_output_tokens=(max_output_tokens or _DEFAULT_MAX_OUT),
                )
                return out.output_parsed

            elif api_style == "chat_json":
                schema_json = json.dumps(schema.model_json_schema())
                out = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": DEEPSEEK_SYS.format(schema=schema_json)},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=(max_output_tokens or _DEFAULT_MAX_OUT),
                )
                content = out.choices[0].message.content or ""
                inst, _coerced = _validate_loose(schema, content)
                return inst

            elif api_style == "chat_reason":
                # Local R1 distill: let it reason, then parse the JSON after </think>.
                # No response_format (json_object would forbid the think block) and no
                # max_tokens cap (reasoning needs room — bounded by the served context).
                schema_json = json.dumps(schema.model_json_schema())
                out = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": DEEPSEEK_SYS.format(schema=schema_json)},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                content = _strip_reasoning(out.choices[0].message.content or "")
                inst, _c = _validate_loose(schema, content)
                return inst

            else:
                raise ValueError(f"unknown api_style {api_style!r}")

        except (ValidationError, json.JSONDecodeError):
            if attempt >= 2:                       # schema-invalid after 3 tries -> abstain
                return None
        except AttributeError as e:
            # vLLM's /v1/responses occasionally returns a degenerate response with
            # output=None; the SDK's parse_response then raises AttributeError on
            # `response.output`. This is a malformed-judge-output failure (same class
            # as a schema-validation miss), NOT an auth/connection error — abstain on
            # this unit (no vote, printed) instead of crashing the whole eval.
            # Deterministic at temp=0, so one retry then give up.
            # (Qwen3-4B variance-eval arm, job 63527871_2, 2026-08-23.)
            if attempt >= 1:
                print(f"  [ABSTAIN] {model} null/degenerate structured response "
                      f"({type(e).__name__}) — skipping this unit's vote.")
                return None
        except _TRANSIENT as e:
            # Transient server/API error (timeout, connection, 5xx, rate-limit): a long
            # API-bound run WILL hit these; retry with backoff rather than crash the whole
            # stage, and abstain on this unit only if it never recovers.
            if attempt >= 4:
                print(f"  [ABSTAIN] {model} transient API error ({type(e).__name__}) "
                      f"after retries — skipping this juror's vote for this unit.")
                return None
            await asyncio.sleep(min(2 ** attempt, 30))
    return None


# ---------------------------------------------------------------------------
# Self-test — run:  python src/evaluation/jury.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- N=1 identity: a one-juror panel must return that juror's label ---
    assert vote(["yes"]).label == "yes"
    assert vote(["harmful"]).label == "harmful"
    assert mean_pool([4]) == 4.0
    print("[ok] N=1 degenerates to the single judge")

    # --- binary majority (odd panel, no ties possible) ---
    r = vote(["yes", "yes", "no"])
    assert r.label == "yes" and r.is_majority and not r.is_tie and not r.is_unanimous
    r = vote(["no", "no", "no"])
    assert r.label == "no" and r.is_unanimous
    print("[ok] binary majority + unanimity")

    # --- 3-way 1-1-1 tie is flagged, not silently scored ---
    r = vote(["entailed", "contradicted", "neutral"])
    assert r.is_tie and not r.is_majority
    # severity tie-break picks the most severe
    lab, res = resolve(["appropriate", "questionable", "harmful"],
                       tie_break=["harmful", "questionable", "appropriate"])
    assert lab == "harmful" and res.is_tie
    # primary-juror tie-break falls back to juror 0's label
    lab, res = resolve(["neutral", "entailed", "contradicted"], primary_index=0)
    assert lab == "neutral"
    print("[ok] 3-way ties flagged + resolved by explicit policy")

    # --- abstentions dropped; empty panel -> None ---
    assert vote(["yes", None, "yes"]).label == "yes"
    assert vote([None, None]).label is None
    assert mean_pool([None, 3, 5]) == 4.0
    assert mean_pool([None, None]) is None
    print("[ok] abstentions dropped, empty -> None")

    # --- agreement: perfect vs. none ---
    perfect = agreement([["a", "a", "a"], ["b", "b", "b"]])
    assert perfect.percent_agreement == 1.0 and perfect.unanimous_rate == 1.0
    mixed = agreement([["a", "a", "b"], ["a", "b", "b"], ["a", "a", "a"]],
                      categories=["a", "b"])
    assert mixed.fleiss_kappa is not None
    print(f"[ok] agreement: perfect pa={perfect.percent_agreement}, "
          f"mixed kappa={mixed.fleiss_kappa:.3f}")

    # --- reasoning-juror /no_think switch: Qwen only, no-op elsewhere ---
    assert no_think("Qwen3-32B", "P").endswith("/no_think")
    assert no_think("gemma-4-31B-it", "P") == "P"
    assert no_think("Llama-3.3-70B-Instruct", "P") == "P"
    assert is_reasoning_juror("Qwen3-32B") and not is_reasoning_juror("gemma-4-31B-it")
    print("[ok] Qwen3 /no_think switch appended; no-op for gemma/Llama")

    # --- DeepSeek routing: string-prefix check only, no network ---
    assert is_deepseek("deepseek-v4-flash")
    assert not is_deepseek("gemma-4-31B-it") and not is_deepseek("Qwen3-32B")
    print("[ok] is_deepseek routing")

    print("\nAll jury.py self-tests passed.")
