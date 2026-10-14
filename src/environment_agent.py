"""
environment_agent.py

Environment Agent for clinical simulation oversight.
Prompt templates are loaded at startup from the file named by prompts.environment in
config/config.yaml.

Retry strategy: Option 3 -- bad turns are discarded entirely and never enter
conversation history. The correction instruction is injected into the agent's
system prompt before regeneration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader

from utils import CONFIG, PROMPTS_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATE_DIR  = PROMPTS_PATH
_TEMPLATE_FILE = CONFIG["prompts"]["environment"]

_jinja_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
_macros = _jinja_env.get_template(_TEMPLATE_FILE).module


def _render_macro(macro_name: str, **variables: Any) -> str:
    return getattr(_macros, macro_name)(**variables)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    PATIENT  = "patient"
    EDUCATOR = "educator"


class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class IssueType(str, Enum):
    ROLE_DRIFT        = "role_drift"
    HALLUCINATION     = "hallucination"
    REPETITION        = "repetition"
    DERAILMENT        = "derailment"
    INCOHERENCE       = "incoherence"
    PREMATURE_END     = "premature_end"
    PROMPT_UNFAITHFUL = "prompt_unfaithful"
    NONE              = "none"


class ActionType(str, Enum):
    NONE            = "none"
    SOFT_CORRECTION = "soft_correction"
    RETRY           = "retry_from_last_turn"
    HARD_STOP       = "hard_stop"


class InterventionLevel(str, Enum):
    """
    Controls how aggressively the environment agent intervenes per agent role.

    MODERATE - regulate the patient to keep realism intact, but bias toward
               letting the simulation continue:
                 WARN or FAIL+minor/moderate -> soft correction only
                 FAIL+severe                 -> retry; hard-stop only after
                                               all retries are exhausted
               Hard-stop is reserved for persistent severe failures such as
               the patient repeatedly claiming to be an educator, refusing to
               engage entirely, or producing fully incoherent output across
               every retry attempt.

    OBSERVE  - evaluate and log every turn but never intervene.
               Used for the educator: if an educator turn goes astray that
               is the educator's fault and the simulation carries on.

    STRICT   - follow the evaluator's recommendation exactly, including
               immediate HARD_STOP on a single severe FAIL (not used by
               default; available for research / debugging).
    """
    STRICT   = "strict"
    MODERATE = "moderate"
    OBSERVE  = "observe"


class PromptFaithfulness(str, Enum):
    FAITHFUL   = "faithful"
    PARTIAL    = "partial"
    UNFAITHFUL = "unfaithful"
    NA         = "n/a"


@dataclass
class EvaluationResult:
    verdict:                Verdict
    issue:                  IssueType
    severity:               str | None
    action:                 ActionType
    correction_instruction: str | None
    reasoning:              str
    prompt_faithfulness:    PromptFaithfulness = PromptFaithfulness.NA
    faithfulness_note:      str | None = None
    session_complete:       bool = False
    raw:                    dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    turn_index:         int
    role:               AgentRole
    message:            str
    agent_prompt:       str | None = None
    evaluation:         EvaluationResult | None = None
    was_corrected:      bool = False
    correction_applied: str | None = None


@dataclass
class DiscardedTurn:
    """Audit record for a turn rejected and discarded via retry."""
    turn_index_attempted: int
    retry_number:         int
    role:                 AgentRole
    message:              str
    agent_prompt:         str | None
    evaluation:           EvaluationResult


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class EvalResponse(BaseModel):
    verdict:                Verdict
    issue:                  IssueType
    severity:               str | None = None
    action:                 ActionType
    correction_instruction: str | None = None
    prompt_faithfulness:    PromptFaithfulness = PromptFaithfulness.NA
    faithfulness_note:      str | None = None
    reasoning:              str
    session_complete:       bool = False


# ---------------------------------------------------------------------------
# Environment Agent
# ---------------------------------------------------------------------------

class EnvironmentAgent:
    """
    Monitors a patient-educator discharge-education simulation turn by turn.

    Intervention levels are fixed by design and not configurable:
      Patient  -> MODERATE : regulate patient realism but bias toward keeping
                             the simulation alive; hard-stop only when the
                             patient repeatedly fails at a severe level.
      Educator -> OBSERVE  : log and evaluate every educator turn but never
                             block it; educator faults stay in the record and
                             the simulation continues regardless.

    Parameters
    ----------
    eval_model : str
        Model used by the environment agent for evaluation.
    max_retries : int
        Max discarded attempts per turn before escalating to HARD_STOP.
    min_turns_before_complete : int
        A session_complete verdict is ignored until the conversation has this many
        turns, so the judge cannot end the session after only an opening exchange.
    """

    PATIENT_LEVEL:  InterventionLevel = InterventionLevel.MODERATE
    EDUCATOR_LEVEL: InterventionLevel = InterventionLevel.OBSERVE

    _SYSTEM_PROMPT: str = _render_macro("eval_system")

    def __init__(
        self,
        eval_model:  str = "gpt-5-nano",
        max_retries: int = 2,
        port:        int | None = None,
        host:        str = "localhost",
        min_turns_before_complete: int = 10,
    ) -> None:
        self.eval_model  = eval_model
        self.max_retries = max_retries
        self.MIN_TURNS_BEFORE_COMPLETE = min_turns_before_complete
        if port is not None:
            self._client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")
        else:
            self._client = OpenAI()

        self._history:           list[ConversationTurn]     = []
        self._accepted:          list[ConversationTurn]     = []
        self._discarded:         list[DiscardedTurn]        = []
        self._retry_counts:      dict[int, int]             = {}
        self._stopped:           bool       = False
        self._stop_reason:       str | None = None
        self._session_complete:  bool       = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ConversationTurn]:
        return list(self._history)

    @property
    def accepted(self) -> list[ConversationTurn]:
        """Clean history: only turns that passed without any correction."""
        return list(self._accepted)

    @property
    def discarded(self) -> list[DiscardedTurn]:
        return list(self._discarded)

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def session_complete(self) -> bool:
        return self._session_complete

    def intervention_level_for(self, role: AgentRole) -> InterventionLevel:
        return (
            self.PATIENT_LEVEL
            if role == AgentRole.PATIENT
            else self.EDUCATOR_LEVEL
        )

    def submit_turn(
        self,
        role:         AgentRole,
        message:      str,
        agent_prompt: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Submit a conversation turn for evaluation.

        Parameters
        ----------
        role         : which agent produced this turn
        message      : the agent's output text
        agent_prompt : the system prompt that generated this output;
                       enables prompt-faithfulness checking when provided

        Returns
        -------
        (True,  None) -- accepted, no issues
        (True,  str)  -- accepted with soft correction; append str to agent's
                         system prompt for its NEXT turn, then discard it
        (False, str)  -- DISCARDED (retry); append str to agent's system prompt
                         and regenerate; bad turn never enters history
        (False, None) -- simulation hard-stopped
        """
        if self._stopped:
            raise RuntimeError(f"Simulation has been stopped: {self._stop_reason}")

        turn_index = len(self._history)

        eval_result = self._evaluate_turn(role, message, agent_prompt)
        level       = self.intervention_level_for(role)
        action      = self._resolve_action(eval_result, level, turn_index)

        if action == ActionType.NONE:
            turn = ConversationTurn(
                turn_index=turn_index, role=role, message=message,
                agent_prompt=agent_prompt, evaluation=eval_result,
            )
            self._history.append(turn)
            self._accepted.append(turn)
            if eval_result.session_complete:
                if len(self._history) >= self.MIN_TURNS_BEFORE_COMPLETE:
                    self._session_complete = True
                    logger.info("[env] Turn %d signals natural session end", turn_index)
                else:
                    logger.info(
                        "[env] Turn %d session_complete=true ignored (%d turns < MIN_TURNS_BEFORE_COMPLETE=%d)",
                        turn_index, len(self._history), self.MIN_TURNS_BEFORE_COMPLETE,
                    )
            logger.debug("[env] Turn %d PASS (%s)", turn_index, role.value)
            return True, None

        if action == ActionType.SOFT_CORRECTION:
            snippet = _render_macro(
                "soft_correction",
                issue=eval_result.issue.value,
                correction_instruction=eval_result.correction_instruction or "",
            )
            self._history.append(ConversationTurn(
                turn_index=turn_index, role=role, message=message,
                agent_prompt=agent_prompt, evaluation=eval_result,
                was_corrected=True,
                correction_applied=eval_result.correction_instruction,
            ))
            if eval_result.session_complete:
                if len(self._history) >= self.MIN_TURNS_BEFORE_COMPLETE:
                    self._session_complete = True
                    logger.info("[env] Turn %d signals natural session end", turn_index)
                else:
                    logger.info(
                        "[env] Turn %d session_complete=true ignored (%d turns < MIN_TURNS_BEFORE_COMPLETE=%d)",
                        turn_index, len(self._history), self.MIN_TURNS_BEFORE_COMPLETE,
                    )
            logger.info(
                "[env] Turn %d SOFT_CORRECTION (%s): %s",
                turn_index, role.value, eval_result.issue.value,
            )
            return True, snippet

        if action == ActionType.RETRY:
            retry_count = self._retry_counts.get(turn_index, 0)
            self._discarded.append(DiscardedTurn(
                turn_index_attempted=turn_index,
                retry_number=retry_count + 1,
                role=role, message=message,
                agent_prompt=agent_prompt, evaluation=eval_result,
            ))
            if retry_count >= self.max_retries:
                logger.warning("[env] Turn %d exhausted retries -> HARD_STOP", turn_index)
                self._hard_stop(f"Max retries exceeded at turn {turn_index}")
                return False, None
            self._retry_counts[turn_index] = retry_count + 1
            logger.info(
                "[env] Turn %d DISCARD + RETRY %d/%d (%s): %s",
                turn_index, retry_count + 1, self.max_retries,
                role.value, eval_result.reasoning,
            )
            return False, eval_result.correction_instruction

        if action == ActionType.HARD_STOP:
            self._hard_stop(f"Turn {turn_index} ({role.value}): {eval_result.reasoning}")
            return False, None

        # Fallback
        self._history.append(ConversationTurn(
            turn_index=turn_index, role=role, message=message,
            agent_prompt=agent_prompt, evaluation=eval_result,
        ))
        return True, None

    def conversation_log(self) -> dict[str, Any]:
        """JSON-serialisable log with 'accepted' and 'discarded' sections."""
        accepted = []
        for t in self._history:
            entry: dict[str, Any] = {
                "turn_index":         t.turn_index,
                "role":               t.role.value,
                "message":            t.message,
                "was_corrected":      t.was_corrected,
                "correction_applied": t.correction_applied,
            }
            if t.evaluation:
                e = t.evaluation
                entry["evaluation"] = {
                    "verdict":             e.verdict.value,
                    "issue":               e.issue.value,
                    "severity":            e.severity,
                    "action":              e.action.value,
                    "prompt_faithfulness": e.prompt_faithfulness.value,
                    "faithfulness_note":   e.faithfulness_note,
                    "reasoning":           e.reasoning,
                }
            accepted.append(entry)

        discarded = []
        for d in self._discarded:
            e = d.evaluation
            discarded.append({
                "turn_index_attempted": d.turn_index_attempted,
                "retry_number":         d.retry_number,
                "role":                 d.role.value,
                "message":              d.message,
                "evaluation": {
                    "verdict":                e.verdict.value,
                    "issue":                  e.issue.value,
                    "severity":               e.severity,
                    "correction_instruction": e.correction_instruction,
                    "reasoning":              e.reasoning,
                },
            })

        return {"accepted": accepted, "discarded": discarded}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_turn(
        self,
        role:         AgentRole,
        message:      str,
        agent_prompt: str | None,
    ) -> EvaluationResult:
        user_prompt = _render_macro(
            "eval_user",
            history=self._history,
            current_role=role.value,
            current_message=message,
            agent_prompt=agent_prompt,
        )
        inputs = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            response = self._client.responses.parse(
                model=self.eval_model,
                input=inputs,
                text_format=EvalResponse,
            )
            data = response.output_parsed
            return EvaluationResult(
                verdict=data.verdict,
                issue=data.issue,
                severity=data.severity,
                action=data.action,
                correction_instruction=data.correction_instruction,
                reasoning=data.reasoning,
                prompt_faithfulness=data.prompt_faithfulness,
                faithfulness_note=data.faithfulness_note,
                session_complete=data.session_complete,
                raw=data.model_dump(),
            )
        except Exception:
            logger.error("[env] Structured parse failed for turn (%s)", role.value)
            return EvaluationResult(
                verdict=Verdict.WARN,
                issue=IssueType.INCOHERENCE,
                severity="minor",
                action=ActionType.SOFT_CORRECTION,
                correction_instruction="Please continue the conversation naturally.",
                reasoning="Environment agent received unparseable evaluation output.",
            )

    def _resolve_action(
        self,
        result:     EvaluationResult,
        level:      InterventionLevel,
        turn_index: int,
    ) -> ActionType:
        """
        Map the evaluator's recommended action to the effective action given
        the configured intervention level.

        OBSERVE
          Always NONE -- log only, never block.

        MODERATE  (patient)
          PASS                      -> NONE
          WARN                      -> SOFT_CORRECTION
          FAIL + minor              -> SOFT_CORRECTION
          FAIL + moderate           -> SOFT_CORRECTION  (nudge; do not discard)
          FAIL + severe             -> RETRY; HARD_STOP only after max_retries

          Severe failures warranting HARD_STOP after retries are exhausted:
            - patient persistently claims to be an educator / medical expert
            - patient refuses to engage across all retry attempts
            - patient output is fully incoherent across all retry attempts
          Everything else -- role drift, minor hallucination, repetition --
          is handled with a soft correction and the simulation continues.

        STRICT
          Follows the evaluator's recommendation exactly (not used by default).
        """
        if level == InterventionLevel.OBSERVE:
            return ActionType.NONE

        if result.verdict == Verdict.PASS:
            return ActionType.NONE

        if level == InterventionLevel.MODERATE:
            if result.verdict == Verdict.WARN:
                return ActionType.SOFT_CORRECTION
            # FAIL path -- soft-correct anything below severe
            if result.severity in ("minor", "moderate"):
                return ActionType.SOFT_CORRECTION
            # severe FAIL -- retry first; hard-stop only after retries exhausted
            retries = self._retry_counts.get(turn_index, 0)
            return (
                ActionType.HARD_STOP
                if retries >= self.max_retries
                else ActionType.RETRY
            )

        # STRICT -- follow evaluator recommendation exactly
        return result.action

    def _hard_stop(self, reason: str) -> None:
        self._stopped     = True
        self._stop_reason = reason
        logger.error("[env] HARD STOP -- %s", reason)
