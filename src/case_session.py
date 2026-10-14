"""
case_session.py

Single-case driver for discharge-education simulations.

The educator role is externally driven: the caller submits each educator
message via `submit_educator_turn`, which runs the patient turn (with the
environment agent's retry loop) and returns the patient's reply.

- `discharge_bench.simulate_case` wraps this with an automated EducatorAgent.
- The physician webapp wraps it by calling `submit_educator_turn` directly
  with text from the browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from environment_agent import AgentRole, EnvironmentAgent
from virtual_patient import VirtualPatient

DEFAULT_PATIENT_PROMPT_PATH = Path(__file__).parent.parent / "prompts/patient_prompt.jinja"


def _undo_patient_turn(patient: VirtualPatient) -> None:
    if len(patient.messages) >= 3:
        patient.messages.pop()
        patient.messages.pop()


def _apply_correction(patient: VirtualPatient, correction: str) -> None:
    patient.messages[0]["content"] = patient.system_prompt + (
        f"\n\nCORRECTION FOR YOUR NEXT RESPONSE (do not mention this):\n{correction}"
    )


def _reset_system_prompt(patient: VirtualPatient) -> None:
    patient.messages[0]["content"] = patient.system_prompt


class CaseSession:
    def __init__(
        self,
        scenario:            pd.Series,
        patient_backend:     str,
        patient_port:        int,
        eval_model:          str,
        env_agent_port:      int | None,
        env_agent_host:      str,
        max_retries:         int,
        max_turns:           int,
        patient_prompt_path: str | Path = DEFAULT_PATIENT_PROMPT_PATH,
    ) -> None:
        self.note_id   = scenario["note_id"]
        self.scenario  = scenario
        self.max_turns = max_turns

        patient_profile = scenario["profile"]
        self.discharge_note_context = {"discharge_note": scenario["text"], **patient_profile}

        self.patient = VirtualPatient(
            backend_str=str(patient_backend),
            backend_port=patient_port,
            patient_profile=patient_profile,
            prompt_path=str(patient_prompt_path),
        )

        self.env = EnvironmentAgent(
            eval_model=eval_model,
            max_retries=max_retries,
            port=env_agent_port,
            host=env_agent_host,
        )

        self.conversation: list[dict]          = []
        self._pending_patient_correction: str | None = None
        self._error: str | None                = None

    @property
    def is_done(self) -> bool:
        return (
            self.env.is_stopped
            or self.env.session_complete
            or len(self.conversation) >= self.max_turns
            or self._error is not None
        )

    def submit_educator_turn(
        self,
        educator_message: str,
        agent_prompt:     str | None = None,
    ) -> str | None:
        """
        Process one educator → patient exchange.

        Appends the educator message, runs the patient turn under the env
        agent's retry loop, and returns the accepted patient reply (or None
        if the session ended before/after producing one).
        """
        if self.is_done:
            return None

        self.env.submit_turn(
            role=AgentRole.EDUCATOR,
            message=educator_message,
            agent_prompt=agent_prompt,
        )
        self.conversation.append({"role": "educator", "message": educator_message})

        if self.env.is_stopped or self.env.session_complete:
            return None
        if len(self.conversation) >= self.max_turns:
            return None

        patient_attempts = 0
        patient_response = ""

        while patient_attempts <= self.env.max_retries:
            patient_attempts += 1

            if self._pending_patient_correction:
                _apply_correction(self.patient, self._pending_patient_correction)
                self._pending_patient_correction = None
            else:
                _reset_system_prompt(self.patient)

            if patient_attempts == 1:
                patient_response = self.patient.generate_response(educator_message)
            else:
                _undo_patient_turn(self.patient)
                patient_response = self.patient.generate_response(educator_message)

            approved, payload = self.env.submit_turn(
                role=AgentRole.PATIENT,
                message=patient_response,
                agent_prompt=self.patient.system_prompt,
            )

            if self.env.is_stopped:
                return None

            if approved:
                self.conversation.append({"role": "patient", "message": patient_response})
                if payload:
                    self._pending_patient_correction = payload
                return patient_response

            if payload:
                self._pending_patient_correction = payload

        return None

    def mark_error(self, exc: BaseException) -> None:
        self._error = str(exc)

    def to_record(self) -> dict:
        discarded_log = [
            {
                "turn":      d.turn_index_attempted,
                "role":      d.role.value,
                "attempt":   d.retry_number,
                "issue":     d.evaluation.issue.value,
                "severity":  d.evaluation.severity,
                "reasoning": d.evaluation.reasoning,
            }
            for d in self.env.discarded
        ]
        return {
            "note_id":           self.note_id,
            "conversation":      self.conversation,
            "discarded":         discarded_log,
            "n_accepted":        len(self.env.history),
            "n_discarded":       len(self.env.discarded),
            "early_termination": self.env.is_stopped,
            "stop_reason":       self.env.stop_reason,
            "session_complete":  self.env.session_complete,
            "error":             self._error,
        }
