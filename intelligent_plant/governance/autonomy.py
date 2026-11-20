"""Autonomy gates: what the swarm may execute on its own and what needs a
named human (Table 8.2, Chapter 13.4, Chapter 16.3).

Decision rule: automate where the cost of an error is low, the signal is
unambiguous and the loop is fast; keep humans where the reverse holds."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any


class Autonomy(IntEnum):
    AUTOMATE = 6                 # execute, log
    AUTOMATE_WITH_ALARMS = 5     # execute, raise alarm on anomaly
    AUTOMATE_SUPERVISED = 4      # execute, notify supervisor who may reverse
    RECOMMEND = 3                # queue; human approves before execution
    TRIAGE = 2                   # queue; human confirms the finding
    DECISION_SUPPORT = 1         # information only; human decides and acts
    HUMAN_ONLY = 0               # the system may not even propose an action

    @property
    def executes_without_human(self) -> bool:
        return self >= Autonomy.AUTOMATE_SUPERVISED


@dataclass(frozen=True)
class TaskPolicy:
    autonomy: Autonomy
    owner_role: str              # named accountability (Chapter 16.3)
    error_cost: str
    signal_quality: str
    source: str


TASKS: dict[str, TaskPolicy] = {
    # Table 8.2 - cultivation
    "data_logging": TaskPolicy(Autonomy.AUTOMATE, "data_steward", "low", "high", "Table 8.2"),
    "alerting": TaskPolicy(Autonomy.AUTOMATE, "data_steward", "low", "high", "Table 8.2"),
    "climate_setpoint": TaskPolicy(Autonomy.AUTOMATE_WITH_ALARMS, "head_grower", "low-medium", "high", "Table 8.2"),
    "irrigation_schedule": TaskPolicy(Autonomy.AUTOMATE_SUPERVISED, "head_grower", "medium", "medium-high", "Table 8.2"),
    "nutrient_recipe": TaskPolicy(Autonomy.RECOMMEND, "head_grower", "medium", "medium", "Table 8.2"),
    "disease_identification": TaskPolicy(Autonomy.TRIAGE, "head_grower", "medium", "medium", "Table 8.2"),
    "harvest_timing": TaskPolicy(Autonomy.DECISION_SUPPORT, "head_grower", "high", "medium", "Table 8.2"),
    "crop_destruction": TaskPolicy(Autonomy.HUMAN_ONLY, "facility_manager", "very high", "varies", "Table 8.2"),
    "sensor_failsafe": TaskPolicy(Autonomy.AUTOMATE_WITH_ALARMS, "facility_engineer", "low", "high", "Ch 16.3"),
    "calibration_schedule": TaskPolicy(Autonomy.AUTOMATE, "facility_engineer", "low", "high", "Ch 8.1"),
    "lighting_energy": TaskPolicy(Autonomy.AUTOMATE_SUPERVISED, "facility_engineer", "low-medium", "high", "Ch 10"),
    # Breeding
    "breeding_selection": TaskPolicy(Autonomy.RECOMMEND, "breeder", "high", "medium", "Ch 9"),
    "sex_cull_by_marker": TaskPolicy(Autonomy.RECOMMEND, "breeder", "medium", "high", "Ch 9.1"),
    # Processing / QC
    "decarb_plan": TaskPolicy(Autonomy.RECOMMEND, "process_lead", "medium", "medium", "Ch 3.3"),
    "drying_environment": TaskPolicy(Autonomy.AUTOMATE_WITH_ALARMS, "process_lead", "low-medium", "high", "Ch 11.1"),
    "coa_audit": TaskPolicy(Autonomy.AUTOMATE, "qa_manager", "low", "high", "Ch 3.4"),
    "batch_hold": TaskPolicy(Autonomy.AUTOMATE_SUPERVISED, "qa_manager", "low", "high", "Ch 11.3"),
    "batch_release": TaskPolicy(Autonomy.RECOMMEND, "qa_manager", "high", "medium", "Ch 11"),
    "spc_monitoring": TaskPolicy(Autonomy.AUTOMATE, "qa_manager", "low", "high", "Ch 11.3"),
    # Compliance
    "inventory_reconciliation": TaskPolicy(Autonomy.AUTOMATE_WITH_ALARMS, "compliance_officer", "low", "high", "Ch 12.1"),
    "anomaly_investigation": TaskPolicy(Autonomy.TRIAGE, "compliance_officer", "medium", "medium", "Ch 12.1"),
    "regulatory_monitoring": TaskPolicy(Autonomy.AUTOMATE, "compliance_officer", "low", "high", "Ch 12.3"),
    "regulatory_interpretation": TaskPolicy(Autonomy.DECISION_SUPPORT, "counsel", "very high", "low", "Ch 12.3"),
    "hemp_compliance_check": TaskPolicy(Autonomy.AUTOMATE_SUPERVISED, "compliance_officer", "medium", "high", "Ch 14.2"),
    # Clinical (Chapter 13.4: the tool proposes; the clinician disposes)
    "patient_log_capture": TaskPolicy(Autonomy.AUTOMATE, "data_steward", "low", "high", "Table 13.2"),
    "patient_reminder": TaskPolicy(Autonomy.AUTOMATE, "care_coordinator", "low", "high", "Ch 13.2"),
    "patient_progress_summary": TaskPolicy(Autonomy.AUTOMATE, "care_coordinator", "low", "high", "Ch 13.3"),
    "interaction_alert": TaskPolicy(Autonomy.AUTOMATE, "pharmacist", "low", "high", "Ch 13.4 hard rules"),
    "safety_alert": TaskPolicy(Autonomy.AUTOMATE, "treating_clinician", "low", "high", "Ch 13.2"),
    "consultation_report": TaskPolicy(Autonomy.DECISION_SUPPORT, "treating_clinician", "high", "medium", "Ch 13.1"),
    "product_recommendation": TaskPolicy(Autonomy.RECOMMEND, "treating_clinician", "high", "medium", "Ch 13.1"),
    "titration_step": TaskPolicy(Autonomy.RECOMMEND, "treating_clinician", "high", "medium", "Table 13.1"),
    "dose_selection": TaskPolicy(Autonomy.HUMAN_ONLY, "treating_clinician", "very high", "low", "Ch 13.2"),
    "continue_or_stop_review": TaskPolicy(Autonomy.DECISION_SUPPORT, "treating_clinician", "high", "medium", "Table 13.1"),
    "n_of_1_enrolment": TaskPolicy(Autonomy.RECOMMEND, "treating_clinician", "medium", "high", "Ch 5.4"),
    # Enterprise
    "model_deployment": TaskPolicy(Autonomy.RECOMMEND, "model_owner", "high", "medium", "Appendix C.5"),
    "model_rollback": TaskPolicy(Autonomy.AUTOMATE_SUPERVISED, "model_owner", "low", "high", "Ch 16.2"),
    "breeding_objective_update": TaskPolicy(Autonomy.RECOMMEND, "breeder", "high", "low", "Ch 16.1"),
}


def policy_for(task: str) -> TaskPolicy:
    # Unknown tasks get the most conservative treatment that still lets a human see them.
    return TASKS.get(task, TaskPolicy(Autonomy.DECISION_SUPPORT, "operations_manager", "unknown", "unknown",
                                      "default-conservative"))


@dataclass
class ApprovalRequest:
    proposal_id: str
    task: str
    owner_role: str
    summary: str
    payload: dict
    evidence: list[str]
    objections: list[str]
    status: str = "pending"            # pending | approved | rejected | acknowledged
    decided_by: str | None = None
    note: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class ApprovalQueue:
    """Human authority. Overrides are recorded with the outcome so they feed
    back into evaluation (Chapter 16.3: 'override is easy and respected')."""

    def __init__(self) -> None:
        self._items: dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()
        self.overrides: list[dict[str, Any]] = []

    def submit(self, req: ApprovalRequest) -> ApprovalRequest:
        with self._lock:
            self._items[req.request_id] = req
        return req

    def pending(self, role: str | None = None) -> list[ApprovalRequest]:
        return [r for r in self._items.values() if r.status == "pending" and (role is None or r.owner_role == role)]

    def decide(self, request_id: str, approve: bool, by: str, role: str, note: str = "") -> ApprovalRequest:
        r = self._items[request_id]
        if role != r.owner_role:
            raise PermissionError(f"{role} is not the accountable owner ({r.owner_role}) for {r.task}")
        r.status = "approved" if approve else "rejected"
        r.decided_by, r.note = by, note
        if not approve:
            self.overrides.append({"request_id": request_id, "task": r.task, "by": by, "note": note})
        return r

    def all(self) -> list[ApprovalRequest]:
        return list(self._items.values())
