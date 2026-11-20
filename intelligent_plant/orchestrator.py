"""The Intelligent Cannabis Enterprise: one closed loop from genotype to
patient outcome and back (Chapter 16, Figure 16.1)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from .agents.base import Context, Proposal
from .agents.breeding import breeding_agents
from .agents.clinical import TitrationPlan, clinical_agents
from .agents.compliance import compliance_agents
from .agents.cultivation import cultivation_agents
from .agents.enterprise import enterprise_agents
from .agents.learning import learning_agents
from .agents.llm import ClaudeReasoner
from .agents.processing import processing_agents
from .agents.swarm import CycleReport, Swarm, default_critics
from .calibration.facility import load_calibration
from .data.privacy import PseudonymVault, key_from_env
from .patient.communication import PatientMessenger
from .data.store import Provenance, RecordStore
from .governance.autonomy import ApprovalQueue, ApprovalRequest
from .governance.readiness import ReadinessReport, check_readiness, default_artifacts
from .governance.registry import ModelRegistry
from .science.chemistry import VolatileLossModel
from .science.environment import RoomModel
from .knowledge.regulation import load_jurisdictions

log = logging.getLogger(__name__)


@dataclass
class FacilityState:
    """What the actuators touch. In production these calls go to the BMS /
    fertigation / ERP / messaging integrations."""
    setpoints: dict[str, dict] = field(default_factory=dict)
    batch_status: dict[str, str] = field(default_factory=dict)
    outbox: list[dict] = field(default_factory=list)
    work_orders: list[dict] = field(default_factory=list)
    irrigation_pulses: dict[str, int] = field(default_factory=dict)
    lighting: dict[str, float] = field(default_factory=dict)
    quarantined_products: set[str] = field(default_factory=set)


class ProductionNotReady(RuntimeError):
    def __init__(self, problems: list[str], report: ReadinessReport | None = None):
        self.problems, self.report = problems, report
        detail = "\n".join(f"  - {p}" for p in problems)
        super().__init__("Refusing to start in production mode:\n" + detail
                         + (("\n" + report.summary()) if report else ""))


class IntelligentEnterprise:
    """mode="demo" (default): in-memory, synthetic, nothing touches real patients.
    mode="production": refuses to start unless storage is persistent, the vault key comes from a secrets
    manager, a facility calibration is loaded, and every human review in governance/readiness.py is signed
    against the exact artifacts in use. The checks are deliberately not overridable by a flag."""

    def __init__(self, db_path: str = ":memory:", vault_path: str = ":memory:", use_llm: bool | None = None,
                 jury: bool = True, config_path: str | None = None, mode: str = "demo",
                 vault_key: bytes | None = None, calibration_path: str | None = None,
                 signoffs_path: str | None = None, regulatory_review_path: str | None = None,
                 phi_to_llm_permitted: bool = False):
        if mode not in ("demo", "production"):
            raise ValueError("mode must be 'demo' or 'production'")
        self.mode = mode
        if mode == "production":
            self.readiness = self._production_checks(db_path, vault_path, vault_key, calibration_path,
                                                     signoffs_path, regulatory_review_path)
            vault_key = vault_key or key_from_env()
        self.store = RecordStore(db_path)
        self.vault = PseudonymVault(vault_path, key=vault_key)
        self.approvals = ApprovalQueue(self.store)
        self.registry = ModelRegistry()
        self.config = load_jurisdictions(config_path)
        # Pseudonymised data is still PHI: it goes to an outside model only under a signed BAA.
        self.config["privacy"] = {"phi_to_llm_permitted": phi_to_llm_permitted}
        self.llm = ClaudeReasoner(enabled=use_llm)
        self.state = FacilityState()
        self.messenger = PatientMessenger()
        self.patients: dict[str, dict] = {}       # live clinical state keyed by pseudonym
        self.learning_loops: dict = {}              # model_id -> ChampionChallengerLoop (self-iterating)
        agents = (cultivation_agents() + breeding_agents() + processing_agents() + compliance_agents()
                  + clinical_agents() + enterprise_agents() + learning_agents())
        self.swarm = Swarm(agents, default_critics(with_jury=jury and self.llm.available), actuators=self._actuators())
        self.history: list[CycleReport] = []
        self.calibration: dict | None = None
        if calibration_path:
            self.apply_calibration(load_calibration(calibration_path))

    @staticmethod
    def _production_checks(db_path, vault_path, vault_key, calibration_path, signoffs_path,
                           regulatory_review_path) -> ReadinessReport:
        problems = []
        if db_path == ":memory:" or vault_path == ":memory:":
            problems.append("Record store and vault must be persistent files, not in memory.")
        elif db_path == vault_path:
            problems.append("The vault (identities) must be a separate store from the records.")
        if vault_key is None:
            try:
                key_from_env()
            except ValueError as e:
                problems.append(str(e))
        if not calibration_path:
            problems.append("No facility calibration: run `intelligent-plant calibrate` on this facility's data.")
        if not signoffs_path:
            problems.append("No sign-off file given.")
            raise ProductionNotReady(problems)
        artifacts = default_artifacts(calibration_path, regulatory_review_path)
        report = check_readiness(signoffs_path, artifacts)
        if not report.ready:
            problems.append("Required human reviews are missing or stale.")
        if problems:
            raise ProductionNotReady(problems, report)
        return report

    def apply_calibration(self, cal: dict) -> None:
        """Replace illustrative defaults with this facility's fitted parameters."""
        self.calibration = cal
        for a in self.swarm.agents:
            if a.name == "climate_controller" and cal.get("room_model"):
                a.mpc.model = RoomModel(**cal["room_model"])
            elif a.name == "decarb_planner" and cal.get("volatile_loss"):
                v = cal["volatile_loss"]
                a.volatile, a.calibrated = VolatileLossModel(A=v["A"], Ea_J=v["Ea_J"]), True
            elif a.name == "nutrients" and cal.get("nutrient_ranges"):
                for stage, r in cal["nutrient_ranges"].items():
                    a.ec_range[stage], a.ph_range[stage] = tuple(r["ec"]), tuple(r["ph"])
                a.source = f"{cal.get('facility', 'facility')} SOP, checked against batch data"

    # ------------------------------------------------------------ actuators
    def _actuators(self) -> dict[str, Callable[[Proposal, Context], Any]]:
        s = self.state

        def setpoint(p: Proposal, ctx: Context) -> None:
            s.setpoints[p.payload["room_id"]] = {k: p.payload.get(k) for k in
                                                 ("action", "rh_setpoint", "temp_setpoint", "setpoint")
                                                 if p.payload.get(k) is not None}

        def message(p: Proposal, ctx: Context) -> None:
            """Clinicians get the technical summary; patients get plain language through the
            messenger (urgent always delivered; routine de-duplicated, capped, quiet hours)."""
            default = {"interaction_alert": ["treating_clinician", "pharmacist"]}.get(p.task, ["patient"])
            to = p.payload.get("notify", default)
            staff = [r for r in to if r != "patient"]
            if staff:
                s.outbox.append({"to": staff, "patient_id": p.payload.get("patient_id"), "task": p.task,
                                 "text": p.summary})
            if "patient" in to and p.payload.get("patient_event"):
                self.messenger.send(p.payload["patient_id"], p.payload["patient_event"], p.payload, ctx.now)
            if p.task == "patient_progress_summary" and p.payload.get("patient_id") in self.patients:
                self.patients[p.payload["patient_id"]]["last_summary_at"] = ctx.now

        def hold(p: Proposal, ctx: Context) -> None:
            s.batch_status[p.payload["batch_id"]] = "hold"

        def release(p: Proposal, ctx: Context) -> None:
            if s.batch_status.get(p.payload["batch_id"]) != "hold":
                s.batch_status[p.payload["batch_id"]] = "released"

        def work_order(p: Proposal, ctx: Context) -> None:
            s.work_orders.append(p.payload)

        def irrigate(p: Proposal, ctx: Context) -> None:
            s.irrigation_pulses[p.payload["room_id"]] = s.irrigation_pulses.get(p.payload["room_id"], 0) + p.payload["pulses"]

        def dim(p: Proposal, ctx: Context) -> None:
            s.lighting[p.payload["room_id"]] = p.payload["ppfd_setpoint"]

        def quarantine(p: Proposal, ctx: Context) -> None:
            s.quarantined_products.add(p.payload["product_id"])

        def rollback(p: Proposal, ctx: Context) -> None:
            self.registry.rollback(p.payload["model_id"])

        def promote(p: Proposal, ctx: Context) -> None:   # only reached after the model owner approves
            loop = self.learning_loops.get(p.payload["model_id"])
            if loop:
                loop.approve_pending(loop.owner)

        def titration(p: Proposal, ctx: Context) -> None:   # only reached after clinician approval
            pt = self.patients.get(p.payload["patient_id"])
            plan: TitrationPlan | None = pt and pt.get("plan")
            if not plan:
                return
            act = p.payload["action"]
            if act == "advance":
                plan.current_step = p.payload["to_step"]
                plan.step_started = ctx.now
                plan.phase = "titration"
            elif act == "enter_maintenance":
                plan.phase = "maintenance"
            plan.history.append({"at": ctx.now.isoformat(), "action": act})

        return {"climate_setpoint": setpoint, "sensor_failsafe": setpoint, "drying_environment": setpoint,
                "patient_reminder": message, "safety_alert": message, "interaction_alert": message,
                "patient_progress_summary": message,
                "batch_hold": hold, "batch_release": release, "calibration_schedule": work_order,
                "irrigation_schedule": irrigate, "lighting_energy": dim, "hemp_compliance_check": quarantine,
                "model_rollback": rollback, "titration_step": titration, "model_deployment": promote}

    # ------------------------------------------------------------ patients
    def enroll_patient(self, identity: str, record: dict, live: dict) -> str:
        """Pseudonymise at the point of collection; identity goes only to the vault."""
        pid = self.vault.pseudonymize(identity)
        self.store.append("patient", {**record, "patient_id": pid},
                          Provenance(source="intake", operator=record.get("intake_by", "intake")))
        self.patients[pid] = live
        return pid

    def set_titration_plan(self, pid: str, plan: TitrationPlan, clinician_role: str) -> None:
        if clinician_role != "treating_clinician":
            raise PermissionError("Only the treating clinician sets a titration plan.")
        self.patients[pid]["plan"] = plan
        self.patients[pid]["status"] = "titrating"

    # ------------------------------------------------------------ loop
    def context(self, observations: dict, now: datetime | None = None) -> Context:
        obs = dict(observations)
        obs.setdefault("patients", self.patients)
        obs.setdefault("learning_loops", self.learning_loops)
        return Context(self.store, self.approvals, self.registry, self.config, self.llm,
                       now or datetime.now(timezone.utc), obs)

    async def run_cycle(self, observations: dict, now: datetime | None = None) -> CycleReport:
        ctx = self.context(observations, now)
        report = await self.swarm.cycle(ctx)
        self.history.append(report)
        # Consults are delivered once; the patient waits for the clinician's decision.
        for pid, pt in self.patients.items():
            if pt.get("status") == "new_consult":
                pt["status"] = "awaiting_clinician"
        return report

    async def run_forever(self, feed: Callable[[], Awaitable[dict]], interval_s: float = 60.0,
                          stop: asyncio.Event | None = None) -> None:
        """Continuous autonomous operation. `feed` pulls fresh observations from
        sensors, LIMS, track-and-trace, the patient app and regulatory feeds."""
        stop = stop or asyncio.Event()
        while not stop.is_set():
            try:
                await self.run_cycle(await feed())
            except Exception:        # one bad cycle must not stop the facility; it is logged and alerted
                log.exception("cycle failed; controllers remain on last known-good setpoints")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------ humans
    def decide(self, request_id: str, approve: bool, by: str, role: str, note: str = "",
               now: datetime | None = None) -> ApprovalRequest:
        req = self.approvals.decide(request_id, approve, by, role, note)
        ctx = self.context({}, now)
        # The decision itself is persisted by ApprovalQueue (hash-chained, survives restarts).
        if approve:
            act = self.swarm.actuators.get(req.task)
            if act:
                act(Proposal(agent=by, task=req.task, summary=req.summary, payload=req.payload, id=req.proposal_id), ctx)
        return req
