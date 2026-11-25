"""The swarm: parallel divergence, adversarial checking, lexicographic
governance and autonomy gating, with everything logged.

Cycle:
  1. every agent proposes, in parallel, isolated from each other's failures
  2. every applicable critic attacks every proposal, in parallel
  3. deterministic vetoes block; warnings force human review
  4. competing proposals for one resource are resolved lexicographically
  5. the autonomy gate decides: execute, execute+alarm, execute+notify,
     queue for a named human, or refuse (human-only tasks)
  6. every disposition is appended to the hash-chained store
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from ..data.store import Provenance
from ..governance.autonomy import ApprovalRequest, Autonomy, policy_for
from ..governance.ethics import LexicographicPolicy
from .base import Actuator, Agent, Context, Critic, Objection, Proposal
from .llm import GROUNDING

log = logging.getLogger(__name__)


# Every disposition the swarm can produce. Anything that renders or counts outcomes must cover all of them.
OUTCOMES = ("executed", "executed_with_alarm", "executed_notify", "queued", "info", "vetoed", "refused",
            "superseded", "actuation_failed")


@dataclass
class Disposition:
    proposal: Proposal
    outcome: str          # one of OUTCOMES
    objections: list[Objection] = field(default_factory=list)
    detail: str = ""


@dataclass
class CycleReport:
    dispositions: list[Disposition]
    agent_failures: dict[str, str]
    alerts: list[dict]

    def by_outcome(self) -> dict[str, list[Disposition]]:
        d: dict[str, list[Disposition]] = defaultdict(list)
        for x in self.dispositions:
            d[x.outcome].append(x)
        return d


class Swarm:
    def __init__(self, agents: list[Agent], critics: list[Critic], policy: LexicographicPolicy | None = None,
                 actuators: dict[str, Actuator] | None = None, agent_timeout_s: float = 30.0):
        self.agents, self.critics = agents, critics
        self.policy = policy or LexicographicPolicy()
        self.actuators = actuators or {}
        self.agent_timeout_s = agent_timeout_s

    async def _run_agent(self, a: Agent, ctx: Context) -> list[Proposal]:
        return await asyncio.wait_for(a.propose(ctx), timeout=self.agent_timeout_s)

    async def cycle(self, ctx: Context) -> CycleReport:
        results = await asyncio.gather(*(self._run_agent(a, ctx) for a in self.agents), return_exceptions=True)
        proposals: list[Proposal] = []
        failures: dict[str, str] = {}
        for a, r in zip(self.agents, results, strict=True):
            if isinstance(r, BaseException):
                # Fail safe: a failed agent produces no action, and a human hears about it.
                failures[a.name] = f"{type(r).__name__}: {r}"
                ctx.alert("warn", a.name, f"Agent failed and took no action: {r}")
            else:
                proposals.extend(r)

        crit_jobs = [(p, c) for p in proposals for c in self.critics if c.applies_to(p)]
        crit_results = await asyncio.gather(*(c.critique(p, ctx) for p, c in crit_jobs), return_exceptions=True)
        objections: dict[str, list[Objection]] = defaultdict(list)
        for (p, c), cr in zip(crit_jobs, crit_results, strict=True):
            if isinstance(cr, BaseException):
                # A deterministic safety critic that cannot run has not cleared the proposal: fail closed.
                sev = "veto" if isinstance(c, SafetyCritic) else "warn"
                objections[p.id].append(Objection(c.name, p.id, sev, f"critic failed: {cr}"))
            else:
                objections[p.id].extend(cr)

        dispositions: list[Disposition] = []
        live: list[Proposal] = []
        for p in proposals:
            obs = objections[p.id]
            if any(o.severity == "veto" for o in obs):
                dispositions.append(Disposition(p, "vetoed", obs, "; ".join(o.message for o in obs if o.severity == "veto")))
                ctx.alert("warn", "governance", f"Vetoed {p.task} from {p.agent}: {p.summary}")
            else:
                live.append(p)

        # Conflict resolution per contended resource.
        groups: dict[str, list[Proposal]] = defaultdict(list)
        for p in live:
            groups[p.target or p.id].append(p)
        winners: list[Proposal] = []
        for ps in groups.values():
            if len(ps) == 1:
                winners.append(ps[0])
                continue
            best = self.policy.choose(ps)
            for p in ps:
                if p is best:
                    winners.append(p)
                else:
                    dispositions.append(Disposition(p, "superseded", objections[p.id],
                                                    f"Lexicographic ordering preferred {getattr(best, 'id', 'none')}"))

        for p in winners:
            dispositions.append(self._gate(p, objections[p.id], ctx))

        for d in dispositions:
            self._log(d, ctx)
        return CycleReport(dispositions, failures, ctx.alerts)

    def _gate(self, p: Proposal, obs: list[Objection], ctx: Context) -> Disposition:
        pol = policy_for(p.task)
        if not self.policy.admissible(p):
            return Disposition(p, "vetoed", obs, "Fails safety floor of the lexicographic policy.")
        if pol.autonomy == Autonomy.HUMAN_ONLY:
            ctx.alert("info", "governance", f"{p.agent} attempted human-only task {p.task}; refused.")
            return Disposition(p, "refused", obs, f"{p.task} is reserved to a human ({pol.owner_role}).")
        warned = [o for o in obs if o.severity == "warn"]
        if pol.autonomy.executes_without_human and not warned:
            err = self._actuate(p, ctx)
            if err:
                return Disposition(p, "actuation_failed", obs, err)
            if pol.autonomy == Autonomy.AUTOMATE_WITH_ALARMS and p.payload.get("alarm"):
                ctx.alert("alarm", p.agent, p.payload["alarm"])
                return Disposition(p, "executed_with_alarm", obs)
            if pol.autonomy == Autonomy.AUTOMATE_SUPERVISED:
                ctx.alert("notify", p.agent, f"[{pol.owner_role}] executed, reversible: {p.summary}")
                return Disposition(p, "executed_notify", obs)
            return Disposition(p, "executed", obs)
        if pol.autonomy == Autonomy.DECISION_SUPPORT:
            ctx.approvals.submit(ApprovalRequest(p.id, p.task, pol.owner_role, p.summary, p.payload,
                                                 p.evidence, [o.message for o in obs]))
            return Disposition(p, "info", obs, f"Decision support delivered to {pol.owner_role}.")
        # RECOMMEND / TRIAGE, or an automatable action that drew adversarial warnings.
        ctx.approvals.submit(ApprovalRequest(p.id, p.task, pol.owner_role, p.summary, p.payload, p.evidence,
                                             [o.message for o in obs]))
        detail = "Escalated by critic warnings." if warned and pol.autonomy.executes_without_human else ""
        return Disposition(p, "queued", obs, detail)

    def _actuate(self, p: Proposal, ctx: Context) -> str | None:
        """Run the actuator. A failure is never silent and never aborts the cycle (the audit
        log must still be written); a human is alarmed and the action is reported as not done."""
        act = self.actuators.get(p.task)
        if act:
            try:
                act(p, ctx)
            except Exception as e:  # noqa: BLE001 - any actuator failure must alarm a human, never abort audit
                msg = f"Actuator for {p.task} failed ({type(e).__name__}: {e}); not done, human action required."
                ctx.alert("alarm", "governance", f"{msg} {p.summary}")
                return msg
        return None

    def _log(self, d: Disposition, ctx: Context) -> None:
        p = d.proposal
        ctx.store.append("decision", {
            "proposal_id": p.id, "agent": p.agent, "action": p.task, "disposition": d.outcome,
            "summary": p.summary, "payload": p.payload, "scores": p.scores, "evidence": p.evidence,
            "uncertainty": p.uncertainty, "model_version": p.model_version, "detail": d.detail,
            "objections": [f"{o.critic}:{o.severity}:{o.message}" for o in d.objections],
        }, Provenance(source="swarm", operator=p.agent))


# ---------------------------------------------------------------- critics
_DOSE_KEY_PARTS = ("dose_mg", "starting_dose", "next_dose", "dose_step", "mg_per_dose", "dose_amount")


def _originates_dose(o) -> bool:
    """True if a payload (at any depth) carries a dose number. Patient-log fields such as
    dose_mg_thc are records of what the patient took and live in the store, never in proposals."""
    if isinstance(o, dict):
        for k, v in o.items():
            key = str(k).lower()
            if any(part in key for part in _DOSE_KEY_PARTS) or key in {"dose", "mg"}:
                return True
            if _originates_dose(v):
                return True
        return False
    if isinstance(o, (list, tuple, set)):
        return any(_originates_dose(x) for x in o)
    return False


class SafetyCritic(Critic):
    """Deterministic. Vetoes anything that lowers safety below baseline or
    lacks uncertainty disclosure on a clinical recommendation."""
    name = "safety_officer"

    async def critique(self, p: Proposal, ctx: Context) -> list[Objection]:
        out = []
        if p.scores.get("safety", 0) < 0:
            out.append(Objection(self.name, p.id, "veto", "Introduces risk beyond baseline (Safety first)."))
        if p.task in {"product_recommendation", "titration_step", "consultation_report"} and not p.uncertainty:
            out.append(Objection(self.name, p.id, "veto", "Clinical output without stated uncertainty."))
        if _originates_dose(p.payload):   # no self-declared exemption: agents never originate doses
            out.append(Objection(self.name, p.id, "veto", "Agents may not originate dose numbers."))
        return out


class EvidenceCritic(Critic):
    """Deterministic. Recommendations must cite evidence (Chapter 13.4)."""
    name = "evidence_auditor"

    def applies_to(self, p: Proposal) -> bool:
        return policy_for(p.task).autonomy <= Autonomy.RECOMMEND

    async def critique(self, p: Proposal, ctx: Context) -> list[Objection]:
        if not p.evidence:
            return [Objection(self.name, p.id, "warn", "No evidence provenance attached.")]
        return []


class DataQualityCritic(Critic):
    """Deterministic. Level-3 models on level-1 data produce confident nonsense
    (Chapter 16.4): actions built on uncalibrated inputs are escalated."""
    name = "data_quality"

    def applies_to(self, p: Proposal) -> bool:
        return p.task != "sensor_failsafe"      # the fail-safe IS the response to bad data; never delay it

    async def critique(self, p: Proposal, ctx: Context) -> list[Objection]:
        if p.payload.get("inputs_calibrated") is False:
            return [Objection(self.name, p.id, "warn", "Inputs include out-of-calibration sensor data.")]
        return []


JURY_PERSONAS = {
    "skeptic": "You look for hype, unsupported causal claims and evidence graded higher than it is.",
    "patient_advocate": "You represent the patient at the end of the supply chain: consistency, harm, dignity.",
    "regulator": "You check traceability, record integrity and whether a human with authority is accountable.",
    "economist": "You check whether efficiency or profit is being bought with safety or ethics.",
    "equity_officer": "You check that communities harmed by prohibition are not excluded and that genetic or "
                      "behavioural data are not used for stigma or gatekeeping.",
}


class LLMJuryCritic(Critic):
    """A jury of Claude personas debating in parallel. Jurors may only WARN
    (forcing human review); they can never approve or veto - vetoes stay
    deterministic and auditable."""

    def __init__(self, persona: str):
        self.persona = persona
        self.name = f"jury:{persona}"

    def applies_to(self, p: Proposal) -> bool:
        return policy_for(p.task).autonomy <= Autonomy.RECOMMEND

    async def critique(self, p: Proposal, ctx: Context) -> list[Objection]:
        if not ctx.llm or not ctx.llm.available:
            return []
        prompt = (f"Review this proposal from agent '{p.agent}' for task '{p.task}'.\n<data>\n"
                  f"summary: {p.summary}\npayload: {p.payload}\nevidence: {p.evidence}\n"
                  f"uncertainty: {p.uncertainty}\n</data>\n"
                  'Return {"severity": "none" | "warn", "message": "<one sentence>"}.')
        r = await ctx.llm.complete_json(GROUNDING + "\nPersona: " + JURY_PERSONAS[self.persona], prompt)
        if r and str(r.get("severity", "")).strip().lower() == "warn":
            return [Objection(self.name, p.id, "warn", str(r.get("message", ""))[:500])]
        return []


def default_critics(with_jury: bool = True) -> list[Critic]:
    cs: list[Critic] = [SafetyCritic(), EvidenceCritic(), DataQualityCritic()]
    if with_jury:
        cs += [LLMJuryCritic(p) for p in JURY_PERSONAS]
    return cs
