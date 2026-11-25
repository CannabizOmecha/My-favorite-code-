"""Round-2 hardening regressions (static analysis, red team, property fuzzing, chaos/stress).

Every test here failed on the code before its fix. Each names the failure it guards against so
it can never silently come back."""
import asyncio
import secrets
from datetime import timedelta

import numpy as np
import pytest

from intelligent_plant.agents.base import Context
from intelligent_plant.agents.moe import Expert, ExpertOpinion, MixtureOfExperts
from intelligent_plant.agents.swarm import OUTCOMES
from intelligent_plant.data.store import RecordStore
from intelligent_plant.governance.autonomy import ApprovalQueue
from intelligent_plant.governance.registry import ModelRegistry
from intelligent_plant.knowledge.regulation import load_jurisdictions
from intelligent_plant.orchestrator import IntelligentEnterprise
from intelligent_plant.quantum.statevector import Statevector
from intelligent_plant.science import stats
from intelligent_plant.simulation import NOW


def run(coro):
    return asyncio.run(coro)


def bare_ctx(obs=None):
    return Context(RecordStore(), ApprovalQueue(), ModelRegistry(), load_jurisdictions(), None, NOW, obs or {})


# ------------------------------------------------------------------ reporting
def test_demo_renders_every_possible_outcome():
    """A new outcome ('actuation_failed') once crashed the demo report with KeyError."""
    from intelligent_plant.__main__ import ICON
    assert set(OUTCOMES) <= set(ICON)


# ------------------------------------------------------------------ N-of-1 blinding
def test_n_of_1_schedule_uses_cryptographic_rng_when_unseeded(monkeypatch):
    """A predictable allocation order breaks blinding (allocation concealment)."""
    used = []

    class Spy(secrets.SystemRandom):
        def shuffle(self, x):
            used.append(True)
            super().shuffle(x)
    monkeypatch.setattr(stats.secrets, "SystemRandom", Spy)
    sched = stats.n_of_1_schedule(4)
    assert used and sorted(sched) == ["A"] * 4 + ["B"] * 4
    assert stats.n_of_1_schedule(6, seed=1) == stats.n_of_1_schedule(6, seed=1)   # seeded = reproducible tests


def test_n_of_1_allocation_is_concealed_from_clinician_view_and_audit_log():
    from intelligent_plant.agents.clinical import TitrationMonitorAgent, TitrationPlan
    from intelligent_plant.knowledge.interactions import PatientRiskProfile
    from intelligent_plant.simulation import patient_logs
    ent = IntelligentEnterprise(use_llm=False)
    plan = TitrationPlan("P1", "dr", ["step"], 24, 9999, 2.0, NOW - timedelta(days=60), phase="maintenance")
    pts = {"P1": {"plan": plan, "logs": patient_logs(31, NOW - timedelta(days=40)),
                  "profile": PatientRiskProfile(age=40), "last_summary_at": NOW}}
    out = run(TitrationMonitorAgent().propose(ent.context({"patients": pts})))
    prop = next(p for p in out if p.task == "n_of_1_enrolment")
    assert "schedule" not in prop.payload and prop.payload["allocation"] == "concealed"
    ent.patients["P1"] = pts["P1"]
    ent.swarm.actuators["n_of_1_enrolment"](prop, ent.context({}))
    assert plan.n_of_1_enrolled
    for role in ("treating_clinician", "patient", "care_coordinator"):
        with pytest.raises(PermissionError):
            ent.n_of_1_allocation("P1", role)
    sched = ent.n_of_1_allocation("P1", "pharmacist")
    assert sorted(sched) == ["A"] * 4 + ["B"] * 4


# ------------------------------------------------------------------ mixture of experts
def test_blocking_opinion_without_recommendation_still_blocks():
    """A blocking expert that gave no categorical answer was dropped from the vote (fail open)."""
    class Blocker(Expert):
        name, keywords = "blocker", ("x",)

        async def opine(self, q, c):
            return ExpertOpinion(self.name, None, 0.9, ["[STOP] reason"], question="trial", blocking=True)

    class Yes(Expert):
        name, keywords = "yes", ("x",)

        async def opine(self, q, c):
            return ExpertOpinion(self.name, "trial_reasonable", 1.0, question="trial")
    moe = MixtureOfExperts([Blocker(), Yes()])
    res = run(moe.consult({"a": "x"}, bare_ctx()))
    assert moe.answer(res, "trial") == "blocked"


# ------------------------------------------------------------------ quantum state validation
def test_statevector_rejects_wrong_shape_or_unnormalised_amplitudes():
    with pytest.raises(ValueError):
        Statevector(2, np.ones(3, dtype=complex))
    with pytest.raises(ValueError):
        Statevector(1, np.array([1.0, 1.0], dtype=complex))
    sv = Statevector(1, np.array([1.0, 0.0], dtype=complex))
    assert sv.probabilities()[0] == pytest.approx(1.0)

