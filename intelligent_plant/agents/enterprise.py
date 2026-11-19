"""Enterprise swarm (Chapter 16): traceability, maturity, model drift and
the outcome -> breeding feedback loop."""
from __future__ import annotations

import statistics
from collections import defaultdict

from ..governance.registry import assess_maturity
from .base import Agent, Context, Proposal


class TraceabilityAgent(Agent):
    """Every break in outcome -> batch -> room -> genotype costs the ability to learn."""
    name, domain = "traceability", "enterprise"

    async def propose(self, ctx: Context) -> list[Proposal]:
        outcomes = ctx.store.all("outcome")
        breaks: dict[str, int] = defaultdict(int)
        for o in outcomes:
            for b in ctx.store.trace_outcome(o["outcome_id"])["breaks"]:
                breaks[b] += 1
        if not breaks:
            return []
        return [Proposal(self.name, "alerting", f"Traceability breaks across {len(outcomes)} outcomes: {dict(breaks)}",
                         {"breaks": dict(breaks)}, {"accuracy": 1.0}, ["Ch 16.1"])]


class MaturityAgent(Agent):
    name, domain = "maturity", "enterprise"

    async def propose(self, ctx: Context) -> list[Proposal]:
        r = ctx.observations.get("readiness")
        if not r:
            return []
        lvl, label, todo = assess_maturity(**r)
        return [Proposal(self.name, "alerting", f"Analytics maturity {label}. Next: {todo or ['sustain']}",
                         {"level": lvl, "next": todo}, {"accuracy": 0.8}, ["Table 16.1", "Appendix C.4"])]


class DriftAgent(Agent):
    name, domain = "model_drift", "enterprise"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for model_id, recent in ctx.observations.get("model_inputs", {}).items():
            psi, drifted = ctx.registry.drift(model_id, recent)
            if drifted:
                out.append(Proposal(self.name, "model_rollback",
                                    f"{model_id}: input drift PSI={psi:.2f} > 0.25; roll back to previous version and "
                                    f"review.", {"model_id": model_id, "psi": psi}, {"safety": 0.8, "accuracy": 0.8},
                                    ["Ch 16.2 monitoring and rollback"]))
        return out


class OutcomeLearningAgent(Agent):
    """Closes the loop: aggregates patient outcomes by chemotype and proposes
    breeding-objective updates. Observational data: associations, not causes."""
    name, domain = "outcome_learning", "enterprise"

    def __init__(self, min_patients: int = 10):
        self.min_patients = min_patients

    async def propose(self, ctx: Context) -> list[Proposal]:
        by_chem: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for o in ctx.store.all("outcome"):
            t = ctx.store.trace_outcome(o["outcome_id"])
            if t["breaks"] or not t.get("coa"):
                continue
            c = t["coa"]
            tot_thc = c["thc"] + 0.877 * c["thca"]
            tot_cbd = c["cbd"] + 0.877 * c["cbda"]
            chem = "CBD-dominant" if tot_cbd >= 5 * tot_thc else "THC-dominant" if tot_thc >= 5 * tot_cbd else "balanced"
            by_chem[chem][t["administration"]["patient_id"]].append(o["score_pre"] - o["score_post"])
        summary = {}
        for chem, pts in by_chem.items():
            if len(pts) >= self.min_patients:
                means = [statistics.mean(v) for v in pts.values()]
                summary[chem] = {"patients": len(pts), "mean_improvement": round(statistics.mean(means), 2),
                                 "sd": round(statistics.pstdev(means), 2)}
        if not summary:
            return []
        return [Proposal(self.name, "breeding_objective_update",
                         f"Outcome registry by chemotype: {summary}. Consider weighting breeding targets accordingly.",
                         {"summary": summary}, {"accuracy": 0.5, "efficiency": 0.4},
                         ["Ch 16.1 feedback loop", "Ch 5.4 registries"],
                         uncertainty="Observational, confounded by indication and patient selection; use target trial "
                                     "emulation before acting (ref 99).")]


def enterprise_agents() -> list[Agent]:
    return [TraceabilityAgent(), MaturityAgent(), DriftAgent(), OutcomeLearningAgent()]
