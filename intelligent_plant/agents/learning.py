"""Agents that put self-iterating learning and quantum optimisation into the
live swarm, under the same governance as everything else.

ctx.observations["learning_loops"]: {model_id: ChampionChallengerLoop}
ctx.observations["breeding"]["kinship"]: n x n coancestry matrix aligned with
  ["breeding"]["candidates"], and ["breeding"]["advance_k"]: how many to advance.
"""
from __future__ import annotations

import numpy as np

from ..quantum.advantage import decide, portfolio_qubo
from ..science.breeding import patient_first_index
from .base import Agent, Context, Proposal


class ContinualLearningAgent(Agent):
    """Runs one self-iteration of every registered learning loop per cycle."""
    name, domain = "continual_learning", "enterprise"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for model_id, loop in ctx.observations.get("learning_loops", {}).items():
            r = loop.step()
            ev = r.challenger
            payload = {"model_id": model_id, "cycle": r.cycle, "champion_version": r.champion_version,
                       "reason": r.reason, "drift_psi": r.drift_psi,
                       "challenger": ev.__dict__ if ev else None,
                       "champion": r.champion.__dict__ if r.champion else None}
            if loop.suspended:
                out.append(Proposal(self.name, "alerting",
                                    f"{model_id}: champion suspended (degraded on prospective data); predictions "
                                    f"abstain until a validated replacement exists. {r.reason}",
                                    {**payload, "alarm": f"{model_id} model suspended"},
                                    {"safety": 1.0, "accuracy": 1.0}, ["Ch 16.2 monitoring", "Ch 16.3 safe failure"],
                                    target=f"model:{model_id}:status"))
            elif r.pending_human_approval:
                out.append(Proposal(self.name, "model_deployment",
                                    f"{model_id}: challenger v{r.champion_version + 1} beat the champion prospectively "
                                    f"and passed the C.5 gate; owner approval required.", payload,
                                    {"safety": 0.4, "accuracy": 0.9}, ["Appendix C.5 deployment gate"],
                                    uncertainty=f"Prospective RMSE {ev.rmse:.3f}, 90% coverage {ev.coverage90:.0%} "
                                                f"on {ev.n} outcomes.", target=f"model:{model_id}:status"))
            elif r.promoted:
                out.append(Proposal(self.name, "alerting", f"{model_id}: self-retrained and promoted to "
                                    f"v{r.champion_version}. {r.reason}", payload, {"accuracy": 0.9},
                                    ["Ch 16.2 versioned models"], target=f"model:{model_id}:status"))
        return out


class QuantumPortfolioAgent(Agent):
    """Selects which cultivars to advance as a QUBO (patient-first value vs
    kinship penalty), solved classically and by QAOA; the decision gate keeps
    whichever is better and records both."""
    name, domain = "quantum_portfolio", "breeding"

    async def propose(self, ctx: Context) -> list[Proposal]:
        b = ctx.observations.get("breeding") or {}
        cands, K = b.get("candidates"), b.get("kinship")
        if not cands or K is None or len(cands) < 3:
            return []
        scores = [patient_first_index(c, b["target_ratio"]) for c in cands]
        value = np.array([(2 * s["consistency"] + s["contamination_resistance"] + s["terpene_stability"]
                           + 0.5 * s["efficiency"]) if s["eligible"] else -5.0 for s in scores])
        k = int(b.get("advance_k", min(3, len(cands) - 1)))
        d = decide(portfolio_qubo(value, np.asarray(K, float), k, diversity_weight=b.get("diversity_weight", 4.0)))
        chosen = [c.name for c, x in zip(cands, d.chosen.best_bits) if x]
        q = d.quantum
        return [Proposal(self.name, "breeding_selection",
                         f"Advance {chosen} (k={k}, diversity-aware). Solver: {d.chosen.method}. {d.rationale}",
                         {"selected": chosen, "classical": d.classical.__dict__, "quantum": q.__dict__ if q else None,
                          "gate": d.log},
                         {"safety": 0.5, "ethics": 0.2, "accuracy": 0.7},
                         ["Ch 9.3 patient-first objective", "Ch 9.4 germplasm diversity", "Ch 15.3 quantum maturity"],
                         uncertainty="Objective weights are a breeding-policy choice; value scores come from "
                                     "measured chemotype consistency and resistance traits.",
                         target="breeding:advance_set")]


def learning_agents() -> list[Agent]:
    return [ContinualLearningAgent(), QuantumPortfolioAgent()]
