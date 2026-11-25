"""Mixture-of-experts router.

A gating function scores each expert's relevance to a query, weights it by
the expert's track record, and consults the top-k IN PARALLEL. Experts may
abstain (out of scope) rather than extrapolate. Outputs combine by weighted
vote, except that safety flags are a UNION: a minority expert's safety flag
is never outvoted (Safety > everything)."""
from __future__ import annotations

import abc
import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

from .base import Context


@dataclass
class ExpertOpinion:
    expert: str
    recommendation: str | None = None          # categorical answer (may be None for flag-only experts)
    confidence: float = 0.5
    flags: list[str] = field(default_factory=list)       # safety flags - unioned, never weighted
    findings: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    question: str = "default"                  # experts vote only against others answering the same question
    blocking: bool = False                     # a safety-blocking answer wins its question outright


class Expert(abc.ABC):
    name = "expert"
    keywords: tuple[str, ...] = ()

    def relevance(self, query: dict) -> float:
        text = " ".join(str(v) for v in query.values()).lower()
        hits = sum(1 for k in self.keywords if k in text)
        return hits / (len(self.keywords) or 1)

    @abc.abstractmethod
    async def opine(self, query: dict, ctx: Context) -> ExpertOpinion | None:
        """Return None to abstain."""


@dataclass
class MoEResult:
    answers: dict[str, tuple[str, float]]     # question -> (recommendation, support)
    flags: list[str]
    opinions: list[ExpertOpinion]
    consulted: list[str]
    abstained: list[str]
    gate_weights: dict[str, float]


class MixtureOfExperts:
    def __init__(self, experts: list[Expert], top_k: int = 4, temperature: float = 0.5,
                 always_consult: tuple[str, ...] = ()):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.experts = {e.name: e for e in experts}
        self.reliability = {e.name: 1.0 for e in experts}
        self.top_k, self.temperature = top_k, temperature
        self.always_consult = tuple(always_consult)     # ordered: results must not depend on hash seeds

    def gate(self, query: dict) -> dict[str, float]:
        if not self.experts:
            return {}
        logits = {n: (e.relevance(query) * math.log1p(self.reliability[n] * math.e)) / self.temperature
                  for n, e in self.experts.items()}
        mx = max(logits.values())
        exp = {n: math.exp(v - mx) for n, v in logits.items()}
        z = sum(exp.values())
        return {n: v / z for n, v in exp.items()}

    async def consult(self, query: dict, ctx: Context) -> MoEResult:
        w = self.gate(query)
        chosen = sorted(w, key=w.get, reverse=True)[: self.top_k]
        chosen = list(dict.fromkeys(list(self.always_consult) + chosen))
        results = await asyncio.gather(*(self.experts[n].opine(query, ctx) for n in chosen),
                                       return_exceptions=True)
        opinions, abstained, flags = [], [], []
        for n, r in zip(chosen, results):
            if isinstance(r, BaseException):
                abstained.append(n)
                flags.append(f"Expert {n} failed ({type(r).__name__}); its domain is unassessed.")
            elif r is None:
                abstained.append(n)
            else:
                opinions.append(r)
                flags.extend(r.flags)
        answers: dict[str, tuple[str, float]] = {}
        by_q: dict[str, list[ExpertOpinion]] = {}
        for o in opinions:
            if o.recommendation:
                by_q.setdefault(o.question, []).append(o)
        for q, ops in by_q.items():
            blocking = [o for o in ops if o.blocking]
            if blocking:
                # Every blocking verdict is reported, in a deterministic order.
                answers[q] = ("+".join(sorted({o.recommendation for o in blocking})), 1.0)
                continue
            votes: dict[str, float] = {}
            for o in ops:
                votes[o.recommendation] = votes.get(o.recommendation, 0.0) + w.get(o.expert, 0.0) * o.confidence
            rec = max(votes, key=votes.get)
            total = sum(votes.values())
            answers[q] = (rec, votes[rec] / total if total > 0 else 0.0)
        return MoEResult(answers, list(dict.fromkeys(flags)), opinions, chosen, abstained, w)

    def answer(self, result: MoEResult, question: str) -> str | None:
        a = result.answers.get(question)
        return a[0] if a else None

    def feedback(self, expert: str, was_right: bool, eta: float = 0.2) -> None:
        """Multiplicative-weights update from realised outcomes."""
        self.reliability[expert] *= math.exp(eta if was_right else -eta)
        self.reliability[expert] = min(5.0, max(0.2, self.reliability[expert]))
