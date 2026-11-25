"""Lexicographic ethical ordering (Chapter 1.5):
Safety > Ethics > Accuracy > Efficiency > Profitability.

Maximise f1; among solutions within the pre-written tolerance of the best f1,
maximise f2; and so on. Tolerances are declared up front, not tuned after."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

OBJECTIVES = ("safety", "ethics", "accuracy", "efficiency", "profitability")

# "accept any safety-equivalent option within 2% of the best" (Chapter 1.5).
DEFAULT_TOLERANCES = {"safety": 0.0, "ethics": 0.0, "accuracy": 0.02, "efficiency": 0.02, "profitability": 0.0}

SAFETY_BASELINE = 0.0   # a proposal may not introduce risk beyond the activity's baseline


class Scored(Protocol):
    scores: dict[str, float]


@dataclass
class LexicographicPolicy:
    tolerances: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOLERANCES))
    safety_floor: float = SAFETY_BASELINE

    def admissible(self, s: Scored) -> bool:
        """Hard constraint: no net added patient/crop/worker risk."""
        return s.scores.get("safety", 0.0) >= self.safety_floor

    def rank(self, options: Sequence[Scored]) -> list[Scored]:
        pool = [o for o in options if self.admissible(o)]
        for obj in OBJECTIVES:
            if len(pool) <= 1:
                break
            best = max(o.scores.get(obj, 0.0) for o in pool)
            tol = self.tolerances.get(obj, 0.0) * abs(best)
            pool = [o for o in pool if o.scores.get(obj, 0.0) >= best - tol - 1e-12]
        rest = [o for o in options if o not in pool]
        rest.sort(key=lambda o: tuple(-o.scores.get(k, 0.0) for k in OBJECTIVES))
        return pool + rest

    def choose(self, options: Sequence[Scored]) -> Scored | None:
        ranked = self.rank(options)
        return ranked[0] if ranked and self.admissible(ranked[0]) else None
