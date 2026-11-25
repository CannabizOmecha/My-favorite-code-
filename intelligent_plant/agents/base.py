"""Core agent abstractions. Agents observe shared context and emit
Proposals; Critics attack Proposals; the Swarm governs what happens next."""
from __future__ import annotations

import abc
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..governance.ethics import OBJECTIVES

if TYPE_CHECKING:
    from ..data.store import RecordStore
    from ..governance.autonomy import ApprovalQueue
    from ..governance.registry import ModelRegistry
    from .llm import ClaudeReasoner


@dataclass
class Proposal:
    agent: str
    task: str                         # key into governance.autonomy.TASKS
    summary: str
    payload: dict = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)   # objective -> score (higher is better)
    evidence: list[str] = field(default_factory=list)
    uncertainty: str | None = None
    target: str | None = None         # resource contended by competing proposals (e.g. "room:F1:climate")
    model_version: str = "rules-1.0"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        for k in OBJECTIVES:
            self.scores.setdefault(k, 0.0)


@dataclass
class Objection:
    critic: str
    proposal_id: str
    severity: str                     # "veto" (deterministic critics only) | "warn"
    message: str


@dataclass
class Context:
    store: RecordStore
    approvals: ApprovalQueue
    registry: ModelRegistry
    config: dict
    llm: ClaudeReasoner | None = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    observations: dict[str, Any] = field(default_factory=dict)   # live feeds keyed by domain
    alerts: list[dict] = field(default_factory=list)

    def alert(self, level: str, source: str, message: str, **extra: Any) -> None:
        self.alerts.append({"level": level, "source": source, "message": message,
                            "at": self.now.isoformat(), **extra})


class Agent(abc.ABC):
    name: str = "agent"
    domain: str = "general"
    version: str = "1.0"

    @abc.abstractmethod
    async def propose(self, ctx: Context) -> list[Proposal]:
        ...


class Critic(abc.ABC):
    name: str = "critic"

    def applies_to(self, p: Proposal) -> bool:
        return True

    @abc.abstractmethod
    async def critique(self, p: Proposal, ctx: Context) -> list[Objection]:
        ...


Actuator = Callable[[Proposal, Context], Any]
