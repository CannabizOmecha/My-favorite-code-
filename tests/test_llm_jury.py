import asyncio

from intelligent_plant.agents.base import Context, Proposal
from intelligent_plant.agents.llm import ClaudeReasoner, parse_json_object
from intelligent_plant.agents.swarm import LLMJuryCritic, Swarm, default_critics
from intelligent_plant.data.store import RecordStore
from intelligent_plant.governance.autonomy import ApprovalQueue
from intelligent_plant.governance.registry import ModelRegistry
from intelligent_plant.knowledge.regulation import load_jurisdictions
from intelligent_plant.simulation import NOW
from tests.test_governance import OneShot


class StubLLM:
    available = True

    def __init__(self, verdict):
        self.verdict, self.calls = verdict, 0

    async def complete_json(self, system, prompt):
        self.calls += 1
        assert "<data>" in prompt and "Never invent doses" in system
        return self.verdict


def _ctx(llm):
    return Context(RecordStore(), ApprovalQueue(), ModelRegistry(), load_jurisdictions(), llm, NOW, {})


def test_disabled_reasoner_is_inert():
    r = ClaudeReasoner(enabled=False)
    assert not r.available and asyncio.run(r.complete("s", "p")) is None


def test_jury_runs_in_parallel_and_only_warns():
    llm = StubLLM({"severity": "warn", "message": "Evidence is graded too high."})
    p = Proposal("a", "nutrient_recipe", "adjust EC", evidence=["Table 8.2"], uncertainty="u")
    c = _ctx(llm)
    r = asyncio.run(Swarm([OneShot(p)], default_critics(with_jury=True)).cycle(c))
    assert llm.calls == 5                          # five personas deliberated
    d = r.dispositions[0]
    assert d.outcome == "queued" and sum(o.critic.startswith("jury:") for o in d.objections) == 5
    assert all(o.severity == "warn" for o in d.objections if o.critic.startswith("jury:"))


def test_jury_ignores_automated_low_risk_tasks():
    llm = StubLLM({"severity": "warn", "message": "x"})
    asyncio.run(Swarm([OneShot(Proposal("a", "data_logging", "log"))], [LLMJuryCritic("skeptic")]).cycle(_ctx(llm)))
    assert llm.calls == 0


def test_json_parse_is_tolerant():
    assert parse_json_object('Sure: {"severity": "none"} ok') == {"severity": "none"}
    assert parse_json_object("no json") is None
