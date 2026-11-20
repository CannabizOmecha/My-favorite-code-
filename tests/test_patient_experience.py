"""The patient is the reason the system exists. These tests hold that line."""
import asyncio
import re
from datetime import datetime, timedelta, timezone

import pytest

from intelligent_plant.agents.clinical import ConsultationAgent
from intelligent_plant.knowledge.interactions import PatientRiskProfile
from intelligent_plant.orchestrator import IntelligentEnterprise
from intelligent_plant.patient.communication import (TEMPLATES, URGENT_HELP, PatientMessenger, progress_summary,
                                                     reading_grade)
from intelligent_plant.simulation import NOW, build_world

JARGON = re.compile(r"P-[0-9a-f]{8,}|DDI|CYP|COA|mg/kg|titration|pharmacokinetic|\bAE\b", re.I)


def _all_texts():
    m = PatientMessenger()
    payload = {"adverse_effects": [{"name": "dizziness"}], "missing": ["score_post", "route"],
               "summary": progress_summary([{"score_pre": 7, "score_post": 5}] * 10, "sleep")}
    return {k: m.compose(k, payload)[1] for k in TEMPLATES}


@pytest.mark.parametrize("event", list(TEMPLATES))
def test_patient_messages_are_plain_language(event):
    text = _all_texts()[event]
    assert reading_grade(text) <= 8.0, text
    assert not JARGON.search(text), text


def test_urgent_messages_always_say_when_to_get_help():
    for event, (urgency, _) in TEMPLATES.items():
        if urgency == "urgent":
            assert URGENT_HELP in _all_texts()[event]
    assert "1-800-222-1222" in URGENT_HELP and "911" in URGENT_HELP


def test_urgent_always_delivered_routine_is_rate_limited():
    m = PatientMessenger(max_routine_per_day=2)
    night = datetime(2026, 11, 19, 23, 30, tzinfo=timezone.utc)
    assert m.send("p", "oral_redose", {}, night).delivered                    # urgent: even at night
    assert not m.send("p", "multiple_changes", {}, night).delivered           # routine: quiet hours
    day = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
    assert m.send("p", "multiple_changes", {}, day).delivered
    assert not m.send("p", "multiple_changes", {}, day + timedelta(hours=1)).delivered     # duplicate
    assert m.send("p", "missing_log_fields", {"missing": ["route"]}, day).delivered
    assert not m.send("p", "medication_check", {}, day).delivered             # daily cap of 2 routine
    assert m.send("p", "adverse_effect", {}, day).delivered                   # urgent still goes through


def test_progress_summary_is_honest_about_little_data():
    assert "too few records" in progress_summary([{"score_pre": 7, "score_post": 3}] * 3)
    assert "clear change" in progress_summary([{"score_pre": 7, "score_post": 4}] * 10)
    assert "little change" in progress_summary([{"score_pre": 6, "score_post": 6}] * 10)


def test_avoid_psychoactivity_preference_is_respected():
    ent = IntelligentEnterprise(use_llm=False)
    world = build_world(ent)
    ctx = ent.context(world)
    agent = ConsultationAgent()
    base = {"indication": "chronic_pain_adult", "standard_care_tried": True, "profile": PatientRiskProfile(age=50),
            "route_hint": "inhaled"}
    open_ = asyncio.run(agent.moe.experts["product_match"].opine(base, ctx))
    avoid = asyncio.run(agent.moe.experts["product_match"].opine(
        {**base, "preferences": {"avoid_psychoactivity": True}}, ctx))
    assert open_.recommendation == "PR-FLOWER-HI" and avoid.recommendation is None


def test_route_preference_honoured_only_when_clinically_equivalent():
    ent = IntelligentEnterprise(use_llm=False)
    ctx = ent.context({})
    route = ConsultationAgent().moe.experts["route"]
    o = asyncio.run(route.opine({"onset_need": "sustained", "preferences": {"route_preference": "oromucosal"}}, ctx))
    assert o.recommendation == "oromucosal"
    o = asyncio.run(route.opine({"onset_need": "sustained", "preferences": {"route_preference": "inhaled"}}, ctx))
    assert o.recommendation == "oral" and any("Patient prefers inhaled" in f for f in o.flags)


def test_end_to_end_patient_gets_plain_messages_and_progress():
    ent = IntelligentEnterprise(use_llm=False)
    asyncio.run(ent.run_cycle(build_world(ent), now=NOW))
    texts = [m.text for m in ent.messenger.sent]
    assert any("do not take any more today" in t for t in texts)
    assert any(t.startswith("Here is how you are doing") for t in texts)
    assert all(not JARGON.search(t) for t in texts)
    consult = [d for d in ent.store.all("decision") if d["action"] == "consultation_report"
               and "chronic_pain_adult" in d["summary"]][0]
    assert consult["payload"]["sequence"]["0_patient_priorities"]["goals"] == ["sleep", "walking the dog again"]
    # Weekly cadence: no second summary the next day.
    asyncio.run(ent.run_cycle({}, now=NOW + timedelta(days=1)))
    assert sum(t.startswith("Here is how") for t in (m.text for m in ent.messenger.sent)) == 1
