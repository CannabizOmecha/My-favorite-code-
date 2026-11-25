"""Safety and governance invariants: what the swarm may and may not do on its own."""
import asyncio
import sqlite3
from datetime import date, timedelta

import pytest

from intelligent_plant.agents.base import Agent, Context, Proposal
from intelligent_plant.agents.clinical import ConsultationAgent, TitrationPlan
from intelligent_plant.agents.swarm import Swarm, default_critics
from intelligent_plant.data.privacy import PseudonymVault, redact_for_role
from intelligent_plant.data.store import Provenance, RecordStore, ValidationError
from intelligent_plant.governance.autonomy import ApprovalQueue
from intelligent_plant.governance.ethics import LexicographicPolicy
from intelligent_plant.governance.registry import ModelCard, ModelRegistry, assess_maturity, deployment_gate
from intelligent_plant.knowledge.interactions import PatientRiskProfile, check_interactions, screen_contraindications
from intelligent_plant.knowledge.regulation import check_hemp_product, diff_regulation, load_jurisdictions
from intelligent_plant.orchestrator import IntelligentEnterprise
from intelligent_plant.simulation import NOW, build_world

P = Provenance("test", "tester")


def run(coro):
    return asyncio.run(coro)


def ctx():
    return Context(RecordStore(), ApprovalQueue(), ModelRegistry(), load_jurisdictions(), None, NOW, {})


class OneShot(Agent):
    def __init__(self, *proposals, name="test_agent"):
        self.ps, self.name = proposals, name

    async def propose(self, c):
        return list(self.ps)


class Crash(Agent):
    name = "crasher"

    async def propose(self, c):
        raise RuntimeError("sensor bus down")


# ------------------------------------------------------------------ store
def test_validation_at_entry_rejects_bad_records():
    s = RecordStore()
    with pytest.raises(ValidationError):
        s.append("outcome", {"outcome_id": "o", "admin_id": "a", "instrument": "x", "score_pre": 14,
                             "score_post": 3, "interval_min": 60}, P)
    with pytest.raises(ValidationError):
        s.append("coa", {"coa_id": "c"}, P)


def test_records_are_never_overwritten_only_corrected():
    s = RecordStore()
    rec = {"batch_id": "B1", "genotype_id": "G", "room_id": "R", "start_date": "2026-01-01", "yield_g": 100}
    s.append("batch", rec, P)
    with pytest.raises(ValidationError):
        s.append("batch", {**rec, "yield_g": 90}, P)
    s.correct("batch", {**rec, "yield_g": 90}, P, reason_code="UNIT_ERROR_G_VS_KG")
    h = s.history("batch", "B1")
    assert [v["data"]["yield_g"] for v in h] == [100, 90] and h[1]["reason_code"] == "UNIT_ERROR_G_VS_KG"
    assert s.get("batch", "B1")["yield_g"] == 90


def test_hash_chain_detects_tampering(tmp_path):
    db = str(tmp_path / "r.db")
    s = RecordStore(db)
    for i in range(3):
        s.append("batch", {"batch_id": f"B{i}", "genotype_id": "G", "room_id": "R", "start_date": "2026-01-01"}, P)
    assert s.verify_chain() == (True, None)
    raw = sqlite3.connect(db)
    raw.execute("UPDATE records SET data=replace(data,'\"R\"','\"R9\"') WHERE seq=2")
    raw.commit()
    assert s.verify_chain() == (False, 2)


def test_trace_outcome_full_chain():
    ent = IntelligentEnterprise(use_llm=False)
    build_world(ent)
    o = ent.store.all("outcome")[0]
    t = ent.store.trace_outcome(o["outcome_id"])
    assert t["breaks"] == [] and t["genotype"]["genotype_id"] == "G-CBD1" and t["environment_records"] > 0


def test_pseudonym_vault_least_privilege():
    v = PseudonymVault()
    p = v.pseudonymize("Jane Doe")
    assert p.startswith("P-") and "Jane" not in p and v.pseudonymize("Jane Doe") == p
    with pytest.raises(PermissionError):
        v.reidentify(p, "bob", "head_grower", "curiosity")
    assert v.reidentify(p, "dr", "treating_clinician", "care") == "Jane Doe"
    assert any("denied" in row[3] for row in v.audit_log())
    assert redact_for_role({"cyp2c9_genotype": "*3/*3"}, "compliance_officer")["cyp2c9_genotype"] == "[REDACTED]"


# ------------------------------------------------------------------ rules
def test_hard_interaction_rules():
    a = check_interactions(["Clobazam 10mg"], {"CBD"})
    assert a[0].rule_id == "DDI-CLOBAZAM" and a[0].severity == "major" and not a[0].overridable_by_model
    assert not any(x.rule_id == "DDI-CLOBAZAM" for x in check_interactions(["clobazam"], {"THC"}))
    assert any(x.rule_id == "DDI-WARFARIN" for x in check_interactions(["warfarin"], {"THC"}))
    assert any(x.rule_id == "POLYPHARMACY" for x in check_interactions(list("abcde"), {"CBD"}))


def test_age_under_25_is_consideration_not_exclusion():
    flags = {f.code: f.level for f in screen_contraindications(PatientRiskProfile(age=22))}
    assert flags == {"AGE_UNDER_25": "consider"}
    assert screen_contraindications(PatientRiskProfile(age=10))[0].level == "stop"


def test_hemp_rules_switch_on_dec_11_2026():
    cfg = load_jurisdictions()
    args = dict(delta9_pct=0.2, total_thc_pct=0.5, container_total_thc_mg=12.0, contains_synthesized=False)
    assert check_hemp_product(cfg, date(2026, 12, 10), **args).compliant
    v = check_hemp_product(cfg, date(2026, 12, 11), **args)
    assert not v.compliant and len(v.reasons) == 2
    assert not check_hemp_product(cfg, date(2026, 11, 13), 0.1, 0.1, 0.1, True).compliant


def test_regulatory_diff_classifies():
    ch = diff_regulation("x", "Labels shall state THC.", "Labels shall state THC.\nChild-resistant packaging required.")
    assert ch.added and "packaging_labeling" in ch.operational_areas


# ------------------------------------------------------------------ ethics
def test_lexicographic_safety_beats_profit():
    a = Proposal("a", "x", "safe", scores={"safety": 0.9, "profitability": 0.1})
    b = Proposal("b", "x", "lucrative", scores={"safety": 0.5, "profitability": 1.0})
    assert LexicographicPolicy().choose([a, b]) is a


def test_lexicographic_tolerance_lets_efficiency_break_ties():
    a = Proposal("a", "x", "", scores={"safety": 1, "ethics": 1, "accuracy": 0.99, "efficiency": 0.2})
    b = Proposal("b", "x", "", scores={"safety": 1, "ethics": 1, "accuracy": 1.00, "efficiency": 0.9})
    c = Proposal("c", "x", "", scores={"safety": 1, "ethics": 1, "accuracy": 1.00, "efficiency": 0.1})
    assert LexicographicPolicy().choose([a, b, c]) is b


# ------------------------------------------------------------------ swarm
def test_human_only_tasks_are_refused():
    p = Proposal("rogue", "crop_destruction", "destroy room F1", scores={"safety": 0.5})
    r = run(Swarm([OneShot(p)], default_critics(False)).cycle(ctx()))
    assert r.dispositions[0].outcome == "refused"


def test_agents_cannot_originate_doses():
    p = Proposal("rogue", "titration_step", "increase", {"dose_mg": 10}, uncertainty="x", evidence=["e"])
    r = run(Swarm([OneShot(p)], default_critics(False)).cycle(ctx()))
    assert r.dispositions[0].outcome == "vetoed"


def test_clinical_output_requires_uncertainty():
    p = Proposal("a", "product_recommendation", "use X", evidence=["e"])
    assert run(Swarm([OneShot(p)], default_critics(False)).cycle(ctx())).dispositions[0].outcome == "vetoed"


def test_negative_safety_is_vetoed():
    p = Proposal("a", "climate_setpoint", "cheap but risky", scores={"safety": -0.2, "profitability": 1})
    assert run(Swarm([OneShot(p)], default_critics(False)).cycle(ctx())).dispositions[0].outcome == "vetoed"


def test_critic_warning_escalates_automatable_action():
    p = Proposal("a", "climate_setpoint", "adjust", {"inputs_calibrated": False})
    c = ctx()
    r = run(Swarm([OneShot(p)], default_critics(False)).cycle(c))
    assert r.dispositions[0].outcome == "queued" and c.approvals.pending()


def test_agent_crash_is_isolated_and_reported():
    p = Proposal("ok", "data_logging", "log")
    c = ctx()
    r = run(Swarm([Crash(), OneShot(p)], default_critics(False)).cycle(c))
    assert "crasher" in r.agent_failures and r.dispositions[0].outcome == "executed"
    assert any(a["source"] == "crasher" for a in c.alerts)


def test_every_disposition_is_logged():
    c = ctx()
    run(Swarm([OneShot(Proposal("a", "data_logging", "x"), Proposal("b", "crop_destruction", "y"))],
              default_critics(False)).cycle(c))
    assert {d["disposition"] for d in c.store.all("decision")} == {"executed", "refused"}


# ------------------------------------------------------------------ clinical end to end
def test_full_cycle_clinical_and_operations():
    ent = IntelligentEnterprise(use_llm=False)
    world = build_world(ent)
    r = run(ent.run_cycle(world, now=NOW))
    out = r.by_outcome()
    assert not r.agent_failures
    tasks_executed = {d.proposal.task for k in ("executed", "executed_with_alarm", "executed_notify") for d in out[k]}
    assert {"interaction_alert", "safety_alert", "coa_audit", "batch_hold", "sensor_failsafe"} <= tasks_executed
    # Nothing clinical-decisional executed without a clinician
    assert not tasks_executed & {"product_recommendation", "titration_step", "dose_selection", "harvest_timing",
                                 "regulatory_interpretation", "batch_release"}
    reports = {d.proposal.payload["patient_id"]: d.proposal for d in r.dispositions
               if d.proposal.task == "consultation_report"}
    by_ind = {ent.patients[pid]["indication"]: p for pid, p in reports.items()}
    assert "out_of_scope" in by_ind["dravet"].summary
    assert "evidence_insufficient" in by_ind["ptsd"].summary
    assert "trial_reasonable" in by_ind["chronic_pain_adult"].summary
    assert ent.state.batch_status["B-405"] == "hold" and ent.store.verify_chain()[0]


def test_titration_step_needs_the_treating_clinician():
    ent = IntelligentEnterprise(use_llm=False)
    world = build_world(ent)
    run(ent.run_cycle(world, now=NOW))
    req = next(r for r in ent.approvals.pending() if r.task == "titration_step")
    with pytest.raises(PermissionError):
        ent.decide(req.request_id, True, "grower", "head_grower")
    ent.decide(req.request_id, False, "Dr. C", "treating_clinician", "stopping instead")
    assert ent.approvals.overrides and ent.store.all("approval")[0]["status"] == "rejected"


def test_clinician_approval_advances_plan():
    ent = IntelligentEnterprise(use_llm=False)
    pid = ent.enroll_patient("x", {"age_band": "45-54", "indication": "chronic_pain_adult", "consent_version": "v3"},
                             {"indication": "chronic_pain_adult", "profile": PatientRiskProfile(age=50), "logs": []})
    start = NOW - timedelta(days=5)
    ent.set_titration_plan(pid, TitrationPlan(pid, "Dr", ["s1", "s2"], 72, 35, 2.0, start), "treating_clinician")
    run(ent.run_cycle({}, now=NOW))
    req = next(r for r in ent.approvals.pending() if r.task == "titration_step")
    assert req.payload["action"] == "advance"
    ent.decide(req.request_id, True, "Dr", "treating_clinician", now=NOW)
    assert ent.patients[pid]["plan"].current_step == 1
    with pytest.raises(PermissionError):
        ent.set_titration_plan(pid, TitrationPlan(pid, "x", ["a"], 1, 1, 1, NOW), "care_coordinator")


def test_moe_blocking_opinion_cannot_be_outvoted():
    ent = IntelligentEnterprise(use_llm=False)
    c = ent.context({})
    agent = ConsultationAgent()
    q = {"indication": "chronic_pain_adult", "standard_care_tried": True,
         "profile": PatientRiskProfile(age=40, pregnant_or_lactating=True)}
    res = run(agent.moe.consult(q, c))
    assert agent.moe.answer(res, "trial") == "do_not_initiate"


# ------------------------------------------------------------------ enterprise
def test_deployment_gate_and_rollback():
    card = ModelCard("dr", "1", "owner", "snap", {}, "adult chronic pain", ["pediatric"], {"auc": 0.7}, 0.9)
    assert deployment_gate(card, clinical=True).passed
    bad = ModelCard("dr", "2", "owner", "snap", {}, "", [], None, 0.6, rule_based_interactions=False)
    assert len(deployment_gate(bad, clinical=True).failures) == 4
    reg = ModelRegistry()
    reg.register(card)
    reg.register(ModelCard("dr", "3", "owner", "snap", {}, "s", ["x"], {"auc": 0.72}, 0.9))
    reg.deploy("dr", "1", "approver", True)
    reg.deploy("dr", "3", "approver", True)
    assert reg.rollback("dr").version == "1"


def test_maturity_is_sequential():
    lvl, _, _ = assess_maturity(True, False, True, True, True, True, True, True)
    assert lvl == 1   # cannot claim predictive capability on ungoverned data


# ------------------------------------------------------------------ swarm-review regressions (agents)
@pytest.mark.parametrize("payload", [{"advice": {"dose_mg": 5}}, {"dose_mg_thc": 5}, {"next_dose_mg": 10},
                                     {"steps": [{"starting_dose_mg": 2.5}]}])
def test_dose_numbers_are_vetoed_at_any_depth_and_name(payload):
    p = Proposal("rogue", "patient_reminder", "remind", payload)
    assert run(Swarm([OneShot(p)], default_critics(False)).cycle(ctx())).dispositions[0].outcome == "vetoed"


def test_self_declared_clinician_agent_cannot_originate_doses():
    p = Proposal("clinician", "titration_step", "increase", {"dose_mg": 10}, uncertainty="x", evidence=["e"])
    assert run(Swarm([OneShot(p)], default_critics(False)).cycle(ctx())).dispositions[0].outcome == "vetoed"


def test_failing_actuator_alarms_and_still_logs():
    def boom(p, c):
        raise ConnectionError("pager offline")
    p = Proposal("ok", "data_logging", "log")
    c = ctx()
    r = run(Swarm([OneShot(p)], default_critics(False), actuators={"data_logging": boom}).cycle(c))
    assert r.dispositions[0].outcome == "actuation_failed"
    assert any(a["level"] == "alarm" for a in c.alerts)
    assert c.store.verify_chain()[0] and {d["disposition"] for d in c.store.all("decision")} == {"actuation_failed"}


def test_crashed_safety_expert_blocks_product_recommendation():
    """A safety screen that crashes has not cleared the patient (fail closed)."""
    ent = IntelligentEnterprise(use_llm=False)
    pt = {"status": "new_consult", "indication": "chronic_pain_adult", "standard_care_tried": True,
          "profile": PatientRiskProfile(age=None, pregnant_or_lactating=True)}
    from intelligent_plant.simulation import make_coa
    cat = [{"product_id": "PR-OIL-20", "form": "tincture", "route": "oral", "unit_mg_thc": 0.5, "unit_mg_cbd": 10.0,
            "coa": make_coa("C-101F", "B-101", 0.02, 0.0, 1.1, 0.0, form="tincture", finished=True),
            "batch_cv": 0.05, "available_months": 12}]
    out = run(ConsultationAgent().propose(ent.context({"patients": {"P1": pt}, "product_catalog": cat})))
    report = next(p for p in out if p.task == "consultation_report")
    assert any("unassessed" in f for f in report.payload["flags"])
    assert not any(p.task == "product_recommendation" for p in out)


def test_one_bad_patient_record_does_not_silence_others():
    from intelligent_plant.agents.clinical import TitrationMonitorAgent
    plan = TitrationPlan("A", "dr", ["clinician step 1"], 24, 42, 2.0, NOW)
    good_logs = [{"timestamp": "2026-11-17T14:00:00+00:00", "route": "oral", "product_id": "X", "dose_units": 1},
                 {"timestamp": "2026-11-17T14:45:00+00:00", "route": "oral", "product_id": "X", "dose_units": 1,
                  "adverse_effects": [{"name": "dizziness", "severity": "Severe"}]}]
    pts = {"A": {"plan": plan, "logs": good_logs, "profile": PatientRiskProfile(age=40)},
           "B": {"plan": plan, "logs": [{"timestamp": "2026-11-2 10:00"}, {"timestamp": "bad"}],
                 "profile": PatientRiskProfile(age=40)}}
    out = run(TitrationMonitorAgent().propose(ent_ctx({"patients": pts})))
    a_alerts = {p.payload.get("patient_event") for p in out if p.payload.get("patient_id") == "A"}
    assert {"oral_redose", "adverse_effect"} <= a_alerts            # "Severe" (capitalised) still counts
    assert any(p.payload.get("patient_id") == "B" and p.task == "safety_alert" for p in out)


def ent_ctx(obs):
    return IntelligentEnterprise(use_llm=False).context(obs)


def test_moe_zero_confidence_does_not_crash_or_drop_flags():
    from intelligent_plant.agents.moe import Expert, ExpertOpinion, MixtureOfExperts

    class Zero(Expert):
        name, keywords = "zero", ("x",)

        async def opine(self, q, c):
            return ExpertOpinion(self.name, "yes", 0.0, ["[STOP] keep me"], question="q")
    moe = MixtureOfExperts([Zero()])
    res = run(moe.consult({"a": "x"}, ctx()))
    assert res.answers["q"] == ("yes", 0.0) and "[STOP] keep me" in res.flags
    assert MixtureOfExperts([]).gate({}) == {}
    with pytest.raises(ValueError):
        MixtureOfExperts([Zero()], temperature=0)


def test_parse_json_object_takes_first_object():
    from intelligent_plant.agents.llm import parse_json_object
    assert parse_json_object('ok {"severity": "warn", "message": "a"} and {"b": 1}') == {"severity": "warn",
                                                                                         "message": "a"}
