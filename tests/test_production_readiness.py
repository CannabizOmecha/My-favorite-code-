"""Production readiness: calibration recovers known physics, reviews bind to exact content, and the
system refuses to run for real patients until people have done the reviews."""
import asyncio
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from intelligent_plant.__main__ import main as cli
from intelligent_plant.agents.base import Context, Proposal
from intelligent_plant.agents.swarm import LLMJuryCritic
from intelligent_plant.calibration.facility import (CalibrationError, calibrate, check_nutrient_ranges,
                                                    fit_room_model, fit_volatile_loss, read_csv, ROOM_COLUMNS)
from intelligent_plant.data.privacy import PseudonymVault, key_from_env, scrub_identifiers
from intelligent_plant.data.store import RecordStore
from intelligent_plant.governance.autonomy import ApprovalQueue, ApprovalRequest
from intelligent_plant.governance.readiness import (REQUIRED, check_readiness, default_artifacts,
                                                    record_signoff)
from intelligent_plant.governance.registry import ModelRegistry
from intelligent_plant.orchestrator import IntelligentEnterprise, ProductionNotReady
from intelligent_plant.science.chemistry import R_GAS
from intelligent_plant.science.environment import RoomModel

KEY = bytes(range(32))
TRUE_ROOM = RoomModel(heat_gain_c=0.5, cool_gain_c=-1.8, dehum_heat_c=0.25, dehum_rh=-5.0, humid_rh=4.0,
                      cool_rh_per_c=-2.0, transpiration_rh=1.2)


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def room_log(n=600, seed=0, model=TRUE_ROOM, vary=("cool", "dehum", "humid")):
    """A realistic log: a simple thermostat/humidistat holds the room near setpoint (closed loop), with the
    natural on/off variation real equipment shows. Nothing saturates or is clamped: clamped readings would
    bias any identification toward zero effect."""
    rng = np.random.default_rng(seed)
    T, H, rows = 25.0, 55.0, []
    for _ in range(n):
        p_cool = 0.85 if T > 25 else 0.15
        p_dehum = 0.85 if H > 55 else 0.15
        u = [float(rng.random() < p_cool), float(rng.random() < p_dehum), 0.0]
        if not u[1] and H < 55:
            u[2] = float(rng.random() < 0.7)
        u = [x if a in vary else 0.0 for x, a in zip(u, ("cool", "dehum", "humid"))]
        Tn, Hn = model.step(T, H, tuple(u))
        Tn += rng.normal(0, 0.05); Hn += rng.normal(0, 0.3)       # sensor noise
        rows.append({"temp_air": T, "rh": H, "cool": u[0], "dehum": u[1], "humid": u[2],
                     "temp_air_next": Tn, "rh_next": Hn})
        T, H = Tn, Hn
    return rows


def volatile_runs(Ea=60_000.0, A=None, temps=(95, 105, 115), seed=1):
    rng = np.random.default_rng(seed)
    A = A or 2e-4 / math.exp(-Ea / (R_GAS * (105 + 273.15)))
    rows = []
    for T in temps:
        k = A * math.exp(-Ea / (R_GAS * (T + 273.15)))
        for t in (600, 1200, 2400):
            rows.append({"temp_c": T, "time_s": t, "retention": math.exp(-k * t) * rng.normal(1, 0.01)})
    return rows


# ------------------------------------------------------------------ calibration
def test_room_model_recovers_known_physics(tmp_path):
    rows, rejected, sha = read_csv(write_csv(tmp_path / "room.csv", room_log()), ROOM_COLUMNS)
    model, stats = fit_room_model(rows)
    for field in ("heat_gain_c", "cool_gain_c", "dehum_heat_c", "dehum_rh", "humid_rh", "transpiration_rh"):
        assert getattr(model, field) == pytest.approx(getattr(TRUE_ROOM, field), abs=0.1), field
    assert model.cool_rh_per_c == pytest.approx(TRUE_ROOM.cool_rh_per_c, abs=0.3)
    assert stats.r2_temp > 0.9 and stats.holdout_rmse_temp_c < 0.1 and not rejected and len(sha) == 64


def test_room_model_refuses_unidentifiable_actuator():
    with pytest.raises(CalibrationError, match="humid"):
        fit_room_model(room_log(vary=("cool", "dehum")))


def test_room_model_refuses_too_little_data():
    with pytest.raises(CalibrationError, match="need at least"):
        fit_room_model(room_log(n=50))


def test_room_model_refuses_physically_wrong_sign():
    wrong = RoomModel(cool_gain_c=+0.8)          # e.g. actuator columns swapped in the export
    with pytest.raises(CalibrationError, match="cooling"):
        fit_room_model(room_log(model=wrong))


def test_room_model_refuses_stuck_sensor():
    rows = room_log()
    for r in rows[100:250]:
        r["temp_air_next"] = r["temp_air"] = 22.0          # sensor frozen at one value
    with pytest.raises(CalibrationError, match="stuck"):
        fit_room_model(rows)


def test_bad_rows_are_rejected_at_entry(tmp_path):
    rows = room_log(n=10)
    rows[3]["rh"] = 140
    rows[4]["temp_air"] = "n/a"
    good, rejected, _ = read_csv(write_csv(tmp_path / "r.csv", rows), ROOM_COLUMNS)
    assert len(good) == 8 and len(rejected) == 2


def test_volatile_fit_recovers_ea_and_reports_direction():
    low = fit_volatile_loss(volatile_runs(Ea=60_000))
    assert low.Ea_J == pytest.approx(60_000, rel=0.05) and "hotter-and-shorter" in low.direction
    high = fit_volatile_loss(volatile_runs(Ea=120_000))
    assert "cooler-and-longer" in high.direction


def test_volatile_fit_needs_three_temperatures():
    with pytest.raises(CalibrationError, match="at least 3"):
        fit_volatile_loss(volatile_runs(temps=(95, 115)))


def test_nutrient_check_flags_sop_disagreement():
    rng = np.random.default_rng(0)
    rows = [{"stage": "flowering", "ec": rng.normal(2.7, 0.1), "ph": 6.0, "met_quality_target": 1.0}
            for _ in range(30)]
    chk = check_nutrient_ranges(rows, {"flowering": {"ec": [1.6, 2.4], "ph": [5.6, 6.4]}})[0]
    assert chk.share_good_inside_sop < 0.2 and "discuss with the head grower" in chk.note


def test_calibration_is_applied_to_agents(tmp_path):
    cal = calibrate("Test Farm", write_csv(tmp_path / "room.csv", room_log()),
                    write_csv(tmp_path / "vol.csv", volatile_runs()), None,
                    {"flowering": {"ec": [1.8, 2.2], "ph": [5.8, 6.2]}})
    cal.write(tmp_path / "cal.json")
    ent = IntelligentEnterprise(use_llm=False, calibration_path=str(tmp_path / "cal.json"))
    agents = {a.name: a for a in ent.swarm.agents}
    assert agents["climate_controller"].mpc.model.cool_gain_c == pytest.approx(-1.8, abs=0.1)
    assert agents["decarb_planner"].calibrated
    assert agents["nutrients"].ec_range["flowering"] == (1.8, 2.2)


def test_cli_calibrate_refuses_with_exit_code(tmp_path, capsys):
    path = write_csv(tmp_path / "room.csv", room_log(n=40))
    assert cli(["calibrate", "--facility", "F", "--room-log", str(path), "--out", str(tmp_path / "c.json")]) == 2
    assert "CALIBRATION REFUSED" in capsys.readouterr().out and not (tmp_path / "c.json").exists()


# ------------------------------------------------------------------ privacy fixes
def test_persistent_vault_requires_stable_key(tmp_path):
    with pytest.raises(ValueError, match="stable key"):
        PseudonymVault(str(tmp_path / "v.db"))


def test_same_patient_keeps_same_id_across_restart(tmp_path):
    p1 = PseudonymVault(str(tmp_path / "v.db"), key=KEY).pseudonymize("Jane Doe")
    p2 = PseudonymVault(str(tmp_path / "v.db"), key=KEY).pseudonymize("Jane Doe")
    assert p1 == p2


def test_wrong_vault_key_is_detected(tmp_path):
    PseudonymVault(str(tmp_path / "v.db"), key=KEY)
    with pytest.raises(ValueError, match="does not match"):
        PseudonymVault(str(tmp_path / "v.db"), key=bytes(range(1, 33)))


def test_key_from_env(monkeypatch):
    monkeypatch.delenv("INTELLIGENT_PLANT_VAULT_KEY", raising=False)
    with pytest.raises(ValueError):
        key_from_env()
    monkeypatch.setenv("INTELLIGENT_PLANT_VAULT_KEY", KEY.hex())
    assert key_from_env() == KEY
    monkeypatch.setenv("INTELLIGENT_PLANT_VAULT_KEY", "abcd")
    with pytest.raises(ValueError, match="32 bytes"):
        key_from_env()


class RecordingLLM:
    available = True

    def __init__(self):
        self.prompts = []

    async def complete_json(self, system, prompt):
        self.prompts.append(prompt)
        return {"severity": "none"}


def _ctx(llm, phi_ok):
    return Context(RecordStore(), ApprovalQueue(), ModelRegistry(), {"privacy": {"phi_to_llm_permitted": phi_ok}},
                   llm)


def test_patient_data_never_sent_to_llm_without_baa():
    llm = RecordingLLM()
    p = Proposal("consultation", "product_recommendation", "Patient P-0123456789abcdef: try X",
                 {"patient_id": "P-0123456789abcdef", "medications": ["warfarin"]}, evidence=["e"], uncertainty="u")
    asyncio.run(LLMJuryCritic("skeptic").critique(p, _ctx(llm, False)))
    assert llm.prompts == []
    asyncio.run(LLMJuryCritic("skeptic").critique(p, _ctx(llm, True)))
    sent = llm.prompts[0]
    assert "P-0123456789abcdef" not in sent and "warfarin" not in sent and "[patient]" in sent


def test_scrub_identifiers():
    assert scrub_identifiers("Patient P-0123456789abcdef and P-fedcba9876543210") == "Patient [patient] and [patient]"


def test_pending_decisions_survive_restart(tmp_path):
    db = str(tmp_path / "r.db")
    q = ApprovalQueue(RecordStore(db))
    req = q.submit(ApprovalRequest("prop1", "titration_step", "treating_clinician", "hold", {"patient_id": "P"}, [], []))
    q2 = ApprovalQueue(RecordStore(db))
    assert [r.request_id for r in q2.pending()] == [req.request_id]
    q2.decide(req.request_id, False, "Dr", "treating_clinician", "stop")
    q3 = ApprovalQueue(RecordStore(db))
    assert not q3.pending() and q3.overrides and RecordStore(db).verify_chain()[0]
    with pytest.raises(ValueError, match="already"):
        q3.decide(req.request_id, True, "Dr", "treating_clinician")


# ------------------------------------------------------------------ sign-offs and production gate
def _all_signed(tmp_path, cal_path, reg_path):
    arts = default_artifacts(cal_path, reg_path)
    so = tmp_path / "signoffs.json"
    for r in REQUIRED:
        record_signoff(so, r.item, f"Reviewer for {r.role}", r.role, "Reviewed in full.", arts)
    return so, arts


def _calibration(tmp_path):
    cal = calibrate("Test Farm", write_csv(tmp_path / "room.csv", room_log()), None, None,
                    {"flowering": {"ec": [1.6, 2.4], "ph": [5.6, 6.4]}})
    cal.write(tmp_path / "cal.json")
    return tmp_path / "cal.json"


def _reg_copy(tmp_path):
    dst = tmp_path / "REGULATORY_REVIEW.md"
    shutil.copy(Path(__file__).resolve().parents[1] / "docs" / "REGULATORY_REVIEW.md", dst)
    return dst


def test_signoff_binds_to_exact_content(tmp_path):
    cal, reg = _calibration(tmp_path), _reg_copy(tmp_path)
    so, arts = _all_signed(tmp_path, cal, reg)
    assert check_readiness(so, arts).ready
    reg.write_text(reg.read_text() + "\nA later edit.\n")
    rep = check_readiness(so, arts)
    stale = {i.item for i in rep.items if i.status == "stale"}
    assert not rep.ready and stale == {"regulatory_legal", "privacy_security"}


def test_json_comment_or_status_edit_does_not_invalidate(tmp_path):
    cal, reg = _calibration(tmp_path), _reg_copy(tmp_path)
    so, arts = _all_signed(tmp_path, cal, reg)
    doc = json.loads(cal.read_text()); doc["_comment"] = "reviewed"; doc["status"] = "approved"
    cal.write_text(json.dumps(doc, indent=4))
    assert check_readiness(so, arts).ready
    doc["nutrient_ranges"]["flowering"]["ec"] = [1.0, 3.0]
    cal.write_text(json.dumps(doc))
    assert not check_readiness(so, arts).ready


def test_signoff_requires_the_accountable_role(tmp_path):
    with pytest.raises(PermissionError):
        record_signoff(tmp_path / "s.json", "patient_messages_clinical", "Bob", "head_grower", "ok",
                       default_artifacts())
    with pytest.raises(ValueError):
        record_signoff(tmp_path / "s.json", "patient_messages_clinical", " ", "clinical_lead", "ok",
                       default_artifacts())


def test_production_refuses_until_everything_is_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELLIGENT_PLANT_VAULT_KEY", KEY.hex())
    with pytest.raises(ProductionNotReady, match="persistent"):
        IntelligentEnterprise(mode="production", signoffs_path=str(tmp_path / "s.json"))
    cal, reg = _calibration(tmp_path), _reg_copy(tmp_path)
    kw = dict(mode="production", db_path=str(tmp_path / "r.db"), vault_path=str(tmp_path / "v.db"),
              calibration_path=str(cal), regulatory_review_path=str(reg), use_llm=False)
    with pytest.raises(ProductionNotReady) as e:
        IntelligentEnterprise(signoffs_path=str(tmp_path / "none.json"), **kw)
    assert not e.value.report.ready
    so, _ = _all_signed(tmp_path, cal, reg)
    ent = IntelligentEnterprise(signoffs_path=str(so), **kw)
    assert ent.mode == "production" and ent.readiness.ready
    monkeypatch.delenv("INTELLIGENT_PLANT_VAULT_KEY")
    with pytest.raises(ProductionNotReady, match="VAULT_KEY"):
        IntelligentEnterprise(signoffs_path=str(so), **kw)


def test_review_pack_contains_every_message(tmp_path):
    assert cli(["review-pack", "--out", str(tmp_path)]) == 0
    clin = (tmp_path / "patient_message_review_clinical.md").read_text()
    pat = (tmp_path / "patient_message_review_patients.md").read_text()
    for key in ("oral_redose", "adverse_effect", "medication_check", "progress_summary"):
        assert f"`{key}`" in clin
    assert "In your own words" in pat and "fingerprint" in clin
