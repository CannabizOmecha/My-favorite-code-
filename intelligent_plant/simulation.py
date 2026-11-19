"""A SYNTHETIC seed-to-patient world for demonstration and testing. Every
number here is simulated; none is an empirical result."""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .agents.clinical import TitrationPlan
from .data.store import Provenance
from .knowledge.interactions import PatientRiskProfile
from .science.breeding import CandidateCultivar
from .science.environment import vpd_air
from .science.quality import COA

NOW = datetime(2026, 11, 19, 14, 0, tzinfo=timezone.utc)
PASS_PANELS = {k: "pass" for k in ("pesticides", "residual_solvents", "heavy_metals", "mycotoxins", "microbials")}


def make_coa(coa_id: str, batch_id: str, thc: float, thca: float, cbd: float, cbda: float, *,
             form: str = "flower", finished: bool = False, iso: bool = True, method: str = "HPLC",
             panels: dict | None = None) -> COA:
    return COA(coa_id, batch_id, "LAB-1", iso, method, thc, thca, cbd, cbda, date(2026, 11, 1),
               packaging_date=date(2026, 11, 10), lod={"thc": 0.01}, loq={"thc": 0.03}, uncertainty_pct=5.0,
               sample_plan="ASTM D8334 composite", moisture_pct=11.0, dry_weight_basis=True,
               contaminants=PASS_PANELS if panels is None else panels, product_form=form, finished_product=finished)


def oligogenic_trait(n: int = 300, p: int = 400, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.binomial(2, 0.5, (n, p)).astype(float)
    g = X[:, [3, 50, 120]] @ np.array([1.0, 0.8, 0.6])
    y = g + rng.normal(0, g.std(), n)       # h2 ~ 0.5
    return X, y


def polygenic_trait(n: int = 300, p: int = 400, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.binomial(2, 0.5, (n, p)).astype(float)
    g = X[:, :200] @ rng.normal(0, 0.1, 200)
    y = g + rng.normal(0, g.std(), n)
    return X, y


def seed_registry_records(ent, n_patients: int = 12, seed: int = 7) -> None:
    """Populate the governed store so traceability and outcome learning have a
    full chain: genotype -> environment -> batch -> COA -> product -> admin -> outcome."""
    rng = random.Random(seed)
    prov = Provenance(source="simulation", operator="sim", instrument="SIM", calibrated=True)
    s = ent.store
    s.append("genotype", {"genotype_id": "G-CBD1", "cultivar_name": "Fairland CBD-20", "synthase_genotype": "BdBd",
                          "marker_panel_version": "v2", "source": "in-house"}, prov)
    for h in range(6):
        T, rh = 25 + rng.uniform(-1, 1), 58 + rng.uniform(-3, 3)
        s.append("environment_log", {"room_id": "F1", "timestamp": (NOW - timedelta(hours=h)).isoformat(),
                                     "temp_air": T, "temp_leaf": T - 2, "rh": rh, "co2": 900, "ppfd": 900,
                                     "vpd_air": vpd_air(T, rh), "sensor_cal_date": "2026-10-01"}, prov)
    s.append("batch", {"batch_id": "B-101", "genotype_id": "G-CBD1", "room_id": "F1", "start_date": "2026-07-01",
                       "harvest_date": "2026-09-15", "yield_g": 41000, "dli_cumulative": 2600}, prov)
    s.append("coa", {"coa_id": "C-101", "batch_id": "B-101", "lab_id": "LAB-1", "method": "HPLC", "thc": 0.2,
                     "thca": 0.6, "cbd": 1.0, "cbda": 14.0, "test_date": "2026-09-20"}, prov)
    s.append("product", {"product_id": "PR-OIL-20", "batch_id": "B-101", "coa_id": "C-101", "form": "tincture",
                         "route": "oral", "unit_mg_thc": 0.5, "unit_mg_cbd": 10.0, "lot": "L-101"}, prov)
    for i in range(n_patients):
        pid = ent.vault.pseudonymize(f"registry-patient-{i}")
        s.append("patient", {"patient_id": pid, "age_band": "45-54", "indication": "chronic_pain_adult",
                             "consent_version": "v3"}, prov)
        for k in range(4):
            aid = f"A-{i}-{k}"
            s.append("administration", {"admin_id": aid, "patient_id": pid, "product_id": "PR-OIL-20",
                                        "coa_id": "C-101", "timestamp": (NOW - timedelta(days=k)).isoformat(),
                                        "dose_units": 1, "dose_mg_thc": 0.5, "dose_mg_cbd": 10.0, "route": "oral"}, prov)
            pre = rng.uniform(5, 8)
            s.append("outcome", {"outcome_id": f"O-{i}-{k}", "admin_id": aid, "instrument": "BPI-severity",
                                 "score_pre": round(pre, 1), "score_post": round(max(0, pre - rng.gauss(1.2, 1)), 1),
                                 "interval_min": 180, "adverse_events_json": []}, prov)


def patient_logs(n: int, start: datetime, *, oral_redose_at: int | None = None,
                 ae_at: int | None = None, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    logs, t = [], start
    for i in range(n):
        t = t + timedelta(hours=24)
        ts = t if oral_redose_at != i else logs[-1]["_t"] + timedelta(minutes=45)
        pre = rng.uniform(5.5, 7.5)
        logs.append({"_t": ts, "timestamp": ts.isoformat(), "product_id": "PR-OIL-20", "batch_lot": "L-101",
                     "total_thc_unit": 0.5, "total_cbd_unit": 10.0, "dose_units": 1, "dose_mg_thc": 0.5,
                     "dose_mg_cbd": 10.0, "route": "oral", "score_pre": round(pre, 1),
                     "score_post": round(pre - rng.gauss(0.8, 0.6), 1), "interval_min": 180,
                     "adverse_effects": ([{"name": "dizziness", "severity": "moderate"}] if ae_at == i else []),
                     "sleep": 6, "function": 5})
    for l in logs:
        l.pop("_t")
    return logs


def build_world(ent) -> dict:
    """Observations for one cycle across all domains."""
    seed_registry_records(ent)
    # Patients (identities go straight into the vault; agents see pseudonyms only)
    p_new = ent.enroll_patient("Jane Example, DOB 1971-03-02",
                               {"age_band": "55-64", "indication": "chronic_pain_adult", "consent_version": "v3"},
                               {"status": "new_consult", "indication": "chronic_pain_adult", "standard_care_tried": True,
                                "onset_need": "sustained", "food_relation": "variable",
                                "profile": PatientRiskProfile(age=55, medications=["warfarin", "lorazepam"],
                                                              cyp2c9_genotype="*1/*3")})
    p_young = ent.enroll_patient("Sam Example, DOB 2003-06-10",
                                 {"age_band": "18-24", "indication": "ptsd", "consent_version": "v3"},
                                 {"status": "new_consult", "indication": "ptsd", "standard_care_tried": False,
                                  "profile": PatientRiskProfile(age=23, psychosis_history_family_strong=True)})
    p_peds = ent.enroll_patient("Child Example",
                                {"age_band": "0-17", "indication": "dravet", "consent_version": "v3"},
                                {"status": "new_consult", "indication": "dravet",
                                 "profile": PatientRiskProfile(age=9, medications=["clobazam", "valproate"])})
    p_tit = ent.enroll_patient("Alex Example, DOB 1960-01-01",
                               {"age_band": "65-74", "indication": "chronic_pain_adult", "consent_version": "v3"},
                               {"indication": "chronic_pain_adult", "meds_changed": True,
                                "profile": PatientRiskProfile(age=66, medications=["clobazam"]),
                                "logs": patient_logs(12, NOW - timedelta(days=13), oral_redose_at=11, ae_at=11)})
    ent.set_titration_plan(p_tit, TitrationPlan(
        p_tit, "Dr. Clinician", steps=["Step 1 as prescribed", "Step 2 as prescribed", "Step 3 as prescribed"],
        hold_hours=72, review_after_days=35, benefit_threshold=2.0, start=NOW - timedelta(days=13)),
        clinician_role="treating_clinician")

    catalog = [
        {"product_id": "PR-OIL-20", "form": "tincture", "route": "oral", "unit_mg_thc": 0.5, "unit_mg_cbd": 10.0,
         "coa": make_coa("C-101F", "B-101", 0.02, 0.0, 1.1, 0.0, form="tincture", finished=True), "batch_cv": 0.05,
         "available_months": 12},
        {"product_id": "PR-GUMMY-10", "form": "edible", "route": "oral", "unit_mg_thc": 10.0, "unit_mg_cbd": 0.0,
         "coa": make_coa("C-202", "B-202", 0.3, 0.0, 0.0, 0.0, form="edible", finished=False), "batch_cv": 0.2,
         "available_months": 6},
        {"product_id": "PR-FLOWER-HI", "form": "flower", "route": "inhaled", "unit_mg_thc": 200.0, "unit_mg_cbd": 1.0,
         "coa": make_coa("C-303", "B-303", 1.0, 26.0, 0.1, 0.1), "batch_cv": 0.08, "available_months": 3},
    ]
    breeding = {
        "traits": {"chemotype_ratio": dict(zip(("X", "y"), oligogenic_trait())),
                   "yield": dict(zip(("X", "y"), polygenic_trait()))},
        "candidates": [CandidateCultivar("Fairland CBD-20", 20.5, 0.06, 0.9, 0.8, 0.7, 0.9, 1.2),
                       CandidateCultivar("Prairie 20:1", 19.0, 0.12, 0.8, 0.9, 0.9, 1.1, 1.0),
                       CandidateCultivar("Plains High-THC", 0.03, 0.08, 0.7, 0.6, 0.6, 1.4, 1.0)],
        "target_ratio": 20.0, "kinship_mean": 0.015,
        "seedling_sex_calls": [{"id": "S-17", "call": "male", "confidence": 0.998}],
    }
    rooms = {
        "F1": {"stage": "flowering", "photoperiod_h": 12, "price_forecast": [0.09, 0.21, 0.24],
               "reading": {"temp_air": 27.5, "temp_leaf": 25.0, "rh": 48.0, "co2": 1000, "ppfd": 1100,
                           "sensor_cal_due": "2027-01-01"},
               "substrate_moisture": 38, "target_moisture": 45, "ec": 2.8, "ph": 6.1,
               "detections": [{"label": "powdery_mildew", "p": 0.81, "validated_domain": False}],
               "days_in_flower": 58, "expected_flower_days": 60, "known_good": {"T": 25.0, "rh": 55.0}},
        "V2": {"stage": "vegetative", "photoperiod_h": 18, "price_forecast": [0.12],
               "reading": {"temp_air": 26.0, "rh": 61.0, "ppfd": 600, "sensor_cal_due": "2026-10-30"},
               "known_good": {"T": 26.0, "rh": 62.0}},
    }
    return {
        "rooms": rooms,
        "instruments": [{"instrument_id": "RH-V2-01", "next_due": "2026-10-30"},
                        {"instrument_id": "PAR-F1-02", "next_due": "2027-02-01"}],
        "breeding": breeding,
        "drying_rooms": {"D1": {"temp": 23.5, "rh": 50.0, "water_activity": 0.52}},
        "decarb_requests": [{"batch_id": "B-101", "acid": "CBDA", "target_conversion": 0.95}],
        "coas": [make_coa("C-404", "B-404", 0.5, 22.0, 0.1, 0.1),
                 make_coa("C-405", "B-405", 0.4, 19.0, 0.1, 0.2, method="GC",
                          panels={**PASS_PANELS, "pesticides": "fail"})],
        "market_total_thc": [18 + random.Random(i).uniform(-4, 4) for i in range(40)],
        "batch_potency": {"Plains High-THC": [21.0, 20.6, 21.4, 20.9, 21.2, 20.8, 17.1, 23.9]},
        "label_claims": [{"product_id": "PR-GUMMY-10", "form": "edible", "P_pct": 22.0, "mass_mg": 28000,
                          "eta_decarb": 0.9, "eta_extract": 0.8, "eta_infuse": 0.85, "n_units": 400,
                          "rel_sigmas": [0.10, 0.10, 0.25, 0.15], "finished_product_coa": False}],
        "inventory": [{"item_id": f"LOT-{i}", "expected_g": 1000.0, "recorded_g": 1000.0 + random.Random(i).gauss(0, 3)}
                      for i in range(20)] + [{"item_id": "LOT-X", "expected_g": 1000.0, "recorded_g": 950.0}],
        "regulatory_feed": [{"source": "OAC 442:10-5 (illustrative text)",
                             "old": "Testing shall include potency.\nLabels shall state THC.",
                             "new": "Testing shall include potency and total THC.\nLabels shall state THC.\n"
                                    "Child-resistant packaging is required for all edibles."}],
        "hemp_products": [{"product_id": "HEMP-FS-CBD", "delta9_pct": 0.2, "total_thc_pct": 0.5,
                           "container_total_thc_mg": 12.0, "contains_synthesized": False}],
        "financials": {"category": "state_medical_licensed", "revenue": 10e6, "cogs": 5e6, "opex": 3.5e6},
        "product_catalog": catalog,
        "readiness": {"sensors_deployed": True, "calibration_schedule": True, "canonical_ids": True,
                      "validation_at_entry": True, "full_cycle_history": True, "validated_models_monitored": False,
                      "closed_loop_with_oversight": False, "outcomes_feed_breeding": False},
    }
