"""Cultivation swarm (Part III): calibration, climate (leaf-VPD MPC with
fail-safe), irrigation, nutrients, vision triage, harvest decision support
and lighting energy.

Observations expected under ctx.observations["rooms"][room_id]:
  stage, reading{temp_air,temp_leaf?,rh,co2,ppfd,sensor_cal_date,instrument_id},
  photoperiod_h, price_forecast[$/kWh], substrate_moisture, target_moisture,
  ec, ph, detections[{label,p,validated_domain}], days_in_flower, known_good{T,rh}
"""
from __future__ import annotations

from datetime import date

from ..science.environment import (VPD_BANDS_KPA, MPCController, dli, rh_for_target_leaf_vpd, vpd_air,
                                   vpd_leaf)
from .base import Agent, Context, Proposal

DEFAULT_LEAF_OFFSET_C = 2.0
TEMP_TARGET = {"propagation": 25.0, "vegetative": 26.0, "flowering": 25.0}
DLI_TARGET = {"propagation": 15.0, "vegetative": 30.0, "flowering": 40.0}
EC_RANGE = {"propagation": (0.6, 1.2), "vegetative": (1.2, 2.0), "flowering": (1.6, 2.4)}
PH_RANGE = (5.6, 6.4)


def _cal_ok(reading: dict, today: date) -> bool:
    due = reading.get("sensor_cal_due")
    return bool(due) and date.fromisoformat(due) >= today


class CalibrationWatchdog(Agent):
    """Measurement quality dominates model quality (Chapter 8.1)."""
    name, domain = "calibration_watchdog", "cultivation"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        today = ctx.now.date()
        for inst in ctx.observations.get("instruments", []):
            if date.fromisoformat(inst["next_due"]) <= today:
                out.append(Proposal(self.name, "calibration_schedule",
                                    f"Calibration overdue for {inst['instrument_id']}; work order raised and data "
                                    f"since {inst['next_due']} flagged retrospectively.",
                                    {"instrument_id": inst["instrument_id"], "flag_from": inst["next_due"]},
                                    {"safety": 0.1, "accuracy": 1.0}, ["Ch 8.1", "Ch 16.2 calibration records"]))
        return out


class ClimateAgent(Agent):
    name, domain = "climate_controller", "cultivation"

    def __init__(self, mpc: MPCController | None = None):
        self.mpc = mpc or MPCController()

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        today = ctx.now.date()
        for room_id, room in ctx.observations.get("rooms", {}).items():
            r = room.get("reading") or {}
            stage = room.get("stage", "vegetative")
            target_band = VPD_BANDS_KPA[stage]
            vpd_target = sum(target_band) / 2
            plausible = (r.get("temp_air") is not None and r.get("rh") is not None
                         and -5 < r["temp_air"] < 50 and 0 <= r["rh"] <= 100)
            if not plausible or not _cal_ok(r, today):
                kg = room.get("known_good", {"T": TEMP_TARGET[stage], "rh": 60.0})
                out.append(Proposal(
                    self.name, "sensor_failsafe",
                    f"{room_id}: sensor data {'implausible' if not plausible else 'out of calibration'}; reverting "
                    f"to known-good setpoint T={kg['T']} C, RH={kg['rh']}%.",
                    {"room_id": room_id, "setpoint": kg, "inputs_calibrated": plausible and _cal_ok(r, today),
                     "alarm": f"{room_id} climate on fail-safe setpoint; sensor needs attention."},
                    {"safety": 1.0, "accuracy": 0.5}, ["Ch 16.3 safe failure modes"],
                    target=f"room:{room_id}:climate"))
                continue
            leaf_measured = r.get("temp_leaf") is not None
            offset = (r["temp_air"] - r["temp_leaf"]) if leaf_measured else DEFAULT_LEAF_OFFSET_C
            v_leaf = vpd_leaf(r["temp_air"] - offset, r["temp_air"], r["rh"])
            v_air = vpd_air(r["temp_air"], r["rh"])
            prices = room.get("price_forecast", [0.12])
            res = self.mpc.solve(r["temp_air"], r["rh"], TEMP_TARGET[stage], vpd_target, offset, prices)
            rh_sp = rh_for_target_leaf_vpd(vpd_target, TEMP_TARGET[stage] - offset, TEMP_TARGET[stage])
            off_band = not (target_band[0] - 0.3 <= v_leaf <= target_band[1] + 0.3)
            out.append(Proposal(
                self.name, "climate_setpoint",
                f"{room_id} ({stage}): leaf VPD {v_leaf:.2f} kPa (air {v_air:.2f}) vs band {target_band}; MPC action "
                f"cool/dehum/humid={res.action}, RH setpoint {rh_sp:.0f}%.",
                {"room_id": room_id, "action": res.action, "rh_setpoint": round(rh_sp, 1),
                 "temp_setpoint": TEMP_TARGET[stage], "vpd_leaf": round(v_leaf, 3), "vpd_air": round(v_air, 3),
                 "leaf_temp_measured": leaf_measured, "predicted": res.predicted, "inputs_calibrated": True,
                 "alarm": (f"{room_id} leaf VPD {v_leaf:.2f} kPa far outside {target_band}" if off_band else None)},
                {"safety": 0.5, "accuracy": 0.9 if leaf_measured else 0.6, "efficiency": max(0.0, 1 - res.cost / 100)},
                ["Tetens VPD (A.8)", "MPC objective (A.10)", "VPD bands are horticultural convention (Fig 8.1)"],
                uncertainty=None if leaf_measured else "Leaf temperature not measured; assumed 2 C below air.",
                target=f"room:{room_id}:climate"))
        return out


class IrrigationAgent(Agent):
    name, domain = "irrigation", "cultivation"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for room_id, room in ctx.observations.get("rooms", {}).items():
            m, t = room.get("substrate_moisture"), room.get("target_moisture")
            if m is None or t is None:
                continue
            if m < t - 3:
                out.append(Proposal(self.name, "irrigation_schedule",
                                    f"{room_id}: substrate moisture {m}% < target {t}%; irrigate one pulse.",
                                    {"room_id": room_id, "pulses": 1, "moisture": m, "target": t},
                                    {"safety": 0.3, "efficiency": 0.6}, ["Table 8.2"]))
        return out


class NutrientAgent(Agent):
    name, domain = "nutrients", "cultivation"

    def __init__(self) -> None:
        self.ec_range = dict(EC_RANGE)
        self.ph_range = {s: PH_RANGE for s in EC_RANGE}
        self.source = "typical practice (uncalibrated)"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for room_id, room in ctx.observations.get("rooms", {}).items():
            stage = room.get("stage", "vegetative")
            ec, ph = room.get("ec"), room.get("ph")
            issues = []
            ecr, phr = self.ec_range[stage], self.ph_range[stage]
            if ec is not None and not ecr[0] <= ec <= ecr[1]:
                issues.append(f"EC {ec} outside {ecr}")
            if ph is not None and not phr[0] <= ph <= phr[1]:
                issues.append(f"pH {ph} outside {phr}")
            if issues:
                out.append(Proposal(self.name, "nutrient_recipe", f"{room_id}: " + "; ".join(issues) +
                                    ". Recommend recipe correction.",
                                    {"room_id": room_id, "ec": ec, "ph": ph, "stage": stage},
                                    {"safety": 0.3, "accuracy": 0.6}, ["Table 8.2: recommend; human approves",
                                                                       "Ch 10.3 closed-loop fertigation"],
                                    uncertainty=f"Ranges: {self.source}."))
        return out


class VisionTriageAgent(Agent):
    """Triage only; a human confirms. Never proposes crop destruction."""
    name, domain = "vision_triage", "cultivation"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for room_id, room in ctx.observations.get("rooms", {}).items():
            for det in room.get("detections", []):
                if det["p"] < 0.5:
                    continue
                note = ("" if det.get("validated_domain") else
                        " Model NOT validated on this facility's images: lab accuracy is not field accuracy (ref 44).")
                out.append(Proposal(self.name, "disease_identification",
                                    f"{room_id}: possible {det['label']} (p={det['p']:.2f}). Inspect and confirm.{note}",
                                    {"room_id": room_id, **det}, {"safety": 0.4, "accuracy": 0.5 if note else 0.8},
                                    ["Ch 8.4", "Table 8.2: triage only"],
                                    uncertainty=f"Classifier probability {det['p']:.2f}; false positives cost labour."))
        return out


class HarvestAdvisor(Agent):
    name, domain = "harvest_advisor", "cultivation"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for room_id, room in ctx.observations.get("rooms", {}).items():
            d = room.get("days_in_flower")
            if room.get("stage") == "flowering" and d and d >= room.get("expected_flower_days", 60) - 5:
                out.append(Proposal(self.name, "harvest_timing",
                                    f"{room_id}: day {d} of flower; harvest window approaching. Inspect trichomes and "
                                    f"decide.", {"room_id": room_id, "days_in_flower": d,
                                                 "trichome_amber_pct": room.get("trichome_amber_pct")},
                                    {"accuracy": 0.5}, ["Table 8.2: decision support only"],
                                    uncertainty="Harvest timing has high error cost and medium signal quality."))
        return out


class LightingEnergyAgent(Agent):
    name, domain = "lighting_energy", "cultivation"

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for room_id, room in ctx.observations.get("rooms", {}).items():
            r = room.get("reading") or {}
            if r.get("ppfd") is None or room.get("photoperiod_h") is None:
                continue
            stage = room.get("stage", "vegetative")
            d = dli(r["ppfd"], room["photoperiod_h"])
            tgt = room.get("dli_target", DLI_TARGET[stage])
            if d > tgt * 1.10:
                new_ppfd = tgt / (room["photoperiod_h"] * 0.0036)
                out.append(Proposal(self.name, "lighting_energy",
                                    f"{room_id}: DLI {d:.1f} > target {tgt} mol/m2/d; dim to {new_ppfd:.0f} PPFD.",
                                    {"room_id": room_id, "dli": round(d, 1), "ppfd_setpoint": round(new_ppfd)},
                                    {"safety": 0.0, "efficiency": 0.8, "profitability": 0.5},
                                    ["DLI (A.11)", "Yield ~linear in PPFD to 1800 (ref 135): target is set by "
                                     "marginal revenue vs energy, not agronomic max."]))
        return out


def cultivation_agents() -> list[Agent]:
    return [CalibrationWatchdog(), ClimateAgent(), IrrigationAgent(), NutrientAgent(), VisionTriageAgent(),
            HarvestAdvisor(), LightingEnergyAgent()]
