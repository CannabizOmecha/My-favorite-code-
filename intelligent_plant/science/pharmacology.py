"""Pharmacokinetic and control-theory relationships (Chapters 2, 4, 6;
Appendix A.4-A.6). These produce RELATIVE exposures for explanation and
decision support - never a dose."""
from __future__ import annotations

import math
from dataclasses import dataclass

# Representative ranges (refs 81, 82)
ROUTES = {
    "inhaled": {"bioavailability": (0.10, 0.35), "onset_min": (2, 10), "peak_min": 10,
                "duration_h": (2, 4)},
    "oral": {"bioavailability": (0.04, 0.12), "onset_min": (60, 120), "peak_min": 180,
             "duration_h": (6, 8)},
    "oromucosal": {"bioavailability": (0.06, 0.20), "onset_min": (15, 45), "peak_min": 120,
                   "duration_h": (4, 6)},
}

# CYP2C9 genotype -> relative THC clearance. *3/*3 reproduces the ~3x AUC of
# ref 26; *1/*3 is interpolated (Figure 4.1); *2 showed no significant difference.
CYP2C9_RELATIVE_CLEARANCE = {
    "*1/*1": 1.0, "*1/*2": 1.0, "*2/*2": 1.0,
    "*1/*3": 0.5, "*2/*3": 0.5, "*3/*3": 1.0 / 3.0,
}

# CBD food effect, AUC fold change vs fasting (ref 94).
CBD_FOOD_EFFECT_AUC = {"fasted": 1.0, "high_fat": 3.8, "low_fat": 2.7, "whole_milk": 2.4,
                       "alcohol": 1.6}


def auc(F: float, dose: float, CL: float) -> float:
    """AUC = F * Dose / CL (Appendix A.4)."""
    return F * dose / CL


def relative_thc_exposure(cyp2c9_genotype: str | None) -> tuple[float, str]:
    if not cyp2c9_genotype:
        return 1.0, "Genotype unknown: exposure uncertainty is wide; prefer conservative titration."
    cl = CYP2C9_RELATIVE_CLEARANCE.get(cyp2c9_genotype.strip())
    if cl is None:
        return 1.0, f"Genotype {cyp2c9_genotype!r} not in evidence table; treated as unknown."
    return 1.0 / cl, ("Moderate evidence (controlled PK study, n=43, ref 26)." if cl < 1
                      else "No significant exposure difference expected.")


def one_compartment_oral(t_h: float, F: float, D: float, ka: float, CL: float, V: float) -> float:
    """C(t) = F D ka / (V (ka - ke)) * (exp(-ke t) - exp(-ka t)), ke = CL/V (A.5)."""
    ke = CL / V
    if abs(ka - ke) < 1e-12:
        return F * D * ka * t_h * math.exp(-ke * t_h) / V
    return F * D * ka / (V * (ka - ke)) * (math.exp(-ke * t_h) - math.exp(-ka * t_h))


@dataclass(frozen=True)
class FeedbackModel:
    """de/dt = (a - b k) e (Appendix A.6). Conceptual only: a, b, k have never
    been measured for any human symptom (Chapter 2.5)."""
    a: float
    b: float
    k: float

    @property
    def stable(self) -> bool:
        return self.b * self.k > self.a

    def deviation(self, e0: float, t: float) -> float:
        return e0 * math.exp((self.a - self.b * self.k) * t)


def oral_redose_safe(minutes_since_last_oral_dose: float, wait_min: float = 120.0) -> bool:
    """The edible-overdose script (Chapter 6.1): re-dosing before the full oral
    effect has arrived is the primary mechanism of accidental overdose."""
    return minutes_since_last_oral_dose >= wait_min
