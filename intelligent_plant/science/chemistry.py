"""Cannabis chemistry: COA totals, decarboxylation kinetics, Arrhenius fitting
and a model-based decarboxylation planner (Chapter 3, Appendix A.1-A.3)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

R_GAS = 8.314462618  # J mol^-1 K^-1

MW_THC = 314.47
MW_THCA = 358.48
ACID_TO_NEUTRAL = MW_THC / MW_THCA  # 0.8772 (also used for CBDA -> CBD)


def total_thc(thc: float, thca: float) -> float:
    """Total THC = THC + 0.877 * THCA-A. Upper bound: assumes complete conversion."""
    return thc + ACID_TO_NEUTRAL * thca


def total_cbd(cbd: float, cbda: float) -> float:
    return cbd + ACID_TO_NEUTRAL * cbda


def remaining_fraction(k: float, t_s: float) -> float:
    """First-order decay: [A](t)/[A]0 = exp(-k t)."""
    return math.exp(-k * t_s)


def half_life_s(k: float) -> float:
    return math.log(2) / k


def time_to_conversion_s(k: float, conversion: float) -> float:
    """Seconds to reach a fractional conversion (0.99 -> t99 = ln 100 / k)."""
    if not 0 < conversion < 1:
        raise ValueError("conversion must be in (0, 1)")
    return math.log(1.0 / (1.0 - conversion)) / k


def arrhenius_k(A: float, Ea_J: float, T_c: float) -> float:
    return A * math.exp(-Ea_J / (R_GAS * (T_c + 273.15)))


@dataclass(frozen=True)
class ArrheniusFit:
    A: float          # s^-1
    Ea_J: float       # J/mol
    r2: float

    @property
    def Ea_kJ(self) -> float:
        return self.Ea_J / 1000.0

    def k(self, T_c: float) -> float:
        return arrhenius_k(self.A, self.Ea_J, T_c)


def fit_arrhenius(temps_c: list[float], ks: list[float]) -> ArrheniusFit:
    """Linear regression of ln k on 1/T (Appendix A.3)."""
    x = 1.0 / (np.asarray(temps_c, float) + 273.15)
    y = np.log(np.asarray(ks, float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-300
    return ArrheniusFit(A=float(math.exp(intercept)), Ea_J=float(-slope * R_GAS), r2=1 - ss_res / ss_tot)


# Wang et al. 2016 (ref 73): THCA-A in heated extract.
WANG_2016_TEMPS_C = [80.0, 95.0, 110.0]
WANG_2016_K = [0.18e-3, 0.66e-3, 1.83e-3]
THCA_FIT = fit_arrhenius(WANG_2016_TEMPS_C, WANG_2016_K)
# CBDA/CBGA decarboxylate ~half as fast under the same conditions (ref 73).
CBDA_RELATIVE_RATE = 0.5


@dataclass(frozen=True)
class VolatileLossModel:
    """First-order loss of monoterpenes. Parameters are an ILLUSTRATIVE
    assumption (no cited data); calibrate against your own GC-FID results.

    First-principles note: at a fixed target conversion, exposure time scales
    as 1/k_decarb, so retained volatiles = exp(-c * k_vol/k_decarb). The ratio
    k_vol/k_decarb falls with temperature when Ea_vol < Ea_decarb, i.e. a
    hotter, SHORTER process then preserves more. Lower-T/longer only wins when
    Ea_vol > Ea_decarb (or when other losses, e.g. THC->CBN oxidation, are
    included). The planner therefore reports what the parameters imply rather
    than assuming a direction; measure Ea_vol for your matrix."""
    A: float = 2.6e2
    Ea_J: float = 45_000.0

    def k(self, T_c: float) -> float:
        return arrhenius_k(self.A, self.Ea_J, T_c)


@dataclass(frozen=True)
class DecarbPlan:
    temperature_c: float
    time_min: float
    conversion: float
    volatile_retention: float
    assumption_note: str


def plan_decarboxylation(target_conversion: float = 0.95,
                         temps_c: tuple[float, ...] = tuple(range(90, 151, 5)),
                         max_time_min: float = 240.0,
                         acid: str = "THCA",
                         volatile: VolatileLossModel = VolatileLossModel()) -> list[DecarbPlan]:
    """Model-based controller (Chapter 3.3): for each temperature compute the
    time to reach the target conversion and the predicted volatile retention.
    Returns feasible plans sorted best-retention first."""
    rel = 1.0 if acid.upper() == "THCA" else CBDA_RELATIVE_RATE
    plans = []
    for T in temps_c:
        k = THCA_FIT.k(T) * rel
        t = time_to_conversion_s(k, target_conversion)
        if t / 60 > max_time_min:
            continue
        retention = math.exp(-volatile.k(T) * t)
        plans.append(DecarbPlan(T, t / 60, target_conversion, retention,
                                "Volatile-loss parameters are illustrative; extrapolation above 110 C "
                                "is outside the measured range (ref 73)."))
    plans.sort(key=lambda p: -p.volatile_retention)
    return plans


def combinatorial_space(n: int) -> tuple[int, int]:
    """Appendix A.14: presence/absence mixtures and pairwise interaction terms."""
    return 2 ** n, n * (n - 1) // 2
