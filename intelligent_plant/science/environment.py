"""Controlled-environment science: VPD (Tetens), DLI, PID and a small
model-predictive controller with energy pricing (Chapters 8 and 10)."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

# Horticultural convention, NOT trial-established for cannabis (Figure 8.1 caption).
VPD_BANDS_KPA = {
    "propagation": (0.4, 0.8),
    "vegetative": (0.8, 1.2),
    "flowering": (1.2, 1.6),
}


def saturation_vp_kpa(T_c: float) -> float:
    return 0.6108 * math.exp(17.27 * T_c / (T_c + 237.3))


def vpd_air(T_air: float, rh: float) -> float:
    return saturation_vp_kpa(T_air) * (1 - rh / 100.0)


def vpd_leaf(T_leaf: float, T_air: float, rh: float) -> float:
    return saturation_vp_kpa(T_leaf) - saturation_vp_kpa(T_air) * (rh / 100.0)


def rh_for_target_leaf_vpd(target_kpa: float, T_leaf: float, T_air: float) -> float:
    """Invert VPD_leaf for the RH setpoint that yields the target leaf VPD."""
    rh = 100.0 * (saturation_vp_kpa(T_leaf) - target_kpa) / saturation_vp_kpa(T_air)
    return max(0.0, min(100.0, rh))


def dli(ppfd: float, hours: float) -> float:
    """DLI = PPFD x hours x 0.0036 (mol m^-2 d^-1)."""
    return ppfd * hours * 0.0036


def grams_per_kwh(photon_efficacy_umol_per_j: float, canopy_fraction: float,
                  grams_per_mol: float) -> float:
    """Efficiency chain (Chapter 10.2). 1 kWh = 3.6e6 J; umol -> mol 1e-6."""
    mol_per_kwh = photon_efficacy_umol_per_j * 3.6e6 * 1e-6
    return mol_per_kwh * canopy_fraction * grams_per_mol


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    out_min: float = -1.0
    out_max: float = 1.0
    _integral: float = 0.0
    _prev_error: float | None = None

    def step(self, error: float, dt: float) -> float:
        self._integral += error * dt
        deriv = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        u = self.kp * error + self.ki * self._integral + self.kd * deriv
        if u > self.out_max or u < self.out_min:   # anti-windup: undo integration when saturated
            self._integral -= error * dt
        return max(self.out_min, min(self.out_max, u))


@dataclass
class RoomModel:
    """Coupled linear room dynamics per control step (illustrative; identify
    from your own data). Actions in [0,1]: cooling, dehumidification, humidification.
    Dehumidifying adds heat; cooling raises RH - the coupling PID loops fight over."""
    heat_gain_c: float = 0.6        # passive drift per step from lights/equipment
    cool_gain_c: float = -1.5
    dehum_heat_c: float = 0.3
    dehum_rh: float = -6.0
    humid_rh: float = 5.0
    cool_rh_per_c: float = -2.5      # RH rises ~2.5 %/C when cooling (sign: dT<0 -> dRH>0)
    transpiration_rh: float = 1.5

    def step(self, T: float, rh: float, u: tuple[float, float, float]) -> tuple[float, float]:
        cool, dehum, humid = u
        dT = self.heat_gain_c + self.cool_gain_c * cool + self.dehum_heat_c * dehum
        drh = (self.transpiration_rh + self.dehum_rh * dehum + self.humid_rh * humid
               + self.cool_rh_per_c * dT)
        return T + dT, max(5.0, min(99.0, rh + drh))


@dataclass
class MPCResult:
    action: tuple[float, float, float]
    predicted: list[tuple[float, float]]
    cost: float


@dataclass
class MPCController:
    """min sum_k ||x_k - x*||_Q^2 + ||u_k||_R^2 + lambda*C_energy(u_k)  (Appendix A.10)
    Solved by exhaustive search over a discretised action set; only the first
    action is applied (receding horizon)."""
    model: RoomModel = field(default_factory=RoomModel)
    horizon: int = 3
    levels: tuple[float, ...] = (0.0, 0.5, 1.0)
    q_temp: float = 1.0
    q_vpd: float = 25.0
    r_effort: float = 0.05
    energy_weight: float = 1.0
    kw_per_unit: tuple[float, float, float] = (3.0, 2.0, 0.5)
    T_max: float = 30.0
    T_min: float = 16.0

    def solve(self, T: float, rh: float, T_target: float, vpd_target: float,
              leaf_offset_c: float, price_per_kwh: list[float]) -> MPCResult:
        actions: list[tuple[float, float, float]] = [(a, b, c) for a, b, c in itertools.product(self.levels, repeat=3)]
        best: MPCResult | None = None
        # Search the first action exhaustively, then hold it (move-blocking) -
        # keeps the search at 27 evaluations while still pricing energy over H.
        for u in actions:
            if u[1] > 0 and u[2] > 0:       # never dehumidify and humidify at once
                continue
            x_T, x_rh, cost, traj = T, rh, 0.0, []
            feasible = True
            for k in range(self.horizon):
                x_T, x_rh = self.model.step(x_T, x_rh, u)
                if not (self.T_min <= x_T <= self.T_max):
                    feasible = False
                    break
                v = vpd_leaf(x_T - leaf_offset_c, x_T, x_rh)
                kwh = sum(a * p for a, p in zip(u, self.kw_per_unit, strict=True))
                price = price_per_kwh[min(k, len(price_per_kwh) - 1)]
                cost += (self.q_temp * (x_T - T_target) ** 2 + self.q_vpd * (v - vpd_target) ** 2
                         + self.r_effort * sum(a * a for a in u) + self.energy_weight * price * kwh)
                traj.append((x_T, x_rh))
            if feasible and (best is None or cost < best.cost):
                best = MPCResult(u, traj, cost)
        if best is None:  # fail safe: no humidity actuation; cool only if the room is on the warm side
            cool = 1.0 if (self.T_min + self.T_max) / 2 <= T else 0.0
            return MPCResult((cool, 0.0, 0.0), [], float("inf"))
        return best
