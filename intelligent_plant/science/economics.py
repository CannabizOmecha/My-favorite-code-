"""Enterprise economics (Chapter 1.4, Appendix A.13) and sustainability
metrics (Chapter 10.4, Table 10.1)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class TaxResult:
    true_profit: float
    taxable_base: float
    tax: float
    effective_rate: float


def tax_under_280e(revenue: float, cogs: float, opex: float, rate: float = 0.21,
                   subject_to_280e: bool = True) -> TaxResult:
    true_profit = revenue - cogs - opex
    base = (revenue - cogs) if subject_to_280e else true_profit
    tax = rate * max(base, 0.0)   # a loss produces no tax, not a payout
    return TaxResult(true_profit, base, tax, tax / true_profit if true_profit else float("inf"))


@dataclass
class SustainabilityMetrics:
    energy_kwh_per_kg: float
    light_g_per_kwh: float
    hvac_share_pct: float
    water_l_per_kg: float
    n_kg_per_kg: float
    p_kg_per_kg: float
    carbon_kg_co2e_per_kg: float
    yield_cv: float


def sustainability_metrics(total_kwh: float, lighting_kwh: float, hvac_kwh: float,
                           water_l: float, condensate_reused_l: float, n_discharged_kg: float,
                           p_discharged_kg: float, grid_kg_co2e_per_kwh: float,
                           onsite_kg_co2e: float, dried_flower_kg: float,
                           yields_g_per_m2_by_cycle: list[float]) -> SustainabilityMetrics:
    """Ratios of output to input, from METERS not nameplate ratings."""
    kg = dried_flower_kg
    cv = (statistics.pstdev(yields_g_per_m2_by_cycle) / statistics.mean(yields_g_per_m2_by_cycle)
          if len(yields_g_per_m2_by_cycle) > 1 else float("nan"))
    return SustainabilityMetrics(
        energy_kwh_per_kg=total_kwh / kg,
        light_g_per_kwh=kg * 1000 / lighting_kwh,
        hvac_share_pct=100 * hvac_kwh / total_kwh,
        water_l_per_kg=(water_l - condensate_reused_l) / kg,
        n_kg_per_kg=n_discharged_kg / kg,
        p_kg_per_kg=p_discharged_kg / kg,
        carbon_kg_co2e_per_kg=(total_kwh * grid_kg_co2e_per_kwh + onsite_kg_co2e) / kg,
        yield_cv=cv,
    )
