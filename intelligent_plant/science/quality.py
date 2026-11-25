"""Quality control: label-claim uncertainty propagation (Chapter 11.2,
Appendix A.12), the ten-point COA audit (Chapter 3.4) and statistical
process control on batch records (Chapter 11.3)."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date

from .chemistry import total_cbd, total_thc


def label_claim_mg(P_pct: float, mass_mg: float, eta_decarb: float, eta_extract: float,
                   eta_infuse: float, n_units: int) -> float:
    """C_label = (P/100) m eta_decarb eta_extract eta_infuse / N."""
    return (P_pct / 100.0) * mass_mg * eta_decarb * eta_extract * eta_infuse / n_units


def label_relative_uncertainty(rel_sigmas: list[float]) -> float:
    """Relative variances of independent multiplicative factors add in quadrature."""
    return math.sqrt(sum(s * s for s in rel_sigmas))


@dataclass
class COA:
    coa_id: str
    batch_id: str
    lab_id: str
    lab_iso17025: bool
    method: str                      # "HPLC", "UHPLC", "GC", "GC-derivatized"
    thc: float                       # % w/w
    thca: float
    cbd: float
    cbda: float
    test_date: date
    packaging_date: date | None = None
    lot_matches_product: bool = True
    lod: dict | None = None
    loq: dict | None = None
    uncertainty_pct: float | None = None
    sample_plan: str | None = None
    moisture_pct: float | None = None
    dry_weight_basis: bool | None = None
    contaminants: dict = field(default_factory=dict)  # panel -> "pass"/"fail"/None
    reported_total_thc: float | None = None
    product_form: str = "flower"
    finished_product: bool = False

    @property
    def total_thc(self) -> float:
        return total_thc(self.thc, self.thca)

    @property
    def total_cbd(self) -> float:
        return total_cbd(self.cbd, self.cbda)


REQUIRED_CONTAMINANT_PANELS = ("pesticides", "residual_solvents", "heavy_metals", "mycotoxins",
                               "microbials")


@dataclass
class AuditResult:
    passed: bool
    findings: list[tuple[str, str]]   # (severity, message); severity in {"fail","warn"}

    @property
    def trust_score(self) -> float:
        fails = sum(1 for s, _ in self.findings if s == "fail")
        warns = sum(1 for s, _ in self.findings if s == "warn")
        return max(0.0, 1.0 - 0.25 * fails - 0.08 * warns)


def audit_coa(coa: COA, market_total_thc: list[float] | None = None,
              max_age_days: int = 365) -> AuditResult:
    """HOW TO READ A COA: A TEN-POINT CHECK (Chapter 3.4)."""
    f: list[tuple[str, str]] = []
    if not coa.lab_iso17025:
        f.append(("fail", "1. Laboratory not ISO/IEC 17025 accredited."))
    method = coa.method.upper()
    if method.startswith("GC") and "DERIV" not in method:   # GC, GC-FID, GC-MS, ...
        f.append(("fail", "1. GC without derivatization decarboxylates acids variably; values not comparable."))
    if not coa.lot_matches_product:
        f.append(("fail", "2. Batch/lot number does not match product in hand."))
    if coa.reported_total_thc is not None and abs(coa.reported_total_thc - coa.total_thc) > 0.05 * max(coa.total_thc, 0.1):
        f.append(("fail", f"3. Reported total THC {coa.reported_total_thc} != THC + 0.877*THCA = {coa.total_thc:.2f}."))
    if not coa.lod or not coa.loq:
        f.append(("warn", "4. Limits of detection/quantitation not reported."))
    if coa.uncertainty_pct is None:
        f.append(("warn", "5. Measurement uncertainty not stated."))
    if not coa.sample_plan:
        f.append(("warn", "6. Sample size/sampling plan not described."))
    if coa.product_form == "flower" and (coa.moisture_pct is None or not coa.dry_weight_basis):
        f.append(("warn", "7. Moisture content / dry-weight basis not declared for flower."))
    for panel in REQUIRED_CONTAMINANT_PANELS:
        r = coa.contaminants.get(panel)
        if r is None:
            f.append(("fail", f"8. Contaminant panel missing: {panel}."))
        elif r != "pass":
            f.append(("fail", f"8. Contaminant panel failed: {panel}."))
    if coa.packaging_date and (coa.packaging_date - coa.test_date).days > max_age_days:
        f.append(("warn", "9. Test date is stale relative to packaging date."))
    if market_total_thc and len(market_total_thc) >= 10:
        mu = statistics.mean(market_total_thc)
        sd = statistics.pstdev(market_total_thc) or 1.0
        if (coa.total_thc - mu) / sd > 3:
            f.append(("warn", "10. Potency implausibly high vs market distribution (possible lab shopping)."))
    return AuditResult(passed=not any(s == "fail" for s, _ in f), findings=f)


@dataclass
class SPCResult:
    mean: float
    sd: float
    cv: float
    ucl: float
    lcl: float
    out_of_control: list[int]
    rule: str


def shewhart(values: list[float], baseline_n: int | None = None) -> SPCResult:
    """Individuals chart with 3-sigma limits from a baseline (moving-range sigma)."""
    base = values[:baseline_n] if baseline_n else values
    if len(base) < 2:
        raise ValueError("need at least 2 baseline points")
    mean = statistics.mean(base)
    mr = [abs(b - a) for a, b in zip(base, base[1:])]
    sigma = (statistics.mean(mr) / 1.128) if mr else 0.0
    ucl, lcl = mean + 3 * sigma, mean - 3 * sigma
    ooc = [i for i, v in enumerate(values) if v > ucl or v < lcl]
    cv = (statistics.pstdev(values) / mean) if mean else float("inf")
    return SPCResult(mean, sigma, cv, ucl, lcl, ooc, "Shewhart individuals, 3-sigma, MR-based")
