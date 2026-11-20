"""Facility calibration: replace the illustrative defaults with parameters fitted to THIS facility's data.

Three fits, each from first principles and each refusing to answer when the data cannot support one:

1. Room model (system identification). The climate model is linear per control step:
       dT  = heat_gain + cool_gain*cool + dehum_heat*dehum
       dRH = transpiration + dehum_rh*dehum + humid_rh*humid + cool_rh_per_c*dT
   Ordinary least squares on logged readings and actuator states. A parameter is only identifiable if its
   actuator actually varied in the log; otherwise the fit is refused rather than invented.
   Validated on a time-ordered holdout (last 25%) - the future, not a random shuffle of the past.

2. Volatile (terpene) loss kinetics. From GC measurements of retention after heating at several
   temperatures: k = -ln(retention)/t per run, then Arrhenius ln k = ln A - Ea/RT. Needs >= 3 distinct
   temperatures. The report states which direction (hotter-shorter or cooler-longer) preserves more
   terpenes, because that depends on Ea_volatile vs Ea_decarb (Chapter 3.3).

3. Nutrient (EC/pH) ranges. The head grower's SOP range is authoritative. The data check whether batches
   that met quality targets actually ran inside it, and report the 10th-90th percentile of good batches.

Input is CSV exported from the BMS / LIMS / batch records. Every row is validated at entry.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..science.chemistry import THCA_FIT, fit_arrhenius
from ..science.environment import RoomModel


class CalibrationError(ValueError):
    """The data cannot support the requested fit. Fix the data; do not loosen the check."""


# ----------------------------------------------------------------- CSV loading
def read_csv(path: str | Path, required: dict[str, tuple[float, float]]) -> tuple[list[dict], list[str], str]:
    """Load rows, validating presence and physical range of each numeric column.
    Returns (good_rows, rejected_reasons, sha256_of_file)."""
    p = Path(path)
    raw = p.read_bytes()
    rows, rejected = [], []
    with p.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise CalibrationError(f"{p.name}: missing columns {missing}")
        for n, r in enumerate(reader, start=2):
            try:
                vals = {c: float(r[c]) for c in required}
            except (TypeError, ValueError):
                rejected.append(f"line {n}: non-numeric value")
                continue
            bad = [c for c, (lo, hi) in required.items() if not lo <= vals[c] <= hi]
            if bad:
                rejected.append(f"line {n}: out of physical range {bad}")
                continue
            rows.append({**r, **vals})
    return rows, rejected, hashlib.sha256(raw).hexdigest()


# ----------------------------------------------------------------- 1. room model
ROOM_COLUMNS = {"temp_air": (-5, 50), "rh": (0, 100), "cool": (0, 1), "dehum": (0, 1), "humid": (0, 1),
                "temp_air_next": (-5, 50), "rh_next": (0, 100)}


@dataclass
class FitStats:
    n_train: int
    n_holdout: int
    r2_temp: float
    r2_rh: float
    holdout_rmse_temp_c: float
    holdout_rmse_rh_pct: float
    param_se: dict[str, float]


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    r2 = 1 - float(resid @ resid) / (float(((y - y.mean()) ** 2).sum()) or 1e-12)
    return beta, se, r2


def fit_room_model(rows: list[dict], min_rows: int = 200, min_actuator_sd: float = 0.05) -> tuple[RoomModel, FitStats]:
    if len(rows) < min_rows:
        raise CalibrationError(f"{len(rows)} logged steps; need at least {min_rows} (several days of normal "
                               f"operation) for a stable room model.")
    A = np.array([[r["cool"], r["dehum"], r["humid"]] for r in rows])
    for j, name in enumerate(("cool", "dehum", "humid")):
        if A[:, j].std() < min_actuator_sd:
            raise CalibrationError(f"Actuator '{name}' barely varied in the log (sd {A[:, j].std():.3f}); its "
                                   f"effect is not identifiable. Log a period where it cycles normally.")
    T = np.array([r["temp_air"] for r in rows]); Tn = np.array([r["temp_air_next"] for r in rows])
    H = np.array([r["rh"] for r in rows]); Hn = np.array([r["rh_next"] for r in rows])
    dT, dH = Tn - T, Hn - H
    # A stuck or saturated sensor reports "no change" when the physics says there was one, which biases
    # every effect toward zero. Real readings almost never repeat exactly.
    for name, d in (("temperature", dT), ("humidity", dH)):
        frozen = float(np.mean(np.abs(d) < 1e-9))
        if frozen > 0.10:
            raise CalibrationError(f"{frozen:.0%} of {name} readings repeat exactly: the sensor looks stuck or "
                                   f"clamped at a limit. Fix or exclude that period before calibrating.")
    cut = int(len(rows) * 0.75)                         # time-ordered holdout
    one = np.ones(len(rows))
    XT = np.column_stack([one, A[:, 0], A[:, 1]])
    XH = np.column_stack([one, A[:, 1], A[:, 2], dT])
    bT, seT, r2T = _ols(XT[:cut], dT[:cut])
    bH, seH, r2H = _ols(XH[:cut], dH[:cut])
    rmseT = float(np.sqrt(np.mean((XT[cut:] @ bT - dT[cut:]) ** 2)))
    rmseH = float(np.sqrt(np.mean((XH[cut:] @ bH - dH[cut:]) ** 2)))
    model = RoomModel(heat_gain_c=float(bT[0]), cool_gain_c=float(bT[1]), dehum_heat_c=float(bT[2]),
                      transpiration_rh=float(bH[0]), dehum_rh=float(bH[1]), humid_rh=float(bH[2]),
                      cool_rh_per_c=float(bH[3]))
    if model.cool_gain_c >= 0:
        raise CalibrationError("Fitted cooling effect is not negative: cooling does not lower temperature in this "
                               "log. Check actuator column mapping and sensor placement before using the fit.")
    names = ["heat_gain_c", "cool_gain_c", "dehum_heat_c", "transpiration_rh", "dehum_rh", "humid_rh", "cool_rh_per_c"]
    stats = FitStats(cut, len(rows) - cut, r2T, r2H, rmseT, rmseH,
                     dict(zip(names, [float(x) for x in np.concatenate([seT, seH])])))
    return model, stats


# ----------------------------------------------------------------- 2. volatile loss
VOLATILE_COLUMNS = {"temp_c": (20, 250), "time_s": (1, 86400), "retention": (0.001, 1.0)}


@dataclass
class VolatileFit:
    A: float
    Ea_J: float
    r2: float
    temperatures_c: list[float]
    n_runs: int
    direction: str


def fit_volatile_loss(rows: list[dict]) -> VolatileFit:
    by_t: dict[float, list[float]] = {}
    for r in rows:
        if r["retention"] >= 1.0:
            continue                                     # no measurable loss: k undefined
        by_t.setdefault(r["temp_c"], []).append(-math.log(r["retention"]) / r["time_s"])
    if len(by_t) < 3:
        raise CalibrationError(f"Retention measured at {len(by_t)} distinct temperature(s); Arrhenius needs at "
                               f"least 3 (e.g. 95, 105, 115 C).")
    temps = sorted(by_t)
    ks = [float(np.exp(np.mean(np.log(by_t[t])))) for t in temps]   # geometric mean of replicates
    fit = fit_arrhenius(temps, ks)
    if fit.Ea_J <= 0:
        raise CalibrationError("Terpene loss did not increase with temperature; the data are inconsistent "
                               "(check timing, sealing and GC method).")
    if fit.Ea_J > THCA_FIT.Ea_J:
        direction = "cooler-and-longer preserves more terpenes at equal decarboxylation"
    else:
        direction = "hotter-and-shorter preserves more terpenes at equal decarboxylation"
    return VolatileFit(fit.A, fit.Ea_J, fit.r2, temps, sum(len(v) for v in by_t.values()), direction)


# ----------------------------------------------------------------- 3. nutrient ranges
NUTRIENT_COLUMNS = {"ec": (0, 10), "ph": (3, 9), "met_quality_target": (0, 1)}


@dataclass
class NutrientCheck:
    stage: str
    sop_ec: tuple[float, float]
    sop_ph: tuple[float, float]
    good_batches: int
    good_ec_p10_p90: tuple[float, float] | None
    good_ph_p10_p90: tuple[float, float] | None
    share_good_inside_sop: float | None
    note: str


def check_nutrient_ranges(rows: list[dict], sop: dict[str, dict], min_good: int = 20) -> list[NutrientCheck]:
    out = []
    for stage, rng in sop.items():
        good = [r for r in rows if r.get("stage") == stage and r["met_quality_target"] >= 0.5]
        ec, ph = tuple(rng["ec"]), tuple(rng["ph"])
        if len(good) < min_good:
            out.append(NutrientCheck(stage, ec, ph, len(good), None, None, None,
                                     f"Only {len(good)} good batches; SOP range used unchecked."))
            continue
        e = np.array([r["ec"] for r in good]); p = np.array([r["ph"] for r in good])
        inside = float(np.mean((e >= ec[0]) & (e <= ec[1]) & (p >= ph[0]) & (p <= ph[1])))
        note = ("Good batches ran inside the SOP range." if inside >= 0.8 else
                f"Only {inside:.0%} of good batches ran inside the SOP range: discuss with the head grower.")
        out.append(NutrientCheck(stage, ec, ph, len(good),
                                 (float(np.percentile(e, 10)), float(np.percentile(e, 90))),
                                 (float(np.percentile(p, 10)), float(np.percentile(p, 90))), inside, note))
    return out


# ----------------------------------------------------------------- output
@dataclass
class FacilityCalibration:
    facility: str
    created_at: str
    room_model: dict | None = None
    room_fit: dict | None = None
    volatile_loss: dict | None = None
    nutrient_ranges: dict | None = None
    nutrient_checks: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)        # file name -> sha256 (provenance)
    rejected_rows: dict = field(default_factory=dict)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def calibrate(facility: str, room_csv: str | Path | None = None, volatile_csv: str | Path | None = None,
              nutrient_csv: str | Path | None = None, sop: dict | None = None) -> FacilityCalibration:
    cal = FacilityCalibration(facility, datetime.now(timezone.utc).isoformat())
    if room_csv:
        rows, rej, sha = read_csv(room_csv, ROOM_COLUMNS)
        model, stats = fit_room_model(rows)
        cal.room_model, cal.room_fit = asdict(model), asdict(stats)
        cal.sources[Path(room_csv).name], cal.rejected_rows[Path(room_csv).name] = sha, rej
    if volatile_csv:
        rows, rej, sha = read_csv(volatile_csv, VOLATILE_COLUMNS)
        cal.volatile_loss = asdict(fit_volatile_loss(rows))
        cal.sources[Path(volatile_csv).name], cal.rejected_rows[Path(volatile_csv).name] = sha, rej
    if sop:
        cal.nutrient_ranges = sop
        if nutrient_csv:
            rows, rej, sha = read_csv(nutrient_csv, NUTRIENT_COLUMNS)
            cal.nutrient_checks = [asdict(c) for c in check_nutrient_ranges(rows, sop)]
            cal.sources[Path(nutrient_csv).name], cal.rejected_rows[Path(nutrient_csv).name] = sha, rej
    return cal


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
