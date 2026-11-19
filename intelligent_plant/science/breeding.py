"""Genomic breeding (Chapter 9): ridge/GBLUP vs top-k marker-assisted
selection, chosen by trait architecture, and a patient-first breeding index."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ridge_gblup(X: np.ndarray, y: np.ndarray, lam: float | None = None) -> np.ndarray:
    """Ridge regression on centred markers (equivalent to GBLUP). Solved in the
    n x n dual so p >> n is cheap. Returns marker effects."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    n, p = Xc.shape
    if lam is None:
        lam = float(p)  # h2 ~ 0.5 default shrinkage: lambda = p (1-h2)/h2
    alpha = np.linalg.solve(Xc @ Xc.T + lam * np.eye(n), yc)
    return Xc.T @ alpha


def top_k_markers(X: np.ndarray, y: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Marker-assisted selection: pick the k markers most associated with y
    and fit OLS on them. Returns (indices, coefficients)."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12
    r = np.abs(Xc.T @ yc) / denom
    idx = np.argsort(-r)[:k]
    coef, *_ = np.linalg.lstsq(Xc[:, idx], yc, rcond=None)
    return idx, coef


@dataclass
class SelectionAdvice:
    method: str
    reason: str
    cv_accuracy_gblup: float
    cv_accuracy_mas: float


def choose_method(X: np.ndarray, y: np.ndarray, folds: int = 5, k: int = 5,
                  seed: int = 0) -> SelectionAdvice:
    """Let the data decide (Figure 9.1): cross-validate both methods.
    Caution (Chapter 9.2): relatives spanning folds inflate naive CV."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    acc_g, acc_m = [], []
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        mu = X[tr].mean(0)
        b = ridge_gblup(X[tr], y[tr])
        pg = (X[te] - mu) @ b
        mi, mc = top_k_markers(X[tr], y[tr], k)
        pm = (X[te][:, mi] - mu[mi]) @ mc
        acc_g.append(np.corrcoef(pg, y[te])[0, 1])
        acc_m.append(np.corrcoef(pm, y[te])[0, 1])
    g, m = float(np.nanmean(acc_g)), float(np.nanmean(acc_m))
    if m >= g:
        return SelectionAdvice("marker_assisted_selection", "Few large-effect loci (oligogenic, e.g. chemotype, sex).", g, m)
    return SelectionAdvice("genomic_selection", "Many small-effect loci (polygenic, e.g. yield, vigor).", g, m)


@dataclass
class CandidateCultivar:
    name: str
    cbd_thc_ratio: float
    profile_cv: float           # coefficient of variation of the full chemical profile across grows
    terpene_stability: float    # 0..1
    mildew_resistance: float    # 0..1
    botrytis_resistance: float  # 0..1
    g_cannabinoid_per_kwh: float
    l_water_per_g: float


def patient_first_index(c: CandidateCultivar, target_ratio: float, tolerance: float = 0.10) -> dict:
    """Breeding backward from the patient (Chapter 9.3). Lexicographic intent:
    a cultivar outside the target-ratio tolerance is not a candidate at all."""
    ratio_ok = abs(c.cbd_thc_ratio - target_ratio) <= tolerance * target_ratio
    return {
        "eligible": ratio_ok,
        "consistency": 1.0 - min(1.0, c.profile_cv),          # safety trait (dosing precondition)
        "terpene_stability": c.terpene_stability,
        "contamination_resistance": (c.mildew_resistance + c.botrytis_resistance) / 2,
        "efficiency": c.g_cannabinoid_per_kwh / (1 + c.l_water_per_g),
    }


def effective_population_size(kinship_mean: float) -> float:
    """Ne ~ 1 / (2 * mean coancestry). Diversity guard against germplasm narrowing."""
    return float("inf") if kinship_mean <= 0 else 1.0 / (2 * kinship_mean)
