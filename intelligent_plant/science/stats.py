"""Statistics for honest decision support (Chapters 4.4, 5.4, 7.1, 13.3, 16.2):
partial pooling, calibration checks, N-of-1 design/analysis, factorial
interaction tests and drift detection."""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

import numpy as np

Z90 = 1.6448536


@dataclass
class PooledEstimate:
    mean: float
    lo90: float
    hi90: float
    n: int
    shrinkage: float           # 0 = all individual data, 1 = all population prior
    abstain: bool
    note: str


def partial_pool(individual: list[float], pop_mean: float, pop_tau: float, sigma: float,
                 min_n: int = 3) -> PooledEstimate:
    """Normal-normal hierarchical posterior. Wide intervals when data are few;
    shrinkage toward the population protects against over-fitting the first
    doses. Abstains (but still reports) below min_n observations."""
    n = len(individual)
    prec_prior = 1.0 / pop_tau ** 2
    prec_data = n / sigma ** 2 if n else 0.0
    post_var = 1.0 / (prec_prior + prec_data)
    ybar = statistics.mean(individual) if n else pop_mean
    mean = post_var * (prec_prior * pop_mean + prec_data * ybar)
    sd = math.sqrt(post_var)
    shrink = prec_prior / (prec_prior + prec_data)
    abstain = n < min_n
    note = (f"Based on {n} record(s) from this patient; "
            + ("too few to individualise - population prior dominates." if abstain
               else f"{100 * (1 - shrink):.0f}% weight on the patient's own data."))
    return PooledEstimate(mean, mean - Z90 * sd, mean + Z90 * sd, n, shrink, abstain, note)


def population_hyperparameters(groups: list[list[float]]) -> tuple[float, float, float]:
    """Method-of-moments estimates of (mu, tau, sigma) across patients."""
    groups = [g for g in groups if g]
    means = [statistics.mean(g) for g in groups]
    mu = statistics.mean(means) if means else 0.0
    within = [statistics.pvariance(g) for g in groups if len(g) > 1]
    sigma2 = statistics.mean(within) if within else 1.0
    between = statistics.pvariance(means) if len(means) > 1 else 1.0
    avg_n = statistics.mean(len(g) for g in groups) if groups else 1
    tau2 = max(between - sigma2 / avg_n, 0.05 * sigma2)
    return mu, math.sqrt(tau2), math.sqrt(max(sigma2, 1e-6))


def interval_coverage(intervals: list[tuple[float, float]], outcomes: list[float]) -> float:
    """Prospective calibration: of 90% intervals, ~90% should contain the outcome."""
    hits = sum(lo <= y <= hi for (lo, hi), y in zip(intervals, outcomes))
    return hits / len(outcomes) if outcomes else float("nan")


def n_of_1_schedule(pairs: int, seed: int | None = None) -> list[str]:
    """Randomised, balanced A/B pairs for a within-patient crossover (ref 98)."""
    rng = random.Random(seed)
    sched: list[str] = []
    for _ in range(pairs):
        pair = ["A", "B"]
        rng.shuffle(pair)
        sched.extend(pair)
    return sched


@dataclass
class NOf1Result:
    mean_difference: float     # A - B (positive = A better when outcome is improvement)
    lo90: float
    hi90: float
    pairs: int
    conclusive: bool


def analyse_n_of_1(schedule: list[str], outcomes: list[float]) -> NOf1Result:
    diffs = []
    for i in range(0, len(schedule) - 1, 2):
        a = outcomes[i] if schedule[i] == "A" else outcomes[i + 1]
        b = outcomes[i + 1] if schedule[i] == "A" else outcomes[i]
        diffs.append(a - b)
    m = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("inf")
    tcrit = {2: 2.92, 3: 2.35, 4: 2.13, 5: 2.02, 6: 1.94, 7: 1.89, 8: 1.86}.get(len(diffs) - 1, 1.8)
    lo, hi = m - tcrit * se, m + tcrit * se
    return NOf1Result(m, lo, hi, len(diffs), conclusive=(lo > 0 or hi < 0))


def factorial_interaction(A: np.ndarray, B: np.ndarray, Y: np.ndarray) -> dict:
    """Fit Y = b0 + b1 A + b2 B + b3 AB + e (Appendix A.7). Synergy for a
    beneficial outcome is b3 > 0 with its CI excluding zero."""
    X = np.column_stack([np.ones_like(A), A, B, A * B]).astype(float)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = max(1, len(Y) - 4)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se3 = math.sqrt(cov[3, 3])
    b3 = float(beta[3])
    verdict = ("synergy" if b3 - 1.96 * se3 > 0 else "antagonism" if b3 + 1.96 * se3 < 0
               else "additive / not identified")
    return {"beta": beta.tolist(), "b3": b3, "se_b3": se3, "verdict": verdict}


def population_stability_index(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Input/output drift monitor. PSI > 0.25 conventionally triggers review."""
    e = np.asarray(expected, float)
    a = np.asarray(actual, float)
    edges = np.quantile(e, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e) + 1e-6
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def robust_z(values: list[float]) -> list[float]:
    """Median/MAD z-scores for anomaly detection on low-base-rate events."""
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values]) or 1e-9
    return [0.6745 * (v - med) / mad for v in values]
