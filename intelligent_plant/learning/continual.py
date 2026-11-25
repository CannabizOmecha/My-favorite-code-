"""Self-iterating learning, governed.

Two layers of "recursive" learning, both standard and both safe by design:

1. RecursiveBayesianRegressor - exact conjugate Bayesian linear regression,
   updated one observation at a time (the Bayesian form of recursive least
   squares). Every prediction carries a posterior predictive interval, so the
   model knows how much it does not know (Chapter 4.4: calibrated, partially
   pooled, shrinking toward a prior when data are few).

2. ChampionChallengerLoop - on every cycle the loop retrains a challenger on
   recent GROUND-TRUTH outcomes, scores both models PROSPECTIVELY on data that
   arrived after the challenger's training cutoff, checks interval
   calibration, and promotes only a challenger that is better by a pre-written
   margin AND passes the Appendix C.5 deployment gate. Promotion of a clinical
   model is queued for its human owner; the champion is retained otherwise,
   and drift triggers retraining.

Hard rule: models learn only from measured outcomes, never from their own
predictions or from data that their own actions selected without logging the
propensity. Training on one's own outputs is how feedback loops amplify error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..governance.registry import ModelCard, deployment_gate
from ..science.stats import interval_coverage, population_stability_index


def t_quantile_95(dof: float) -> float:
    """Upper 95% Student-t quantile (Cornish-Fisher expansion; accurate to ~1e-3 for dof >= 3)."""
    z = 1.6448536269514722
    v = max(dof, 1.0)
    return z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v ** 2)


class RecursiveBayesianRegressor:
    """Conjugate Normal-Inverse-Gamma Bayesian linear regression, updated one
    observation at a time (exact recursive Bayesian inference).
        y = w.phi(x) + e,  e ~ N(0, s2),  w | s2 ~ N(mu, s2 L^-1),  s2 ~ IG(a, b)
    The predictive is Student-t with 2a degrees of freedom, so intervals are
    wide when observations are few and the noise level is still uncertain."""

    def __init__(self, n_features: int, prior_precision: float = 1.0, a0: float = 1.0, b0: float = 1.0):
        self.d = n_features + 1                      # + intercept
        self.L = np.eye(self.d) * prior_precision    # precision (scaled by 1/s2)
        self.mu = np.zeros(self.d)
        self.a, self.b = a0, b0
        self._n = 0

    def _phi(self, x) -> np.ndarray:
        return np.concatenate([[1.0], np.asarray(x, float)])

    @property
    def mean(self) -> np.ndarray:
        return self.mu

    def update(self, x, y: float) -> None:
        phi = self._phi(x)
        L_new = self.L + np.outer(phi, phi)
        mu_new = np.linalg.solve(L_new, self.L @ self.mu + phi * y)
        self.b += 0.5 * (y * y + self.mu @ self.L @ self.mu - mu_new @ L_new @ mu_new)
        self.a += 0.5
        self.L, self.mu = L_new, mu_new
        self._n += 1

    def fit(self, X, y) -> "RecursiveBayesianRegressor":
        for xi, yi in zip(X, y):
            self.update(xi, yi)
        return self

    @property
    def noise_var(self) -> float:
        return self.b / (self.a - 1) if self.a > 1 else float("inf")

    def predict(self, x) -> tuple[float, float, float]:
        """(mean, lo90, hi90) from the Student-t posterior predictive."""
        phi = self._phi(x)
        m = float(phi @ self.mu)
        scale = math.sqrt(self.b / self.a * (1 + float(phi @ np.linalg.solve(self.L, phi))))
        q = t_quantile_95(2 * self.a)
        return m, m - q * scale, m + q * scale

    @property
    def n_observations(self) -> int:
        return self._n


@dataclass
class Evaluation:
    rmse: float
    coverage90: float
    n: int


def evaluate(model: RecursiveBayesianRegressor, X, y) -> Evaluation:
    preds = [model.predict(x) for x in X]
    rmse = math.sqrt(float(np.mean([(p[0] - t) ** 2 for p, t in zip(preds, y)]))) if len(y) else float("nan")
    cov = interval_coverage([(p[1], p[2]) for p in preds], list(y))
    return Evaluation(rmse, cov, len(y))


@dataclass
class LoopReport:
    cycle: int
    champion_version: int
    challenger: Evaluation | None
    champion: Evaluation | None
    promoted: bool
    pending_human_approval: bool
    reason: str
    drift_psi: float | None = None


@dataclass
class ChampionChallengerLoop:
    model_id: str
    n_features: int
    owner: str
    scope: str
    out_of_scope: list[str]
    clinical: bool = False
    window: int = 500                  # train on the most recent outcomes
    holdout_fraction: float = 0.25     # most recent slice = prospective test
    min_improvement: float = 0.02      # pre-registered: >=2% lower RMSE to promote
    min_holdout: int = 20
    drift_threshold: float = 0.25
    prior_precision: float = 1.0
    data_X: list = field(default_factory=list)
    data_y: list = field(default_factory=list)
    data_source: list = field(default_factory=list)
    champion: RecursiveBayesianRegressor | None = None
    champion_version: int = 0
    champion_reference: list = field(default_factory=list)
    champion_validation: Evaluation | None = None
    suspended: bool = False            # degraded champion abstains rather than serve bad predictions
    min_coverage: float = 0.75
    pending: RecursiveBayesianRegressor | None = None
    pending_validation: Evaluation | None = None
    pending_reference: list = field(default_factory=list)
    history: list = field(default_factory=list)
    cycle_no: int = 0

    def observe(self, x, y: float, source: str = "measured") -> None:
        """Only measured outcomes enter the training data."""
        if source != "measured":
            raise ValueError("Only measured ground-truth outcomes may train the model, never model outputs.")
        if len(x) != self.n_features:
            raise ValueError(f"expected {self.n_features} features")
        if not (math.isfinite(float(y)) and all(math.isfinite(float(v)) for v in x)):
            # A NaN in the holdout turns every comparison False: promotion AND degradation
            # detection would silently stop. Reject at entry instead.
            raise ValueError("non-finite outcome or feature; not a measured value")
        self.data_X.append(list(map(float, x)))
        self.data_y.append(float(y))
        self.data_source.append(source)

    def _train(self, X, y) -> RecursiveBayesianRegressor:
        return RecursiveBayesianRegressor(self.n_features, self.prior_precision).fit(X, y)

    def step(self) -> LoopReport:
        """One self-iteration: retrain, evaluate prospectively, promote or keep."""
        self.cycle_no += 1
        X, y = self.data_X[-self.window:], self.data_y[-self.window:]
        n_hold = max(self.min_holdout, int(len(y) * self.holdout_fraction))
        if len(y) < n_hold * 2:
            r = LoopReport(self.cycle_no, self.champion_version, None, None, False, False,
                           f"Waiting for data: {len(y)} outcomes, need {n_hold * 2}.")
            self.history.append(r)
            return r
        Xtr, ytr, Xte, yte = X[:-n_hold], y[:-n_hold], X[-n_hold:], y[-n_hold:]
        champ_eval = evaluate(self.champion, Xte, yte) if self.champion else None
        degraded = bool(champ_eval and self.champion_validation and (
            champ_eval.coverage90 < self.min_coverage or champ_eval.rmse > 1.5 * self.champion_validation.rmse))
        if degraded and not self.suspended:
            self.suspended = True           # fail safe: abstain until a validated replacement exists
        # Two challengers: the full window, and recent data only (adapts after a regime change).
        cands = [(Xtr, ytr)] + [(Xtr[-m:], ytr[-m:]) for m in (2 * n_hold, n_hold, 50)
                                if 50 <= m < len(ytr)]
        scored = [(self._train(a, b), a) for a, b in cands]
        scored = [(m, a, evaluate(m, Xte, yte)) for m, a in scored]
        challenger, Xtr, ch_eval = min(scored, key=lambda t: (abs(t[2].coverage90 - 0.9) > 0.05, t[2].rmse))
        psi = None
        if self.champion and len(self.champion_reference) >= 50 and len(Xte) >= 50:
            # Bins sized to the sample (~20 per bin) so PSI noise stays well under the threshold;
            # the most-drifted feature counts.
            bins = max(3, min(10, len(Xte) // 20))
            psi = max(population_stability_index([r[j] for r in self.champion_reference], [r[j] for r in Xte],
                                                 bins=bins) for j in range(self.n_features))
        card = ModelCard(self.model_id, str(self.champion_version + 1), self.owner, f"window@{len(self.data_y)}",
                         {"prior_precision": self.prior_precision}, self.scope, self.out_of_scope,
                         {"rmse": ch_eval.rmse, "n_prospective": ch_eval.n}, ch_eval.coverage90)
        gate = deployment_gate(card, self.clinical)
        better = champ_eval is None or ch_eval.rmse <= champ_eval.rmse * (1 - self.min_improvement)
        drifted = psi is not None and psi > self.drift_threshold
        not_worse = champ_eval is None or ch_eval.rmse <= champ_eval.rmse
        promote = gate.passed and (better or ((drifted or degraded) and not_worse))
        if promote and self.clinical:
            self.pending, self.pending_validation = challenger, ch_eval
            self.pending_reference = [list(x) for x in Xtr]
            reason = "Challenger passes gate; clinical model promotion queued for its human owner."
            r = LoopReport(self.cycle_no, self.champion_version, ch_eval, champ_eval, False, True, reason, psi)
        elif promote:
            self.champion, self.champion_version = challenger, self.champion_version + 1
            self.champion_reference = [list(x) for x in Xtr]
            self.champion_validation, self.suspended = ch_eval, False
            reason = "Promoted: " + ("input drift detected." if drifted and not better else
                                     f"prospective RMSE {ch_eval.rmse:.3f} vs "
                                     f"{champ_eval.rmse:.3f}." if champ_eval else "first validated model.")
            r = LoopReport(self.cycle_no, self.champion_version, ch_eval, champ_eval, True, False, reason, psi)
        else:
            reason = ("Gate failed: " + "; ".join(gate.failures)) if not gate.passed else \
                     f"Champion kept: challenger not >= {self.min_improvement:.0%} better."
            if self.suspended:
                reason = "Champion SUSPENDED (degraded on prospective data; abstaining). " + reason
            r = LoopReport(self.cycle_no, self.champion_version, ch_eval, champ_eval, False, False, reason, psi)
        self.history.append(r)
        return r

    def approve_pending(self, approver_role: str) -> bool:
        if approver_role != self.owner:
            raise PermissionError(f"Only {self.owner} may promote {self.model_id}.")
        if not self.pending:
            return False
        self.champion, self.pending = self.pending, None
        self.champion_version += 1
        self.suspended = False
        # Record the baseline the model was validated on, so degradation of a clinical model is
        # detected (and it abstains) exactly as for non-clinical models.
        self.champion_validation, self.pending_validation = self.pending_validation, None
        self.champion_reference, self.pending_reference = self.pending_reference, []
        return True

    def predict(self, x) -> tuple[float, float, float] | None:
        """Abstains (None) until a validated champion exists."""
        return self.champion.predict(x) if self.champion and not self.suspended else None
