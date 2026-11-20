"""Model registry, deployment gate (Appendix C.5), drift monitoring,
rollback (Chapter 16.2) and the maturity model (Table 16.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..science.stats import population_stability_index


@dataclass
class ModelCard:
    model_id: str
    version: str
    owner: str
    training_snapshot_id: str
    hyperparams: dict
    scope: str                                 # declared scope statement
    out_of_scope: list[str]
    prospective_validation: dict | None = None  # metrics on data NOT used in training
    calibration_coverage_90: float | None = None
    rule_based_interactions: bool = True
    logs_every_recommendation: bool = True
    approver: str | None = None
    deployed_at: str | None = None
    reference_inputs: list[float] = field(default_factory=list)


@dataclass
class GateResult:
    passed: bool
    failures: list[str]


def deployment_gate(card: ModelCard, clinical: bool) -> GateResult:
    f = []
    if not card.prospective_validation:
        f.append("No prospective validation on held-out data.")
    if card.calibration_coverage_90 is None:
        f.append("Calibration not measured.")
    elif not 0.85 <= card.calibration_coverage_90 <= 0.95:
        f.append(f"Miscalibrated: 90% intervals cover {card.calibration_coverage_90:.0%}.")
    if not card.scope or not card.out_of_scope:
        f.append("Scope and out-of-scope refusal behaviour not documented.")
    if clinical and not card.rule_based_interactions:
        f.append("Known label interactions must be rule-based, not learned.")
    if not card.logs_every_recommendation:
        f.append("Recommendations not logged with inputs and model version.")
    if not card.owner:
        f.append("No named accountable owner.")
    return GateResult(not f, f)


class ModelRegistry:
    def __init__(self) -> None:
        self._versions: dict[str, list[ModelCard]] = {}
        self._active: dict[str, int] = {}

    def register(self, card: ModelCard) -> None:
        self._versions.setdefault(card.model_id, []).append(card)

    def deploy(self, model_id: str, version: str, approver: str, clinical: bool) -> GateResult:
        versions = self._versions[model_id]
        idx = next(i for i, c in enumerate(versions) if c.version == version)
        gate = deployment_gate(versions[idx], clinical)
        if gate.passed:
            versions[idx].approver = approver
            versions[idx].deployed_at = datetime.now(timezone.utc).isoformat()
            self._active[model_id] = idx
        return gate

    def active(self, model_id: str) -> ModelCard | None:
        i = self._active.get(model_id)
        return None if i is None else self._versions[model_id][i]

    def rollback(self, model_id: str) -> ModelCard | None:
        """Revert to the previous deployed version."""
        i = self._active.get(model_id)
        if i is None:
            return None
        for j in range(i - 1, -1, -1):
            if self._versions[model_id][j].deployed_at:
                self._active[model_id] = j
                return self._versions[model_id][j]
        return None

    def drift(self, model_id: str, recent_inputs: list[float], threshold: float = 0.25) -> tuple[float, bool]:
        card = self.active(model_id)
        if not card or len(card.reference_inputs) < 20 or len(recent_inputs) < 20:
            return float("nan"), False
        psi = population_stability_index(card.reference_inputs, recent_inputs)
        return psi, psi > threshold


MATURITY_LEVELS = ["0. Ad hoc", "1. Instrumented", "2. Governed", "3. Predictive", "4. Controlled", "5. Learning"]


def assess_maturity(sensors_deployed: bool, calibration_schedule: bool, canonical_ids: bool,
                    validation_at_entry: bool, full_cycle_history: bool, validated_models_monitored: bool,
                    closed_loop_with_oversight: bool, outcomes_feed_breeding: bool) -> tuple[int, str, list[str]]:
    """Sequential by design: you cannot claim level 3 on level-1 data."""
    gates = [
        (sensors_deployed, "Deploy sensors with verified data capture."),
        (calibration_schedule and canonical_ids and validation_at_entry,
         "Governance: calibration schedule, canonical identifiers, validation at entry."),
        (full_cycle_history and validated_models_monitored, "Validated, monitored predictive models on a full cycle of data."),
        (closed_loop_with_oversight, "Closed-loop optimisation with constraints and human oversight."),
        (outcomes_feed_breeding, "Outcome data feeding breeding and product design."),
    ]
    level = 0
    for ok, _ in gates:
        if not ok:
            break
        level += 1
    todo = [msg for ok, msg in gates[level:level + 1]]
    return level, MATURITY_LEVELS[level], todo
