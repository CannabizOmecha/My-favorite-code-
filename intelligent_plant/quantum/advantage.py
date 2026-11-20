"""Quantum-vs-classical decision gate and the operational problems it serves.

Rule (Chapter 15.3, "data infrastructure first, classical ML second, quantum
when and if it matures"): every problem is solved classically AND, when it
fits, by the quantum routine. The quantum answer is used only if it is at
least as good on the problem's own objective; otherwise the classical answer
is used and the comparison is logged. A hardware backend plugs in through
`QuantumBackend` and is held to exactly the same comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .algorithms import OptimisationResult, qaoa, solve_qubo_annealing, solve_qubo_exact

EXACT_LIMIT = 20          # brute-force classical ground truth up to 2**20 states
SIM_LIMIT = 16            # QAOA statevector simulation kept small enough to run every cycle


class QuantumBackend(Protocol):
    """Implement this to run on real hardware (e.g. via a vendor SDK)."""
    name: str

    def solve_qubo(self, Q: np.ndarray) -> OptimisationResult: ...


class SimulatorBackend:
    name = "statevector_simulator"

    def __init__(self, p: int = 2, restarts: int = 4, seed: int = 0):
        self.p, self.restarts, self.seed = p, restarts, seed

    def solve_qubo(self, Q: np.ndarray) -> OptimisationResult:
        return qaoa(Q, p=self.p, restarts=self.restarts, seed=self.seed)


@dataclass
class GateDecision:
    chosen: OptimisationResult
    classical: OptimisationResult
    quantum: OptimisationResult | None
    quantum_used: bool
    rationale: str
    log: dict = field(default_factory=dict)


def decide(Q: np.ndarray, backend: QuantumBackend | None = None) -> GateDecision:
    n = Q.shape[0]
    classical = solve_qubo_exact(Q) if n <= EXACT_LIMIT else solve_qubo_annealing(Q)
    backend = backend or (SimulatorBackend() if n <= SIM_LIMIT else None)
    quantum = backend.solve_qubo(Q) if backend else None
    if quantum is not None and quantum.best_energy <= classical.best_energy + 1e-9:
        use_q = backend.name != "statevector_simulator"   # on a simulator, a tie is not an advantage
        why = ("Quantum matched the classical optimum" + ("" if use_q else
               "; classical answer used because simulation offers no speed or cost advantage") + ".")
    elif quantum is not None:
        use_q = False
        why = (f"Quantum result {quantum.best_energy:.4f} worse than classical {classical.best_energy:.4f}; "
               f"classical used.")
    else:
        use_q, why = False, f"{n} variables exceeds simulation limit; classical only."
    chosen = quantum if use_q else classical
    return GateDecision(chosen, classical, quantum, use_q, why,
                        {"n": n, "backend": getattr(backend, "name", None),
                         "quantum_success_probability": quantum.success_probability if quantum else None,
                         "quantum_approximation_ratio": quantum.approximation_ratio if quantum else None})


def portfolio_qubo(value: np.ndarray, kinship: np.ndarray, k: int, diversity_weight: float = 1.0,
                   cardinality_weight: float | None = None) -> np.ndarray:
    """Choose exactly k cultivars to advance: maximise patient-first value while
    penalising pairwise kinship (Chapter 9.4: protect diversity).
        min  -sum_i v_i x_i + w sum_{i<j} K_ij x_i x_j + A (sum_i x_i - k)^2
    The penalty A is set above the largest possible gain so the constraint binds."""
    n = len(value)
    A = cardinality_weight if cardinality_weight is not None else 2.0 * (np.abs(value).sum()
                                                                         + diversity_weight * np.abs(kinship).sum())
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] += -value[i] + A * (1 - 2 * k)
        for j in range(i + 1, n):
            Q[i, j] += diversity_weight * kinship[i, j] + 2 * A
    return Q
