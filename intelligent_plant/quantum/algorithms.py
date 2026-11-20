"""Quantum algorithms on the exact simulator, each paired with the classical
answer it must beat.

  * QAOA   - combinatorial optimisation of a QUBO via an Ising cost Hamiltonian
  * Grover - amplitude amplification for unstructured search
  * VQE    - variational ground-state energy of a qubit Hamiltonian (H2 example)

First-principles honesty (Chapter 15): on a classical simulator these are
never faster than classical methods; they are here so the pipeline, the
benchmarks and the acceptance tests exist before hardware is ready.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .statevector import MAX_QUBITS, Statevector, pauli_matrix


# ------------------------------------------------------------------ optimiser
def nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray, step: float = 0.3, iters: int = 300,
                tol: float = 1e-9) -> tuple[np.ndarray, float]:
    """Derivative-free simplex minimiser (no SciPy dependency)."""
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


# ------------------------------------------------------------------ QUBO / Ising
def qubo_energies(Q: np.ndarray) -> np.ndarray:
    """E(x) = x^T Q x for every bitstring x (qubit i = bit i of the index)."""
    n = Q.shape[0]
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    return np.einsum("ki,ij,kj->k", bits, Q, bits)


def bits_of(index: int, n: int) -> list[int]:
    return [(index >> i) & 1 for i in range(n)]


@dataclass
class OptimisationResult:
    method: str
    best_bits: list[int]
    best_energy: float
    optimum_energy: float | None = None
    success_probability: float | None = None   # P(measuring an optimal string)
    approximation_ratio: float | None = None
    evaluations: int = 0
    note: str = ""


def solve_qubo_exact(Q: np.ndarray) -> OptimisationResult:
    """Classical brute force: the ground truth every quantum result is scored against."""
    E = qubo_energies(Q)
    k = int(np.argmin(E))
    return OptimisationResult("classical_exhaustive", bits_of(k, Q.shape[0]), float(E[k]), float(E[k]),
                              evaluations=len(E))


def solve_qubo_annealing(Q: np.ndarray, sweeps: int = 2000, seed: int = 0) -> OptimisationResult:
    """Classical simulated annealing: the scalable baseline for n beyond brute force."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    Qs = Q + Q.T
    x = rng.integers(0, 2, n).astype(float)
    e = float(x @ Q @ x)
    best_x, best_e = x.copy(), e
    for s in range(sweeps):
        T = 2.0 * (1 - s / sweeps) + 1e-3
        i = rng.integers(n)
        d = (1 - 2 * x[i]) * (Qs[i] @ x - Qs[i, i] * x[i] + Q[i, i])
        if d < 0 or rng.random() < math.exp(-d / T):
            x[i] = 1 - x[i]
            e += d
            if e < best_e:
                best_x, best_e = x.copy(), e
    return OptimisationResult("classical_annealing", best_x.astype(int).tolist(), float(best_e), evaluations=sweeps)


def qaoa(Q: np.ndarray, p: int = 2, restarts: int = 4, seed: int = 0, shots: int = 2048) -> OptimisationResult:
    """QAOA: |gamma,beta> = prod_l U_M(beta_l) U_C(gamma_l) H^n |0>.
    U_C is diagonal (the QUBO energy, i.e. the Ising cost after the x=(1-z)/2
    map, which produces the RZ and RZZ entangling terms); U_M = prod RX(2 beta).
    Parameters are tuned classically to minimise <C>; the result is measured."""
    n = Q.shape[0]
    if n > MAX_QUBITS:
        raise ValueError("too many qubits for exact simulation")
    E = qubo_energies(Q)
    scale = float(np.max(np.abs(E))) or 1.0
    En = E / scale
    evals = 0

    def state(theta: np.ndarray) -> Statevector:
        sv = Statevector(n)
        for q in range(n):
            sv.h(q)                                    # uniform superposition over all 2**n candidates
        for l in range(p):
            sv.phase_diagonal(theta[l] * En)          # cost layer: phase kick by energy (entangling)
            for q in range(n):
                sv.rx(q, 2 * theta[p + l])            # mixer layer: interference between neighbours
        return sv

    def cost(theta: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        return float(state(theta).probabilities() @ En)

    rng = np.random.default_rng(seed)
    best_theta, best_val = None, float("inf")
    for _ in range(restarts):
        t0 = np.concatenate([rng.uniform(0, math.pi, p), rng.uniform(0, math.pi / 2, p)])
        t, v = nelder_mead(cost, t0)
        if v < best_val:
            best_theta, best_val = t, v
    sv = state(best_theta)
    probs = sv.probabilities()
    counts = sv.sample(shots, rng)
    sampled_best = min(counts, key=lambda k: E[k])
    opt = float(E.min())
    p_opt = float(probs[np.isclose(E, opt)].sum())
    worst = float(E.max())
    expected = float(probs @ E)
    ratio = (worst - expected) / (worst - opt) if worst != opt else 1.0
    return OptimisationResult(f"qaoa_p{p}_statevector", bits_of(sampled_best, n), float(E[sampled_best]), opt,
                              p_opt, ratio, evals,
                              f"Exact simulation of {n} qubits; uniform random guessing finds the optimum with "
                              f"P={np.isclose(E, opt).sum() / len(E):.3f}, QAOA with P={p_opt:.3f}.")


# ------------------------------------------------------------------ Grover
@dataclass
class GroverResult:
    found: int
    iterations: int
    success_probability: float
    classical_expected_queries: float
    quantum_oracle_queries: int


def grover_search(n: int, marked: set[int], shots: int = 1, seed: int = 0) -> GroverResult:
    """Amplitude amplification. ~ (pi/4) sqrt(N/M) oracle calls vs ~ N/(2M)
    classically: a quadratic, not exponential, query advantage."""
    N, M = 2 ** n, len(marked)
    if not 0 < M < N:
        raise ValueError("need 0 < |marked| < 2**n")
    k = max(1, int(math.floor(math.pi / 4 * math.sqrt(N / M))))
    sv = Statevector(n)
    for q in range(n):
        sv.h(q)
    oracle = np.zeros(N)
    oracle[list(marked)] = math.pi                    # phase flip on solutions
    for _ in range(k):
        sv.phase_diagonal(oracle)
        mean = sv.amps.mean()
        sv.amps = 2 * mean - sv.amps                  # diffusion: inversion about the mean
    probs = sv.probabilities()
    found = int(np.random.default_rng(seed).choice(N, p=probs))
    return GroverResult(found, k, float(probs[list(marked)].sum()), N / (2 * M), k)


# ------------------------------------------------------------------ VQE
# Two-qubit H2 Hamiltonian at 0.735 Angstrom (STO-3G, parity mapping with
# two-qubit reduction), coefficients as published in standard VQE tutorials.
# Electronic energy only; add nuclear repulsion 0.7199689 Ha for total energy.
H2_HAMILTONIAN = {"II": -1.052373245772859, "ZI": 0.39793742484318045, "IZ": -0.39793742484318045,
                  "ZZ": -0.01128010425623538, "XX": 0.18093119978423156}
H2_NUCLEAR_REPULSION = 0.7199689944489797


def hamiltonian_matrix(terms: dict[str, float]) -> np.ndarray:
    return sum(c * pauli_matrix(p) for p, c in terms.items())


@dataclass
class VQEResult:
    energy: float
    exact_energy: float
    error: float
    parameters: list[float]
    evaluations: int
    chemical_accuracy: bool   # |error| < 1.6 mHa


def vqe(terms: dict[str, float], layers: int = 2, restarts: int = 3, seed: int = 0) -> VQEResult:
    """Hardware-efficient RY + CNOT ansatz; energy = sum_P c_P <P>."""
    n = len(next(iter(terms)))
    evals = 0

    def energy(theta: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        sv = Statevector(n)
        k = 0
        for _ in range(layers):
            for q in range(n):
                sv.ry(q, theta[k]); k += 1
            for q in range(n - 1):
                sv.cnot(q, q + 1)                     # entangle neighbouring qubits
        for q in range(n):
            sv.ry(q, theta[k]); k += 1
        return sum(c * sv.expectation_pauli(p) for p, c in terms.items())

    rng = np.random.default_rng(seed)
    nparam = n * (layers + 1)
    best_t, best_e = None, float("inf")
    for _ in range(restarts):
        t, e = nelder_mead(energy, rng.uniform(-math.pi, math.pi, nparam), step=0.5, iters=600)
        if e < best_e:
            best_t, best_e = t, e
    exact = float(np.linalg.eigvalsh(hamiltonian_matrix(terms)).min())
    return VQEResult(best_e, exact, best_e - exact, best_t.tolist(), evals, abs(best_e - exact) < 1.6e-3)


def exhaustive_assignments(n: int):
    return itertools.product((0, 1), repeat=n)
