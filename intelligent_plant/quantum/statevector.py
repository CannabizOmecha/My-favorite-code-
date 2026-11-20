"""Exact state-vector quantum simulator (NumPy only).

This is real quantum mechanics, computed classically: a register of n qubits
is a vector of 2**n complex amplitudes, gates are unitaries, measurement
follows the Born rule. Superposition, interference and entanglement are all
represented exactly. The price is the one Chapter 15 states from first
principles: memory and time grow as 2**n, so classical simulation stops being
practical around 25-30 qubits. Nothing here is a quantum speed-up. It is the
correct, testable reference that a hardware backend must reproduce.

Qubit 0 is the least-significant bit of the basis-state index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_QUBITS = 22  # 2**22 complex128 amplitudes = 64 MiB; refuse beyond this rather than exhaust memory

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def rx(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t: float) -> np.ndarray:
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


@dataclass
class Statevector:
    n: int
    amps: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not 1 <= self.n <= MAX_QUBITS:
            raise ValueError(f"{self.n} qubits outside simulable range 1..{MAX_QUBITS} (state size 2**n)")
        if self.amps is None:
            self.amps = np.zeros(2 ** self.n, dtype=complex)
            self.amps[0] = 1.0

    # ---------------------------------------------------------------- gates
    def _tensor(self) -> np.ndarray:
        # axis k of the tensor is qubit (n-1-k), so qubit q lives on axis n-1-q
        return self.amps.reshape([2] * self.n)

    def apply(self, U: np.ndarray, q: int) -> "Statevector":
        t = np.moveaxis(self._tensor(), self.n - 1 - q, 0)
        t = np.tensordot(U, t, axes=([1], [0]))
        self.amps = np.moveaxis(t, 0, self.n - 1 - q).reshape(-1)
        return self

    def apply2(self, U4: np.ndarray, q_ctrl: int, q_tgt: int) -> "Statevector":
        """Two-qubit unitary in basis |q_ctrl q_tgt> (q_ctrl is the high bit)."""
        a, b = self.n - 1 - q_ctrl, self.n - 1 - q_tgt
        t = np.moveaxis(self._tensor(), (a, b), (0, 1))
        shp = t.shape
        t = (U4 @ t.reshape(4, -1)).reshape(shp)
        self.amps = np.moveaxis(t, (0, 1), (a, b)).reshape(-1)
        return self

    def h(self, q: int) -> "Statevector":
        return self.apply(H, q)

    def x(self, q: int) -> "Statevector":
        return self.apply(X, q)

    def z(self, q: int) -> "Statevector":
        return self.apply(Z, q)

    def rx(self, q: int, t: float) -> "Statevector":
        return self.apply(rx(t), q)

    def ry(self, q: int, t: float) -> "Statevector":
        return self.apply(ry(t), q)

    def rz(self, q: int, t: float) -> "Statevector":
        return self.apply(rz(t), q)

    def cnot(self, c: int, t: int) -> "Statevector":
        U = np.eye(4, dtype=complex)
        U[[2, 3]] = U[[3, 2]]
        return self.apply2(U, c, t)

    def cz(self, a: int, b: int) -> "Statevector":
        return self.apply2(np.diag([1, 1, 1, -1]).astype(complex), a, b)

    def rzz(self, a: int, b: int, t: float) -> "Statevector":
        """exp(-i t/2 Z_a Z_b): the entangling cost layer of QAOA."""
        return self.apply2(np.diag(np.exp(-1j * t / 2 * np.array([1, -1, -1, 1]))), a, b)

    def phase_diagonal(self, phases: np.ndarray) -> "Statevector":
        """Multiply each basis state by exp(-i * phases[k]) (diagonal cost unitary)."""
        self.amps = self.amps * np.exp(-1j * phases)
        return self

    # ---------------------------------------------------------------- readout
    def probabilities(self) -> np.ndarray:
        p = np.abs(self.amps) ** 2
        return p / p.sum()

    def sample(self, shots: int, rng: np.random.Generator | None = None) -> dict[int, int]:
        """Born-rule measurement in the computational basis."""
        rng = rng or np.random.default_rng()
        idx = rng.choice(len(self.amps), size=shots, p=self.probabilities())
        vals, counts = np.unique(idx, return_counts=True)
        return dict(zip(vals.tolist(), counts.tolist()))

    def expectation_pauli(self, pauli: str) -> float:
        """<psi| P |psi> for a Pauli string; pauli[0] acts on qubit 0."""
        phi = Statevector(self.n, self.amps.copy())
        for q, p in enumerate(pauli):
            if p != "I":
                phi.apply(PAULI[p], q)
        return float(np.real(np.vdot(self.amps, phi.amps)))

    def entanglement_entropy(self, subsystem: list[int]) -> float:
        """Von Neumann entropy (bits) of the reduced state of `subsystem`.
        0 for a product state; 1 bit for one qubit of a Bell pair."""
        keep = [self.n - 1 - q for q in subsystem]
        rest = [a for a in range(self.n) if a not in keep]
        t = np.transpose(self._tensor(), keep + rest).reshape(2 ** len(keep), -1)
        s = np.linalg.svd(t, compute_uv=False) ** 2
        s = s[s > 1e-15]
        return max(0.0, float(-(s * np.log2(s)).sum()))


def bell_pair() -> Statevector:
    """(|00> + |11>)/sqrt(2): superposition (H) then entanglement (CNOT)."""
    return Statevector(2).h(0).cnot(0, 1)


def pauli_matrix(pauli: str) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for p in reversed(pauli):           # qubit 0 is least significant -> rightmost factor
        m = np.kron(m, PAULI[p])
    return m
