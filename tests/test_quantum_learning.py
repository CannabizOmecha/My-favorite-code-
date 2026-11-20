"""Quantum mechanics is checked against first principles; every quantum
algorithm against its classical answer; the learning loop against its
governance rules."""
import asyncio
import math

import numpy as np
import pytest

from intelligent_plant.learning.continual import ChampionChallengerLoop, RecursiveBayesianRegressor, evaluate, \
    t_quantile_95
from intelligent_plant.quantum import algorithms as qa
from intelligent_plant.quantum.advantage import decide, portfolio_qubo
from intelligent_plant.quantum.statevector import MAX_QUBITS, Statevector, bell_pair, pauli_matrix


# --------------------------------------------------------------- quantum physics
def test_superposition_is_uniform_and_normalised():
    sv = Statevector(3)
    for q in range(3):
        sv.h(q)
    assert np.allclose(sv.probabilities(), 1 / 8)


def test_bell_pair_is_maximally_entangled():
    b = bell_pair()
    assert np.allclose(b.probabilities(), [0.5, 0, 0, 0.5])
    assert b.entanglement_entropy([0]) == pytest.approx(1.0)
    assert b.expectation_pauli("ZZ") == pytest.approx(1.0) and b.expectation_pauli("XX") == pytest.approx(1.0)
    assert abs(Statevector(2).h(0).h(1).entanglement_entropy([0])) < 1e-12     # product state: no entanglement


def test_measurement_follows_born_rule():
    counts = Statevector(1).ry(0, 2 * math.asin(math.sqrt(0.3))).sample(20000, np.random.default_rng(0))
    assert counts.get(1, 0) / 20000 == pytest.approx(0.3, abs=0.015)


def test_interference_returns_to_zero():
    assert Statevector(1).h(0).h(0).probabilities()[0] == pytest.approx(1.0)


def test_gates_are_unitary_and_expectations_consistent():
    rng = np.random.default_rng(1)
    sv = Statevector(3)
    for q in range(3):
        sv.ry(q, rng.uniform(0, 3)).rz(q, rng.uniform(0, 3))
    sv.cnot(0, 2).cz(1, 2).rzz(0, 1, 0.4)
    assert np.linalg.norm(sv.amps) == pytest.approx(1.0)
    for p in ("ZII", "XYZ", "IXX"):
        direct = float(np.real(np.vdot(sv.amps, pauli_matrix(p) @ sv.amps)))
        assert sv.expectation_pauli(p) == pytest.approx(direct, abs=1e-10)


def test_simulator_refuses_intractable_sizes():
    with pytest.raises(ValueError):
        Statevector(MAX_QUBITS + 1)


# --------------------------------------------------------------- algorithms vs classical
def test_vqe_reaches_h2_ground_state_within_chemical_accuracy():
    r = qa.vqe(qa.H2_HAMILTONIAN)
    assert r.chemical_accuracy
    assert r.exact_energy + qa.H2_NUCLEAR_REPULSION == pytest.approx(-1.13730, abs=1e-4)   # textbook H2 energy


def test_grover_quadratic_query_advantage():
    g = qa.grover_search(10, {777})
    assert g.found == 777 and g.success_probability > 0.99
    assert g.quantum_oracle_queries == 25 and g.classical_expected_queries == 512


def test_qaoa_beats_random_guessing_and_matches_classical_optimum():
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(7, 7)); Q = (Q + Q.T) / 2
    q, c = qa.qaoa(Q, p=2), qa.solve_qubo_exact(Q)
    assert q.success_probability > 10 / 2 ** 7          # far above uniform random (1/128)
    assert q.approximation_ratio > 0.75
    assert q.best_energy == pytest.approx(c.best_energy)


def test_annealing_matches_exact_on_small_problem():
    rng = np.random.default_rng(4)
    Q = rng.normal(size=(10, 10)); Q = (Q + Q.T) / 2
    assert qa.solve_qubo_annealing(Q, sweeps=5000).best_energy == pytest.approx(qa.solve_qubo_exact(Q).best_energy)


def test_gate_never_prefers_a_simulator_without_advantage():
    rng = np.random.default_rng(5)
    Q = rng.normal(size=(6, 6)); Q = (Q + Q.T) / 2
    d = decide(Q)
    assert not d.quantum_used and d.chosen.method == "classical_exhaustive" and d.quantum is not None


def test_gate_accepts_hardware_only_when_it_matches_classical():
    rng = np.random.default_rng(6)
    Q = rng.normal(size=(5, 5)); Q = (Q + Q.T) / 2
    exact = qa.solve_qubo_exact(Q)

    class GoodHW:
        name = "vendor_qpu"
        def solve_qubo(self, Q):
            return qa.OptimisationResult("qpu", exact.best_bits, exact.best_energy)

    class BadHW(GoodHW):
        def solve_qubo(self, Q):
            return qa.OptimisationResult("qpu", [0] * 5, 0.0 if exact.best_energy < 0 else 99.0)

    assert decide(Q, GoodHW()).quantum_used
    assert not decide(Q, BadHW()).quantum_used


def test_portfolio_qubo_enforces_k_and_diversity():
    value = np.array([5.0, 5.0, 4.0, 3.9, 1.0])
    K = np.zeros((5, 5)); K[0, 1] = K[1, 0] = 1.0
    best = qa.solve_qubo_exact(portfolio_qubo(value, K, k=3, diversity_weight=4.0)).best_bits
    assert sum(best) == 3 and not (best[0] and best[1])          # never both siblings


# --------------------------------------------------------------- learning
def _stream(n, rng, shift_at=None):
    w = np.array([1.5, -2.0, 0.5])
    for i in range(n):
        x = rng.normal(size=3) + (np.array([2.5, 0, 0]) if shift_at and i >= shift_at else 0)
        wt = w if not shift_at or i < shift_at else np.array([0.5, -2.0, 0.5])
        yield x, float(x @ wt + 0.7 + rng.normal(0, 0.5))


def test_t_quantile():
    assert t_quantile_95(1e9) == pytest.approx(1.6449, abs=1e-3)
    assert t_quantile_95(10) == pytest.approx(1.8125, abs=3e-3)


def test_recursive_bayes_recovers_truth_and_is_calibrated():
    rng = np.random.default_rng(0)
    data = list(_stream(2000, rng))
    m = RecursiveBayesianRegressor(3).fit([d[0] for d in data[:1500]], [d[1] for d in data[:1500]])
    assert np.allclose(m.mean, [0.7, 1.5, -2.0, 0.5], atol=0.05)
    ev = evaluate(m, [d[0] for d in data[1500:]], [d[1] for d in data[1500:]])
    assert 0.86 <= ev.coverage90 <= 0.94 and ev.rmse == pytest.approx(0.5, abs=0.05)


def test_uncertainty_shrinks_with_data():
    rng = np.random.default_rng(1)
    data = list(_stream(500, rng))
    few = RecursiveBayesianRegressor(3).fit([d[0] for d in data[:5]], [d[1] for d in data[:5]])
    many = RecursiveBayesianRegressor(3).fit([d[0] for d in data], [d[1] for d in data])
    w = lambda m: (lambda p: p[2] - p[1])(m.predict([0, 0, 0]))
    assert w(few) > 1.5 * w(many)


def test_loop_refuses_to_learn_from_its_own_predictions():
    loop = ChampionChallengerLoop("m", 3, "owner", "s", ["x"])
    with pytest.raises(ValueError):
        loop.observe([0, 0, 0], 1.0, source="model_prediction")


def test_loop_abstains_until_validated():
    loop = ChampionChallengerLoop("m", 3, "owner", "s", ["x"])
    loop.observe([0, 0, 0], 1.0)
    assert "Waiting" in loop.step().reason and loop.predict([0, 0, 0]) is None


@pytest.mark.parametrize("seed", range(4))
def test_loop_detects_regime_change_suspends_and_recovers(seed):
    rng = np.random.default_rng(seed)
    loop = ChampionChallengerLoop("yield", 3, "owner", "s", ["x"])
    states = []
    for i, (x, y) in enumerate(_stream(1300, rng, shift_at=800)):
        loop.observe(x, y)
        if i % 100 == 99:
            loop.step()
            states.append((i + 1, loop.suspended, loop.champion_version))
    stable = [s for s in states if 200 <= s[0] <= 800]
    assert not any(s[1] for s in stable)                     # no false alarms while stable
    assert any(s[1] for s in states if 800 < s[0] <= 1000)   # change detected -> abstain
    assert not loop.suspended and loop.history[-1].champion.rmse < 0.7    # recovered to the noise floor


def test_clinical_promotion_waits_for_owner():
    rng = np.random.default_rng(2)
    loop = ChampionChallengerLoop("resp", 3, "treating_clinician", "s", ["pediatric"], clinical=True)
    for x, y in _stream(200, rng):
        loop.observe(x, y)
    r = loop.step()
    assert r.pending_human_approval and not r.promoted and loop.predict([0, 0, 0]) is None
    with pytest.raises(PermissionError):
        loop.approve_pending("data_steward")
    assert loop.approve_pending("treating_clinician") and loop.predict([0, 0, 0]) is not None


def test_agents_in_full_cycle():
    from intelligent_plant.orchestrator import IntelligentEnterprise
    from intelligent_plant.simulation import NOW, build_world
    ent = IntelligentEnterprise(use_llm=False)
    r = asyncio.run(ent.run_cycle(build_world(ent), now=NOW))
    port = next(d for d in r.dispositions if d.proposal.agent == "quantum_portfolio")
    sel = port.proposal.payload["selected"]
    assert len(sel) == 3 and "Plains High-THC" not in sel
    assert not {"Fairland CBD-20", "Fairland CBD-20 sib"} <= set(sel)
    assert port.outcome == "queued"                          # a breeder decides
    assert ent.learning_loops["yield_g_m2"].champion_version == 1
