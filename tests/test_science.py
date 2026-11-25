"""Every worked example in Appendix A, reproduced as a regression test."""
import numpy as np
import pytest

from intelligent_plant.science import breeding, chemistry as ch, economics, environment as env, pharmacology as pk
from intelligent_plant.science import quality, stats
from intelligent_plant import simulation


def test_total_thc_factor_is_mw_ratio():
    assert ch.ACID_TO_NEUTRAL == pytest.approx(0.8772, abs=1e-4)
    assert ch.total_thc(1.0, 20.0) == pytest.approx(1.0 + 0.877 * 20.0, rel=1e-3)


def test_decarb_half_life_and_t99_at_110C():
    k = 1.83e-3
    assert ch.half_life_s(k) / 60 == pytest.approx(6.3, abs=0.05)
    assert ch.time_to_conversion_s(k, 0.99) / 60 == pytest.approx(41.9, abs=0.1)


def test_arrhenius_fit_matches_book():
    assert ch.THCA_FIT.Ea_kJ == pytest.approx(87.1, abs=0.1)
    assert ch.THCA_FIT.A == pytest.approx(1.40e9, rel=0.01)
    assert ch.time_to_conversion_s(ch.THCA_FIT.k(130), 0.99) / 60 == pytest.approx(10.5, abs=0.1)
    assert ch.time_to_conversion_s(ch.THCA_FIT.k(145), 0.99) / 60 == pytest.approx(4.1, abs=0.1)


def test_table_3_1_half_lives():
    for k, t_half in [(0.18e-3, 64.2), (0.66e-3, 17.5)]:
        assert ch.half_life_s(k) / 60 == pytest.approx(t_half, abs=0.1)


def test_vpd_worked_example_A8():
    assert env.saturation_vp_kpa(26) == pytest.approx(3.36, abs=0.01)
    assert env.vpd_air(26, 55) == pytest.approx(1.51, abs=0.01)
    assert env.vpd_leaf(24, 26, 55) == pytest.approx(1.14, abs=0.01)


@pytest.mark.parametrize("T,es", [(22, 2.64), (24, 2.98), (26, 3.36), (28, 3.78)])
def test_table_8_1_saturation_pressures(T, es):
    assert env.saturation_vp_kpa(T) == pytest.approx(es, abs=0.01)


def test_rh_inversion_round_trips():
    rh = env.rh_for_target_leaf_vpd(1.14, 24, 26)
    assert env.vpd_leaf(24, 26, rh) == pytest.approx(1.14, abs=1e-6)


def test_dli_worked_example():
    assert env.dli(900, 12) == pytest.approx(38.9, abs=0.05)


def test_label_uncertainty_A12():
    u = quality.label_relative_uncertainty([0.10, 0.10, 0.25, 0.15])
    assert u == pytest.approx(0.32, abs=0.005)
    assert (1 + u) / (1 - u) == pytest.approx(2.0, abs=0.1)


def test_280e_worked_example():
    t = economics.tax_under_280e(10e6, 5e6, 3.5e6)
    assert t.tax == pytest.approx(1.05e6) and t.effective_rate == pytest.approx(0.70)
    t2 = economics.tax_under_280e(10e6, 5e6, 3.5e6, subject_to_280e=False)
    assert t2.tax == pytest.approx(315_000) and t2.effective_rate == pytest.approx(0.21)


def test_combinatorics_A14():
    assert ch.combinatorial_space(30) == (1_073_741_824, 435)


def test_auc_scales_inversely_with_clearance():
    assert pk.auc(0.1, 10, 1 / 3) == pytest.approx(3 * pk.auc(0.1, 10, 1.0))
    rel, _ = pk.relative_thc_exposure("*3/*3")
    assert rel == pytest.approx(3.0)
    assert pk.relative_thc_exposure(None)[0] == 1.0


def test_feedback_stability_condition():
    assert pk.FeedbackModel(a=0.3, b=1, k=0.5).stable
    assert not pk.FeedbackModel(a=0.3, b=1, k=0.2).stable
    assert pk.FeedbackModel(0.3, 1, 0.5).deviation(1, 10) < 1 < pk.FeedbackModel(0.3, 1, 0.2).deviation(1, 10)


def test_one_compartment_peaks_and_decays():
    c = [pk.one_compartment_oral(t, 0.1, 10, 1.0, 10, 100) for t in np.linspace(0.01, 24, 200)]
    assert max(c) > c[-1] > 0 and c[0] < max(c)


def test_pid_drives_error_down():
    pid, x = env.PID(0.5, 0.05, 0.0), 0.0
    for _ in range(200):
        x += pid.step(10 - x, 1.0)
    assert abs(10 - x) < 0.5


def test_mpc_beats_no_action():
    mpc, model = env.MPCController(), env.RoomModel()
    T, rh = 28.0, 45.0
    T0, rh0 = T, rh
    for _ in range(6):
        u = mpc.solve(T, rh, 25, 1.4, 2.0, [0.1]).action
        T, rh = model.step(T, rh, u)
    for _ in range(6):
        T0, rh0 = model.step(T0, rh0, (0, 0, 0))
    err = lambda T, rh: (T - 25) ** 2 + 25 * (env.vpd_leaf(T - 2, T, rh) - 1.4) ** 2
    assert err(T, rh) < err(T0, rh0)


def test_breeding_method_matches_architecture():
    X, y = simulation.oligogenic_trait()
    assert breeding.choose_method(X, y).method == "marker_assisted_selection"
    X, y = simulation.polygenic_trait()
    assert breeding.choose_method(X, y).method == "genomic_selection"


def test_partial_pooling_shrinks_and_abstains():
    few = stats.partial_pool([4.0], pop_mean=1.0, pop_tau=0.5, sigma=2.0)
    many = stats.partial_pool([4.0] * 40, pop_mean=1.0, pop_tau=0.5, sigma=2.0)
    assert few.abstain and not many.abstain
    assert few.mean < many.mean and (few.hi90 - few.lo90) > (many.hi90 - many.lo90)


def test_n_of_1_detects_true_difference():
    sched = stats.n_of_1_schedule(6, seed=1)
    outcomes = [3.0 if s == "A" else 1.0 for s in sched]
    outcomes = [o + 0.1 * ((i * 7) % 3 - 1) for i, o in enumerate(outcomes)]
    r = stats.analyse_n_of_1(sched, outcomes)
    assert r.conclusive and r.mean_difference == pytest.approx(2.0, abs=0.2)


def test_factorial_interaction_identifies_synergy():
    rng = np.random.default_rng(0)
    A = np.repeat([0, 1, 0, 1], 60)
    B = np.repeat([0, 0, 1, 1], 60)
    Y = 1 + 0.5 * A + 0.5 * B + 1.5 * A * B + rng.normal(0, 0.5, len(A))
    assert stats.factorial_interaction(A, B, Y)["verdict"] == "synergy"
    Y2 = 1 + 0.5 * A + 0.5 * B + rng.normal(0, 0.5, len(A))
    assert stats.factorial_interaction(A, B, Y2)["verdict"] == "additive / not identified"


def test_calibration_coverage():
    assert stats.interval_coverage([(0, 1)] * 10, [0.5] * 9 + [2]) == pytest.approx(0.9)


def test_coa_audit_catches_gc_and_failed_panel():
    bad = simulation.make_coa("C", "B", 0.5, 20, 0, 0, method="GC",
                              panels={**simulation.PASS_PANELS, "pesticides": "fail"})
    r = quality.audit_coa(bad)
    assert not r.passed and r.trust_score < 1
    assert quality.audit_coa(simulation.make_coa("C", "B", 0.5, 20, 0, 0)).passed


def test_spc_flags_shift():
    r = quality.shewhart([20, 20.2, 19.9, 20.1, 20.0, 20.1, 24.0], baseline_n=6)
    assert 6 in r.out_of_control


def test_decarb_planner_reports_physics_not_assumption():
    """With Ea_vol < Ea_decarb a hotter, shorter run retains more (see docstring)."""
    plans = ch.plan_decarboxylation(0.95, temps_c=(95, 105, 115))
    assert plans[0].temperature_c == 115
    hot_vol = ch.VolatileLossModel(A=1e14, Ea_J=120_000)   # Ea_vol > Ea_decarb reverses the conclusion
    plans2 = ch.plan_decarboxylation(0.95, temps_c=(95, 105, 115), volatile=hot_vol)
    assert plans2[0].temperature_c == 95


def test_sustainability_metrics():
    m = economics.sustainability_metrics(100_000, 50_000, 40_000, 200_000, 50_000, 1, 0.5, 0.4, 0, 25, [500, 520, 480])
    assert m.energy_kwh_per_kg == 4000 and m.hvac_share_pct == 40 and m.water_l_per_kg == 6000
    assert env.grams_per_kwh(2.7, 0.9, 1.0) == pytest.approx(2.7 * 3.6 * 0.9)


def test_n_of_1_two_pairs_not_falsely_conclusive():
    from intelligent_plant.science.stats import analyse_n_of_1
    r = analyse_n_of_1(["A", "B", "B", "A"], [5, 3, 4, 5])
    assert r.lo90 < 0 < r.hi90 and not r.conclusive


def test_mpc_fail_safe_does_not_cool_a_cold_room():
    from intelligent_plant.science.environment import MPCController
    res = MPCController().solve(T=15.0, rh=55, T_target=24, vpd_target=1.0, leaf_offset_c=1.0, price_per_kwh=[0.2])
    assert res.action[0] == 0.0


def test_280e_loss_never_yields_negative_tax():
    assert economics.tax_under_280e(100, 200, 10).tax == 0.0
    assert economics.tax_under_280e(100, 50, 80, subject_to_280e=False).tax == 0.0


@pytest.mark.parametrize("method,flagged", [("GC", True), ("GC-FID", True), ("gc-ms", True),
                                            ("GC-MS derivatized", False), ("HPLC", False)])
def test_coa_audit_flags_every_underivatized_gc_variant(method, flagged):
    r = quality.audit_coa(simulation.make_coa("C", "B", 0.5, 20, 0, 0, method=method))
    assert any("GC without derivatization" in m for _, m in r.findings) == flagged
