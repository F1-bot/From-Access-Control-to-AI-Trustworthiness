from aitcert import feasibility as FZ
from aitcert import stats as S


def test_best_bounds_match_exact_bounds():
    for n in (1, 7, 9, 14, 58):
        assert abs(FZ.best_bound(n, ">=", 0.05) - S.cp_lower(n, n, 0.05)) < 1e-12
        assert abs(FZ.best_bound(n, "<=", 0.05) - S.cp_upper(0, n, 0.05)) < 1e-12
    assert abs(FZ.best_bound(7, ">=", 0.05) - 0.6518) < 1e-4      # seven poisoned traces
    assert abs(FZ.best_bound(9, ">=", 0.05) - 0.7169) < 1e-4      # nine drifted days
    assert abs(FZ.best_bound(1, "<=", 0.05) - 0.95) < 1e-12       # one in-distribution day


def test_undecidable_tests_are_flagged():
    assert FZ.check_proportion("drift TPR", 9, 0.90, ">=", 0.05)["verdict"] == "not decidable"
    assert FZ.check_proportion("drift FPR", 1, 0.05, "<=", 0.05)["verdict"] == "not decidable"
    assert FZ.check_proportion("poison detection", 14, 0.80, ">=", 0.05)["decidable"]
    assert not FZ.check_proportion("poison detection", 13, 0.80, ">=", 0.05)["decidable"]


def test_trivial_baseline_and_power():
    r = FZ.check_proportion("localisation", 38, 0.60, ">=", 0.05, baseline=0.79)
    assert r["verdict"] == "inadmissible threshold"
    assert FZ.n_min(0.5, ">=", 0.2, 0.05) == 42 and FZ.n_min(0.5, ">=", 0.1, 0.05) == 169
    r = FZ.check_proportion("recall", 58, 0.50, ">=", 0.05, delta=0.2, deff=2.2)
    assert r["verdict"] == "underpowered" and r["power_at_delta"] < 0.8
    assert FZ.check_proportion("recall", 58, 0.50, ">=", 0.05, delta=0.2)["verdict"] == "feasible"
    md = FZ.min_detectable_margin(58, 0.5, ">=", 0.05)
    assert 0.15 < md < 0.20


def test_rates():
    r = FZ.check_rate("false alarms", 95.96, 0.25, 0.05, pilot_rate=0.28)
    assert r["decidable"] and abs(r["best_bound"] - 2.995732 / 95.96) < 1e-6
    assert r["p_accept_at_pilot"] < 0.01 and r["verdict"] == "low predicted acceptance"
    assert abs(FZ.p_accept_rate(0.10, 60, 0.25, 0.05) - 0.8) < 0.06
    t10, t15 = FZ.exposure_for_rate(0.10, 0.25, 0.05), FZ.exposure_for_rate(0.15, 0.25, 0.05)
    assert 50 <= t10 <= 70 and 125 <= t15 <= 155
    lam = FZ.max_rate_at_power(95.96, 0.25, 0.05)
    assert FZ.p_accept_rate(lam, 95.96, 0.25, 0.05) >= 0.8 > FZ.p_accept_rate(lam + 0.005, 95.96, 0.25, 0.05)
