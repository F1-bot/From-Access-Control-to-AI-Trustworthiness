import itertools, math, json, os
import numpy as np
import networkx as nx
from scipy import stats
from aitcert import stats as S
from aitcert.model import Catalogue, parse_profile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_clopper_pearson_known_values():
    # zero successes: one-sided 95% upper bound = 1 - 0.05^(1/n) (rule of three ~ 3/n)
    for n in (10, 30, 100):
        assert abs(S.cp_upper(0, n, 0.05) - (1 - 0.05 ** (1 / n))) < 1e-9
    # symmetry
    assert abs(S.cp_lower(7, 10, 0.05) - (1 - S.cp_upper(3, 10, 0.05))) < 1e-9
    lo, hi = S.cp_interval(50, 100, 0.05)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61


def test_consumer_risk_bounded_by_alpha():
    # exactness: for p just below tau, acceptance probability of the LCB rule <= alpha
    for n, tau, a in [(58, 0.7, 0.05), (200, 0.9, 0.01), (30, 0.5, 0.10)]:
        oc = S.oc_curve([tau - 1e-9], n, tau, a)[0]
        assert oc <= a + 1e-9, (n, tau, a, oc)


def test_naive_rule_consumer_risk_large():
    oc = S.oc_curve([0.6999], 58, 0.7, None)[0]
    assert oc > 0.4


def test_sample_sizes():
    assert S.n_hoeffding(0.05, 0.1) == math.ceil(math.log(20) / 0.02)
    assert S.n_zero_failure(0.05, 0.01) == 299
    n = S.n_exact_power(0.7, 0.15, 0.05, 0.2)
    k = S.min_successes(n, 0.7, 0.05)
    assert 1 - stats.binom.cdf(k - 1, n, 0.85) >= 0.8


def test_poisson_bounds():
    ub = S.poisson_upper(0, 10.0, 0.05)
    assert abs(ub - (-math.log(0.05) / 10.0)) < 1e-6


def test_catalogue_structure():
    cat = Catalogue.load(include_ai=True)
    assert not cat.check_references()
    g = cat.dependency_graph(True)
    sccs = [c for c in nx.strongly_connected_components(g) if len(c) > 1]
    assert sccs == [{"NC-1", "NR-1", "NO-1", "NI-1"}]
    ai = g.subgraph([n for n, d in g.nodes(data=True) if d.get("origin") == "ai"])
    assert nx.is_directed_acyclic_graph(ai)


def test_standard_profiles_valid():
    cat = Catalogue.load(include_ai=False)
    prof = json.load(open(os.path.join(ROOT, "data/processed/nd_tzi_2.5-005-99_profiles.json"), encoding="utf8"))["profiles"]
    assert len(prof) == 90
    assert all(not cat.violations(p["services"]) for p in prof)


def test_closure_least_fixed_point():
    cat = Catalogue.load(include_ai=True)
    p, g = cat.closure({"IA": 3})
    assert not cat.violations(p, g)
    assert p["NC"] >= 3 and p["CO"] >= 1 and g >= 5
    # minimality: removing any added requirement breaks validity
    for s in p:
        if s == "IA":
            continue
        q = dict(p); q[s] -= 1
        if q[s] == 0:
            del q[s]
        assert cat.violations(q, g), s


def test_parse_profile():
    assert parse_profile("{ KD-2, NR-3, NR-1 }") == {"KD": 2, "NR": 3}


def test_effective_conditions_monotone():
    # Proposition 1 needs N*(s, l) to be monotone in l; this holds under the cumulative (hierarchical) reading,
    # and the per-level reading of the printed legacy tables fails it for exactly two pairs.
    assert Catalogue.load(True).monotonicity_violations() == []
    per_level = Catalogue.load(True, legacy_cumulative=False).monotonicity_violations()
    assert sorted(v.split(":")[0] for v in per_level) == ["KK-2 -> KK-3", "NC-1 -> NC-2"]


def _random_profile(cat, rng, k):
    codes = rng.choice(sorted(cat.services), size=k, replace=False)
    return {c: int(rng.integers(1, cat.services[c].max_level + 1)) for c in codes}


def _join(p, q):
    return {s: max(p.get(s, 0), q.get(s, 0)) for s in set(p) | set(q)}


def test_closure_is_least_valid_profile():
    # leastness (not only local minimality): cl(P) is valid, dominates P, and is dominated by every valid
    # profile that dominates P; valid dominating profiles are generated as closures of random supersets.
    cat = Catalogue.load(True)
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = _random_profile(cat, rng, int(rng.integers(1, 6)))
        c, gc = cat.closure(p)
        assert not cat.violations(c, gc) and cat.dominates(p, c)
        for _ in range(5):
            q, gq = cat.closure(_join(p, _random_profile(cat, rng, int(rng.integers(1, 6)))), int(rng.integers(0, 8)))
            assert not cat.violations(q, gq)
            assert cat.dominates(c, q) and gc <= gq


def test_interior_is_greatest_valid_subprofile():
    # brute force over all sub-profiles of small random profiles
    cat = Catalogue.load(True)
    rng = np.random.default_rng(1)
    for _ in range(150):
        p = _random_profile(cat, rng, int(rng.integers(1, 5)))
        g = int(rng.integers(0, 8))
        inn = cat.interior(p, g)
        assert cat.dominates(inn, p) and not cat.violations(inn, g)
        codes = sorted(p)
        for combo in itertools.product(*[range(p[c] + 1) for c in codes]):
            q = {c: l for c, l in zip(codes, combo) if l > 0}
            if not cat.violations(q, g):
                assert cat.dominates(q, inn), (p, g, q, inn)
