"""Checks of the sensitivity helpers (sens_lib.py) used by run_sensitivity.py and make_second_coder.py.
Run: cd code/aitcert && python -m pytest tests/test_sensitivity.py -q"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sens_lib as L  # noqa: E402


def test_shapley_closed_form_matches_enumeration():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.choice([0.0, 0.25, 0.5, 1.0], size=int(rng.integers(1, 9)))
        phi = L.shapley_max_rows(v[None, :])[0]
        assert np.allclose(phi, L.shapley_bruteforce(v), atol=1e-12)
        assert abs(phi.sum() - v.max()) < 1e-12                      # efficiency


def test_shapley_null_players_do_not_change_values():
    v = np.array([0.5, 1.0, 0.5])
    w = np.concatenate([v, np.zeros(5)])
    assert np.allclose(L.shapley_max_rows(v[None, :])[0], L.shapley_max_rows(w[None, :])[0][:3])


def test_agreement_statistics_reference_values():
    nan = np.nan
    k11 = np.array([[1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan], [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
                    [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan], [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan]])
    for lv, e in {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797}.items():
        assert abs(L.krippendorff_alpha(k11, lv) - e) < 5e-4
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, 200)
    b = np.where(rng.random(200) < 0.6, a, rng.integers(0, 3, 200))
    for w in (None, "linear", "quadratic"):
        assert abs(L.weighted_kappa(a, b, 3, w) - cohen_kappa_score(a, b, weights=w)) < 1e-12


def test_reassign_changes_exactly_k_grades_with_allowed_transitions():
    rng = np.random.default_rng(0)
    codes = rng.choice([0, 0, 0, 1, 2], size=(40, 30)).astype(np.int8)
    n = int((codes > 0).sum())
    new, k = L.reassign(codes, np.random.default_rng(5), 0.1)
    assert k == round(0.1 * n)
    ch = new != codes
    assert int(ch.sum()) == k
    assert set(zip(codes[ch].tolist(), new[ch].tolist())) <= {(2, 1), (2, 0), (1, 2), (1, 0)}
    assert (codes[codes == 0] == new[codes == 0]).all()


def test_add_random_only_on_allowed_empty_cells():
    codes = np.zeros((10, 6), dtype=np.int8)
    codes[0, 0] = 2
    rows = np.array([True] * 5 + [False] * 5)
    cols = np.array([True, True, True, False, False, False])
    new = L.add_random(codes, np.random.default_rng(0), 8, rows, cols)
    ch = new != codes
    assert int(ch.sum()) == 8 and set(new[ch].tolist()) <= {1, 2}
    assert not ch[~rows].any() and not ch[:, ~cols].any() and new[0, 0] == 2


def test_best_code_equals_max_weight():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 3, size=(50, 12))
    mask = rng.random(12) < 0.5
    for w in (0.25, 0.5, 0.75):
        for strict in (False, True):
            lut = L.lut(w, strict)
            assert np.allclose(lut[L.best_codes(codes, mask)], lut[codes][:, mask].max(axis=1))


def test_level_specific_targets():
    E = L.Elements()
    targets = ["IR", "IR-2", "KD"]
    assert L.xmask(targets, ["IR", "KD"], cat=E.cat).tolist() == [True, True, True]        # catalogue level
    assert L.xmask(targets, ["KD"], cat=E.cat).tolist() == [False, False, True]
    assert L.xmask(targets, ["IR", "KD"], levels={"IR": 1, "KD": 2}).tolist() == [True, False, True]
    assert L.xmask(targets, ["IR", "KD"], levels={"IR": 3, "KD": 2}).tolist() == [True, True, True]


def test_reproduces_existing_coverage_results():
    p_a, p_c = L.PATHS["atlas_coverage"], L.PATHS["crosswalk_coverage"]
    if not (os.path.exists(p_a) and os.path.exists(p_c)):
        return
    E = L.Elements()
    path, _ = L.atlas_yaml_path()
    if not os.path.exists(path):
        import pytest
        pytest.skip("MITRE ATLAS data not available (see data/raw/README.md)")
    grades, _ = L.load_atlas_grades()
    fw, w, high, _ = L.load_crosswalk()
    targets = E.X_ext + L.level_targets([grades] + list(fw.values()))
    am, cm = L.AtlasModel(L.Atlas(path), grades, targets), L.CrosswalkModel(fw, w, high, targets)
    old_a, old_c = json.load(open(p_a, encoding="utf8")), json.load(open(p_c, encoding="utf8"))
    for lab, X in (("legacy", E.X_leg), ("extended", E.X_ext)):
        s = am.summary(L.xmask(targets, X, cat=E.cat))
        for k in ("addressed", "full", "weighted"):
            assert abs(s["mitigated"][k] - old_a["mitigated"][lab][k]) < 1e-12
        c = cm.summary(L.xmask(targets, X, cat=E.cat))
        for n in L.FW_ORDER:
            assert abs(c[n]["w0_5"] - old_c[n][lab]["w0.5"]) < 1e-12
            assert abs(c[n]["strict"] - old_c[n][lab]["strict"]) < 1e-12
