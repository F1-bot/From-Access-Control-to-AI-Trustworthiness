"""Sample sizes and the half-sample exercise of the probabilistic conformity semantics (major revision).

(a) Exact sample size for power 0.8 of the LCB rule (accept 'p >= tau' iff the one-sided exact Clopper-Pearson
    lower bound clears tau) at tau = 0.5, delta in {0.05, ..., 0.30}, alpha in {0.10, 0.05, 0.01}. Binomial power
    is saw-toothed in n, so besides the FIRST n with power >= 0.8 (the value in data/results/stats_semantics.json,
    aitcert.stats.n_exact_power) the script reports the smallest n such that power >= 0.8 for ALL n' >= n
    (checked up to N_MAX, beyond which the power is >= 1 - 1e-6 and the largest possible saw-tooth drop, one binomial
    point mass, is far smaller than the margin above 0.8).
    The acceptance threshold uses the exact duality  LCB(k, n, alpha) >= tau  <=>  P_tau(X >= k) <= alpha,
    vectorised over n and cross-checked against aitcert.stats.min_successes (beta quantiles) for n <= 3000.
(b) The power-matched Hoeffding requirement n >= (sqrt(ln(1/alpha)) + sqrt(ln(1/beta)))^2 / (2 delta^2), beta = 0.2,
    next to the current consumer's-risk-only Hoeffding n = ln(1/alpha) / (2 delta^2) and the exact requirement.
(c) The exact acceptance probability at p = tau + delta of the current Hoeffding n, under the Hoeffding rule
    (accept iff p_hat - sqrt(ln(1/alpha) / (2n)) >= tau) and under the exact LCB rule.
(d) Half-sample replication of the 15 detector variants of stats_semantics.json (n = 29 drawn without replacement
    from 58): hypergeometric acceptance probabilities under the naive and LCB rules, next to the empirical
    frequencies (stored and re-computed with the generator of code/aitcert/run_stats_semantics.py) and the binomial
    OC values.

Writes results/stats_revision.json. Usage: python stats_revision.py"""
import glob
import json
import math
import os
import time

import numpy as np
from scipy import stats as st

import rev_lib as R
from aitcert_path import stats as S

SCRIPT = "stats_revision.py"
TAU, BETA = 0.5, 0.2
POWER = 1 - BETA
DELTAS = (0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30)
ALPHAS = (0.10, 0.05, 0.01)
N_MAX = 30000
N_CHECK = 3000
t0, started = time.time(), R.utc_now()
sem = json.load(open(os.path.join(R.ROOT, "data", "results", "stats_semantics.json")))


# ---------------------------------------------------------------- exact acceptance threshold, vectorised
def kmin_vec(n, tau, alpha):
    """Smallest k with P_tau(X >= k) <= alpha for X ~ Bin(n, tau); n + 1 means 'no k works'."""
    n = np.asarray(n, dtype=np.int64)
    k = (st.binom.isf(alpha, n, tau) + 1).astype(np.int64)
    for _ in range(100):
        down = (k >= 1) & (st.binom.sf(k - 2, n, tau) <= alpha)
        up = (k <= n) & (st.binom.sf(k - 1, n, tau) > alpha)
        if not down.any() and not up.any():
            break
        k = k - down.astype(np.int64) + up.astype(np.int64)
    return np.minimum(k, n + 1)


def check_kmin(n, k, tau, alpha):
    ok_hi = np.where(k <= n, st.binom.sf(k - 1, n, tau) <= alpha, st.binom.sf(n - 1, n, tau) > alpha)
    ok_lo = np.where(k >= 1, st.binom.sf(k - 2, n, tau) > alpha, True)
    with np.errstate(divide="ignore"):
        g1 = np.abs(np.log(st.binom.sf(np.minimum(k, n) - 1, n, tau) / alpha))
        g2 = np.abs(np.log(st.binom.sf(np.minimum(k, n) - 2, n, tau) / alpha))
    return bool(ok_hi.all() and ok_lo.all()), float(min(g1.min(), g2.min()))


def power_lcb(n, k, p):
    return np.where(k <= n, st.binom.sf(k - 1, n, p), 0.0)


def certificate(alpha, delta, n=N_MAX):
    """Rigorous lower bound on the power for every n' >= n. By Hoeffding at tau, k*(n') <= n' tau + t0 + 1 with
    t0 = sqrt(n' ln(1/alpha) / 2); by Hoeffding at tau + delta, power(n') >= 1 - exp(-2 h(s)^2) with s = sqrt(n'),
    h(s) = s delta - sqrt(ln(1/alpha) / 2) - 1 / s, which increases in s; so the bound at n holds for all n' >= n."""
    s = math.sqrt(n)
    h = s * delta - math.sqrt(math.log(1 / alpha) / 2) - 1 / s
    if h <= 0:
        return {"valid": False}
    return {"valid": True, "h": h, "power_lower_bound_for_all_n_ge_N_MAX": 1 - math.exp(-2 * h * h),
            "log10_of_1_minus_bound": (-2 * h * h) / math.log(10)}


ns = np.arange(1, N_MAX + 1)
kmins, checks = {}, {}
for a in ALPHAS:
    k = kmin_vec(ns, TAU, a)
    ok, gap = check_kmin(ns, k, TAU, a)
    ref = np.array([S.min_successes(int(n), TAU, a) or (int(n) + 1) for n in range(1, N_CHECK + 1)])
    agree = bool(np.array_equal(ref, k[:N_CHECK]))
    kmins[a] = k
    checks[str(a)] = {"duality_conditions_hold_all_n": ok, "min_abs_log_ratio_sf_to_alpha": gap,
                      f"agrees_with_aitcert_min_successes_n<={N_CHECK}": agree}

part_a = {}
for a in ALPHAS:
    rows = {}
    for d in DELTAS:
        pw = power_lcb(ns, kmins[a], TAU + d)
        above = pw >= POWER
        if not above.any() or not above[-1000:].all():
            raise RuntimeError(f"N_MAX too small for alpha={a}, delta={d}")
        first = int(ns[np.argmax(above)])
        below_idx = np.where(~above)[0]
        last_below = int(ns[below_idx[-1]]) if len(below_idx) else 0
        stable = last_below + 1
        dips = [int(x) for x in ns[(ns > first) & (ns < stable) & ~above]]
        tail = pw[stable - 1:]
        rows[str(d)] = {
            "first_crossing_n": first,
            "first_crossing_n_stats_semantics_json": sem["n_exact_delta"].get(str(a), {}).get(str(d)),
            "first_crossing_n_aitcert": S.n_exact_power(TAU, d, a, BETA),
            "stable_n": stable,
            "n_values_below_0.8_after_first_crossing": len(dips),
            "dips": dips if len(dips) <= 60 else {"first_10": dips[:10], "last_10": dips[-10:]},
            "power_at_first_crossing": float(pw[first - 1]), "power_at_stable_n": float(pw[stable - 1]),
            "min_power_between_first_and_stable": float(pw[first - 1:stable - 1].min()) if stable > first else None,
            "min_power_from_stable_to_N_MAX": float(tail.min()), "power_at_N_MAX": float(pw[-1]),
            "max_binomial_point_mass_at_N_MAX": float(st.binom.pmf(np.arange(N_MAX + 1), N_MAX, TAU + d).max()),
            "certificate_beyond_N_MAX": certificate(a, d)}
    part_a[str(a)] = rows


# ---------------------------------------------------------------- (b), (c) Hoeffding requirements
def hoeffding_accept_k(n, alpha):
    r = math.sqrt(math.log(1 / alpha) / (2 * n))
    return int(math.ceil(n * (TAU + r) - 1e-9))


part_bc = {}
for a in ALPHAS:
    rows = {}
    for d in DELTAS:
        n_h = S.n_hoeffding(a, d)
        n_pm_real = (math.sqrt(math.log(1 / a)) + math.sqrt(math.log(1 / BETA))) ** 2 / (2 * d * d)
        n_pm = int(math.ceil(n_pm_real - 1e-12))
        kh = hoeffding_accept_k(n_h, a)
        kl = int(kmins[a][n_h - 1])
        kh_pm = hoeffding_accept_k(n_pm, a)
        rows[str(d)] = {
            "n_hoeffding_current": n_h,
            "n_hoeffding_current_stats_semantics_json": sem["n_hoeffding_0.05"].get(str(d)) if a == 0.05 else None,
            "n_hoeffding_power_matched_real": n_pm_real, "n_hoeffding_power_matched": n_pm,
            "n_exact_first_crossing": part_a[str(a)][str(d)]["first_crossing_n"],
            "n_exact_stable": part_a[str(a)][str(d)]["stable_n"],
            "ratio_power_matched_to_exact_stable": n_pm / part_a[str(a)][str(d)]["stable_n"],
            "acceptance_at_tau_plus_delta_current_n": {
                "hoeffding_rule": {"accept_if_successes_ge": kh, "probability": float(st.binom.sf(kh - 1, n_h, TAU + d)),
                                   "consumer_risk_at_tau": float(st.binom.sf(kh - 1, n_h, TAU))},
                "lcb_rule": {"accept_if_successes_ge": kl, "probability": float(power_lcb(n_h, kl, TAU + d)),
                             "consumer_risk_at_tau": float(power_lcb(n_h, kl, TAU))}},
            "acceptance_at_tau_plus_delta_power_matched_n": {
                "hoeffding_rule": float(st.binom.sf(kh_pm - 1, n_pm, TAU + d)),
                "lcb_rule": float(power_lcb(n_pm, int(kmins[a][n_pm - 1]), TAU + d))}}
    part_bc[str(a)] = rows
reviewer_check = {"delta": 0.10, "alpha": 0.05, "claimed": "about 450",
                  "computed_real": part_bc["0.05"]["0.1"]["n_hoeffding_power_matched_real"],
                  "computed_ceiling": part_bc["0.05"]["0.1"]["n_hoeffding_power_matched"]}

# ---------------------------------------------------------------- (d) half-sample replication
N_FULL, NH, R_REP = 58, 29, 4000
k_naive = int(math.ceil(TAU * NH - 1e-12))
k_lcb = S.min_successes(NH, TAU, 0.05)
res_dir = R.RES
vecs = []
for fpath in sorted(glob.glob(os.path.join(res_dir, "e1_baseline_s*.json"))):
    dd = json.load(open(fpath))
    for q, v in dd["by_quantile"].items():
        vecs.append((dd["tag"], q, np.array(v["detected_vector"])))
rng = np.random.default_rng(2026)                     # generator and loop order of code/aitcert/run_stats_semantics.py
repro = []
for tag, q, v in vecs:
    acc_n = acc_l = 0
    for _ in range(R_REP):
        idx = rng.choice(len(v), NH, replace=False)
        kk = int(v[idx].sum())
        acc_n += (kk / NH >= TAU)
        acc_l += (S.cp_lower(kk, NH, 0.05) >= TAU)
    repro.append((acc_n / R_REP, acc_l / R_REP))
stored = {e["variant"]: e for e in sem["empirical"]}
part_d = {"N": N_FULL, "n_half": NH, "replicates": R_REP, "accept_naive_if_k_ge": k_naive, "accept_lcb_if_k_ge": k_lcb,
          "variants": []}
for (tag, q, v), (rn, rl) in zip(vecs, repro):
    K = int(v.sum())
    name = f"{tag}@q{q}"
    hg = st.hypergeom(N_FULL, K, NH)
    bn = st.binom(NH, K / N_FULL)
    row = {"variant": name, "detected_full": K, "recall_full": K / N_FULL,
           "naive": {"hypergeometric": float(hg.sf(k_naive - 1)), "binomial_oc": float(bn.sf(k_naive - 1)),
                     "empirical_stored": stored[name]["acc_naive"], "empirical_recomputed": rn},
           "lcb": {"hypergeometric": float(hg.sf(k_lcb - 1)), "binomial_oc": float(bn.sf(k_lcb - 1)),
                   "empirical_stored": stored[name]["acc_lcb"], "empirical_recomputed": rl}}
    for rule in ("naive", "lcb"):
        r_ = row[rule]
        se = math.sqrt(max(r_["hypergeometric"] * (1 - r_["hypergeometric"]), 1e-12) / R_REP)
        r_["z_empirical_vs_hypergeometric"] = (r_["empirical_stored"] - r_["hypergeometric"]) / se
        r_["empirical_minus_binomial"] = r_["empirical_stored"] - r_["binomial_oc"]
    part_d["variants"].append(row)
for rule in ("naive", "lcb"):
    part_d[f"max_abs_empirical_minus_hypergeometric_{rule}"] = max(abs(r[rule]["empirical_stored"] - r[rule]["hypergeometric"]) for r in part_d["variants"])
    part_d[f"max_abs_empirical_minus_binomial_{rule}"] = max(abs(r[rule]["empirical_stored"] - r[rule]["binomial_oc"]) for r in part_d["variants"])
    part_d[f"max_abs_z_vs_hypergeometric_{rule}"] = max(abs(r[rule]["z_empirical_vs_hypergeometric"]) for r in part_d["variants"])
    zs = np.array([r[rule]["z_empirical_vs_hypergeometric"] for r in part_d["variants"]])
    part_d[f"chi2_15df_vs_hypergeometric_{rule}"] = {"statistic": float((zs ** 2).sum()),
                                                      "p_value": float(st.chi2.sf((zs ** 2).sum(), len(zs)))}
part_d["empirical_reproduced_exactly"] = all(abs(r["naive"]["empirical_stored"] - r["naive"]["empirical_recomputed"]) < 1e-12 and
                                             abs(r["lcb"]["empirical_stored"] - r["lcb"]["empirical_recomputed"]) < 1e-12
                                             for r in part_d["variants"])
part_d["finite_population_variance_factor"] = (N_FULL - NH) / (N_FULL - 1)

out = {"script": SCRIPT, "started_utc": started, "finished_utc": R.utc_now(), "runtime_s": time.time() - t0,
       "versions": R.versions(), "settings": {"tau": TAU, "beta": BETA, "power": POWER, "deltas": DELTAS, "alphas": ALPHAS,
                                              "N_MAX": N_MAX, "n_cross_check": N_CHECK},
       "threshold_checks": checks,
       "a_exact_sample_size": part_a, "bc_hoeffding": part_bc, "reviewer_check": reviewer_check,
       "d_half_sample": part_d}
R.jdump(out, "stats_revision.json", SCRIPT)
print(json.dumps(checks))
for a in ALPHAS:
    print("alpha", a, {d: (part_a[str(a)][str(d)]["first_crossing_n"], part_a[str(a)][str(d)]["first_crossing_n_stats_semantics_json"],
                       part_a[str(a)][str(d)]["stable_n"], part_bc[str(a)][str(d)]["n_hoeffding_current"],
                       part_bc[str(a)][str(d)]["n_hoeffding_power_matched"]) for d in DELTAS})
print("reviewer check", reviewer_check)
print("runtime_s", round(time.time() - t0, 1))
