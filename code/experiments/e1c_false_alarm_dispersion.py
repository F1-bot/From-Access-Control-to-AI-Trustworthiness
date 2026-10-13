"""E1c (major revision): dispersion and foreseeability of false alarms.

(a) Hourly false-alarm counts over the normal periods of the HAI 22.04 test files under the E1
    conventions (hai_lib.evaluate_file: alarm episodes merged within 30 s; an episode is a false alarm if
    it overlaps no [attack start, attack end + 60 s]; normal time excludes the first W s and those
    intervals). Bins are clock hours from the start of each test file; each false alarm is assigned to
    the bin of its first second; the exposure of a bin is its normal time. Reported: the dispersion
    index (variance/mean of counts in full hours; Pearson X2/df with exposure weights over all bins),
    the chi-square test of the homogeneous Poisson model (index-of-dispersion test) and a conditional
    Monte Carlo p-value, and the sensitivity of the exact Poisson upper bound of the FAR to the
    observed overdispersion (quasi-Poisson scaling and moving-block bootstrap over hours).
(b) Foreseeability: alarm episodes on the 14.4-h validation segment (last 20% of train6) at the
    pre-registered threshold (99.9th percentile of the smoothed validation score) for the five baseline
    seeds (and the hardened variant), recomputed from the saved models and cross-checked against the
    stored scores; exact Poisson bounds; comparison with the test false-alarm rate.
Writes results/e1c_false_alarm_dispersion.json."""
import json, os, time
import numpy as np
from scipy import stats as st
import hai_lib as H
import revml_lib as R
from aitcert_path import stats as S

t0 = time.time()
A = 0.05
TAGS_A = ["baseline_s0", "hardened_s0"]
TAGS_ALL = ["baseline_s0", "baseline_s1", "baseline_s2", "baseline_s3", "baseline_s4", "hardened_s0"]
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
ys = [H.labels(df) for df in test]
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
tr6 = prep.to_array(H.read_files("hai-22.04", ["train6.csv"])[0])
val = tr6[int(len(tr6) * 0.8):]
out = {"conventions": {"bins": "clock hours from the start of each test file (last bin of test3 is 1200 s)",
                       "false_alarm": "hai_lib.evaluate_file definition (E1)", "full_hour_min_exposure_h": 0.95,
                       "validation_seconds": int(len(val))},
       "a_dispersion": {}, "b_validation": {}}

for tag in TAGS_ALL:
    e1 = json.load(open(os.path.join(H.RES, f"e1_{tag}.json")))
    theta = e1["theta"]
    z = np.load(os.path.join(H.RES, "models", f"{tag}_scores.npz"))
    counts, expo, starts_all, fa_check, nh_check, gaps = [], [], [], 0, 0.0, []
    file_of_bin = []
    for i, y in enumerate(ys, start=1):
        sc = z[f"test{i}"].astype(np.float64)
        ev = H.evaluate_file(sc, y, theta)
        fa = R.e1_false_alarm_episodes(sc, y, theta)
        assert len(fa) == ev["false_alarms"]
        m = R.e1_normal_mask(y)
        assert all(m[a0:a1 + 1].all() for a0, a1 in fa), "false alarm outside the E1 normal time"
        c, t = R.hourly_bins(len(y), m, [a0 for a0, _ in fa])
        counts += c.tolist(); expo += t.tolist(); file_of_bin += [i] * len(c)
        fa_check += len(fa); nh_check += m.sum() / 3600.0
        st_ = [a0 for a0, _ in fa]
        gaps += list(np.diff(st_))
    agg = e1["aggregate"]
    counts = np.array(counts); expo = np.array(expo)
    rec = {"theta": theta, "false_alarms": int(counts.sum()), "normal_hours_bins": float(expo.sum()),
           "check_vs_e1": {"false_alarms_e1": agg["false_alarms"], "normal_hours_e1": agg["normal_hours"],
                           "false_alarms_match": int(counts.sum()) == agg["false_alarms"],
                           "normal_hours_match": bool(abs(expo.sum() - agg["normal_hours"]) < 1e-9)},
           "hourly_counts": counts.tolist(), "hourly_exposure_h": expo.round(6).tolist(), "file_of_bin": file_of_bin}
    full = expo >= out["conventions"]["full_hour_min_exposure_h"]
    cf = counts[full]
    rec["full_hours"] = {"n_hours": int(full.sum()), "false_alarms": int(cf.sum()), "mean": float(cf.mean()),
                         "variance": float(cf.var(ddof=1)), "dispersion_index_var_over_mean": float(cf.var(ddof=1) / cf.mean()),
                         "count_histogram": {str(k): int((cf == k).sum()) for k in range(int(cf.max()) + 1)},
                         "poisson_expected_histogram": {str(k): float(full.sum() * st.poisson.pmf(k, cf.mean())) for k in range(int(cf.max()) + 1)},
                         "test": R.dispersion_test(cf, expo[full], B=100_000, seed=0)}
    rec["all_bins"] = R.dispersion_test(counts, expo, B=100_000, seed=0)
    phi = rec["all_bins"]["dispersion_x2_over_df"]
    N, T = int(counts.sum()), float(expo.sum())
    pos = expo > 0
    rec["far_bound_sensitivity"] = {
        "rate_per_h": N / T, "garwood_ucb95": S.poisson_upper(N, T, A),
        "quasi_poisson_ucb95": R.quasi_poisson_ucb(N, T, phi, A), "phi_used": max(1.0, phi),
        "block_bootstrap_ucb95": {str(b): R.block_bootstrap_rate_ucb(counts[pos], expo[pos], b, B=10000, seed=0, alpha=A) for b in (1, 3, 6)},
        "tau": 0.25}
    g = np.array(gaps, dtype=float)
    rec["clustering"] = {"n_gaps": int(len(g)), "median_gap_s": float(np.median(g)) if len(g) else None,
                         "share_gaps_le_600s": float((g <= 600).mean()) if len(g) else None,
                         "share_gaps_le_3600s": float((g <= 3600).mean()) if len(g) else None,
                         "share_of_false_alarms_in_top10pct_hours": float(np.sort(counts)[::-1][:int(np.ceil(0.1 * pos.sum()))].sum() / max(N, 1)),
                         "hours_with_ge3_false_alarms": int((counts >= 3).sum()),
                         "expected_gap_share_le_600s_if_poisson": float(1 - np.exp(-N / T * 600 / 3600))}
    out["a_dispersion"][tag] = rec
    print(tag, "FA", N, "h", round(T, 3), "full-hour var/mean", round(rec["full_hours"]["dispersion_index_var_over_mean"], 3),
          "p", rec["full_hours"]["test"]["p_chi2_overdispersion"], "| all bins X2/df", round(phi, 3), "p", rec["all_bins"]["p_chi2_overdispersion"],
          "p_mc", rec["all_bins"]["p_monte_carlo_conditional"], flush=True)

# ---------------------------------------------------------------- (b) validation episodes at the pre-registered threshold
val_h = len(val) / 3600.0
val_valid_h = (len(val) - H.W) / 3600.0
for tag in TAGS_ALL:
    e1 = json.load(open(os.path.join(H.RES, f"e1_{tag}.json")))
    theta = e1["theta"]
    model = H.load_model(tag, len(prep.cols))
    cv_saved = np.load(os.path.join(H.RES, "models", f"{tag}_chanvar.npy"))
    res_val = H.residuals(model, val)
    cv = np.nanmean(res_val, axis=0) + 1e-6
    s_val = H.smooth_fast(H.score_from_residuals(res_val, cv))
    th_re = float(np.nanquantile(s_val, 0.999))
    eps = H.alarm_episodes(s_val, theta)
    stored = np.load(os.path.join(H.RES, "models", f"{tag}_scores.npz"))["val"].astype(np.float64)
    eps_stored = H.alarm_episodes(stored, theta)
    k = len(eps)
    agg = e1["aggregate"]
    out["b_validation"][tag] = {
        "theta_e1": theta, "theta_recomputed": th_re, "theta_rel_diff": abs(th_re - theta) / theta,
        "chanvar_max_rel_diff": float(np.max(np.abs(cv - cv_saved) / cv_saved)),
        "seconds_above_theta": int((np.nan_to_num(s_val, nan=-np.inf) > theta).sum()),
        "episodes": [[int(a), int(b), int(b - a + 1)] for a, b in eps], "n_episodes": k,
        "n_episodes_from_stored_scores": len(eps_stored),
        "alarms_per_h_14_4h": k / val_h, "alarms_per_h_valid_seconds": k / val_valid_h,
        "rate_bounds_14_4h": R.rate_summary(k, val_h, A),
        "test_false_alarms": agg["false_alarms"], "test_normal_hours": agg["normal_hours"], "test_far_per_h": agg["far_per_h"],
        "test_vs_validation_rate_ratio": R.rate_ratio_exact(agg["false_alarms"], agg["normal_hours"], k, val_h, A),
        "max_validation_episodes_compatible_with_far_ucb_le_0.25": int(max([c for c in range(0, 20) if S.poisson_upper(c, val_h, A) <= 0.25], default=-1))}
    print(tag, "validation episodes", k, "(stored:", len(eps_stored), ") per h", round(k / val_h, 4), "theta", theta, th_re, flush=True)
    del model
base = [out["b_validation"][f"baseline_s{s}"]["n_episodes"] for s in range(5)]
out["b_validation"]["summary_baseline_seeds"] = {"episodes": base, "per_h_14_4h": [b / val_h for b in base],
                                                 "pooled_rate_per_h": sum(base) / (5 * val_h),
                                                 "reviewer_claim": "4 episodes, 0.28 per hour", "claim_confirmed": all(b == 4 for b in base)}
out["meta"] = R.run_meta("e1c_false_alarm_dispersion.py", t0, {"monte_carlo": 0, "block_bootstrap": 0},
                         {"inputs": "results/models/*_scores.npz (stored E1 scores), e1_*.json thresholds, saved models for the validation recomputation"})
p = R.save_json(out, "e1c_false_alarm_dispersion.json")
print("wrote", p, "runtime", round(time.time() - t0, 1))
