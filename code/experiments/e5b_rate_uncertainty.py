"""E5b supplement: overdispersion-robust uncertainty of the false-alarm rates of the retrained 60-channel
detector (e5b_shared60_s0) on HAI 22.04 test normal periods and HAI 23.05 normal data, and of their ratio.

The exact Poisson (Garwood) intervals reported in e5b_drift_retrained.json assume independent alarms.
Hourly counts are tested for overdispersion (Pearson X2/df with exposure weights), and two robust
intervals are added: quasi-Poisson (Garwood bound on the effective count N/phi over T/phi) and a circular
moving-block bootstrap over hourly bins (blocks of 1, 6 and 24 h; B = 10000; seed 0). The ratio
23.05/22.04 gets the exact conditional interval, a quasi-Poisson log-normal interval and the
independent block-bootstrap percentile interval. Conventions: E1 (hai_lib.evaluate_file) and E5
(exclusion of [start - 300 s, end + 600 s) around attacks). Writes results/e5b_rate_uncertainty.json."""
import json, os, time
import numpy as np
from scipy import stats as st
import hai_lib as H
import revml_lib as R
from aitcert_path import stats as S

t0 = time.time()
A = 0.05
B = 10000
TAG = "e5b_shared60_s0"
e5b = json.load(open(os.path.join(H.RES, "e5b_drift_retrained.json")))
theta = e5b["training"]["theta"]
prep = H.Prep.load(os.path.join(H.RES, "models", "e5b_prep_shared60.json"))
cv = np.load(os.path.join(H.RES, "models", f"{TAG}_chanvar.npy"))
z = np.load(os.path.join(H.RES, "models", f"{TAG}_scores.npz"))
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])

def qp_interval(N, T, phi, alpha=A):
    phi = max(1.0, phi); n, t = N / phi, T / phi
    lo = 0.0 if N == 0 else float(st.chi2.ppf(alpha / 2, 2 * n) / 2) / t
    return [lo, float(st.chi2.ppf(1 - alpha / 2, 2 * (n + 1)) / 2) / t]

def boot_rates(c, t, block, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        k = S.block_bootstrap_indices(len(c), block, rng); out[b] = c[k].sum() / t[k].sum()
    return out

def summarise(c, t, label):
    c = np.asarray(c, float); t = np.asarray(t, float); keep = t > 0; c, t = c[keep], t[keep]
    N, T = int(c.sum()), float(t.sum())
    disp = R.dispersion_test(c, t, B=20000, seed=0)
    phi = disp["dispersion_x2_over_df"]
    res = {"label": label, **R.rate_summary(N, T, A), "dispersion": disp, "phi": phi,
           "quasi_poisson_ci95": qp_interval(N, T, phi), "block_bootstrap_ci95": {}}
    boots = {}
    for blk in (1, 6, 24):
        v = boot_rates(c, t, blk, seed=0); boots[blk] = v
        res["block_bootstrap_ci95"][str(blk)] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return res, boots, (N, T, phi)

out = {"model": TAG, "theta": theta, "B": B, "seed": 0}
# ---- 22.04 test normal periods, both conventions (stored scores of the retrained model)
c1, t1, c5, t5 = [], [], [], []
for i, df in enumerate(test, start=1):
    y = H.labels(df); sc = z[f"test{i}"].astype(np.float64)
    fa = R.e1_false_alarm_episodes(sc, y, theta)
    c, t = R.hourly_bins(len(y), R.e1_normal_mask(y), [a for a, _ in fa]); c1 += c.tolist(); t1 += t.tolist()
    m5 = R.e5_normal_mask(y); s5 = sc.copy(); s5[~m5] = np.nan
    eps5 = H.alarm_episodes(s5, theta)
    c, t = R.hourly_bins(len(y), m5, [a for a, _ in eps5]); c5 += c.tolist(); t5 += t.tolist()
r22e1, b22e1, k22e1 = summarise(c1, t1, "22.04 test normal, E1 convention")
r22e5, b22e5, k22e5 = summarise(c5, t5, "22.04 test normal, E5 convention")
assert r22e1["count"] == e5b["test_2204_e1_convention"]["far"]["count"] and r22e5["count"] == e5b["test_2204_e5_convention"]["count"]
# ---- 23.05 normal training files (recomputed with the saved model)
model = H.load_model(TAG, len(prep.cols))
c23, t23 = [], []
for n in [f"hai-train{i}.csv" for i in range(1, 5)]:
    df = H.read_files("hai-23.05", [n])[0]
    sc = H.smooth_fast(H.score_from_residuals(H.residuals(model, prep.to_array(df)), cv))
    eps = H.alarm_episodes(sc, theta)
    m = np.ones(len(sc), bool); m[:H.W] = False                                   # E1 convention exposure
    c, t = R.hourly_bins(len(sc), m, [a for a, _ in eps]); c23 += c.tolist(); t23 += t.tolist()
r23, b23, k23 = summarise(c23, t23, "23.05 normal training files, E1 convention")
assert r23["count"] == e5b["normal_2305_train_files"]["e1_convention"]["count"]
out["rates"] = {"2204_e1": r22e1, "2204_e5": r22e5, "2305_e1": r23}
def ratio_block(num, den, kn, kd):
    (N1, T1, p1), (N2, T2, p2) = kn, kd
    est = (N1 / T1) / (N2 / T2)
    se = np.sqrt(max(1, p1) / N1 + max(1, p2) / N2)
    z_ = st.norm.ppf(1 - A / 2)
    res = {"ratio": est, "exact_conditional_poisson_ci95": R.rate_ratio_exact(N1, T1, N2, T2, A)["ci95_two_sided"],
           "quasi_poisson_lognormal_ci95": [float(est * np.exp(-z_ * se)), float(est * np.exp(z_ * se))], "block_bootstrap_ci95": {}}
    for blk in (1, 6, 24):
        v = num[blk] / den[blk]
        res["block_bootstrap_ci95"][str(blk)] = [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return res
out["ratio_2305_over_2204"] = {"e1_convention": ratio_block(b23, b22e1, k23, k22e1), "vs_2204_e5_convention": ratio_block(b23, b22e5, k23, k22e5)}
for k, v in out["rates"].items():
    print(k, v["count"], round(v["exposure_h"], 3), round(v["rate_per_h"], 4), "phi", round(v["phi"], 3), "p", v["dispersion"]["p_chi2_overdispersion"],
          "garwood", np.round(v["ci95_two_sided"], 3).tolist(), "qp", np.round(v["quasi_poisson_ci95"], 3).tolist(),
          "boot", {b: np.round(x, 3).tolist() for b, x in v["block_bootstrap_ci95"].items()})
for k, v in out["ratio_2305_over_2204"].items():
    print("ratio", k, round(v["ratio"], 2), np.round(v["exact_conditional_poisson_ci95"], 2).tolist(), np.round(v["quasi_poisson_lognormal_ci95"], 2).tolist(),
          {b: np.round(x, 2).tolist() for b, x in v["block_bootstrap_ci95"].items()})
out["meta"] = R.run_meta("e5b_rate_uncertainty.py", t0, {"bootstrap": 0, "dispersion_mc": 0})
p = R.save_json(out, "e5b_rate_uncertainty.json")
print("wrote", p, "runtime", round(time.time() - t0, 1))
