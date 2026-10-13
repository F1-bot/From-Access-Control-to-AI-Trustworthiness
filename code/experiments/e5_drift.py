"""E5: operating-envelope monitoring (IV-3) - drift between HAI 22.04 and HAI 23.05 (same testbed
configuration, later collection campaign). Measures (i) the validity loss of the 22.04 detector on
23.05 normal data (false-alarm rate) and (ii) the detection performance of a PSI-based drift monitor
on daily (pre-registered) and hourly (exploratory) segments. Negatives: normal portions of the
22.04 test files; positives: 23.05 normal training files.
Drift alarm rule (fixed a priori): median per-channel PSI against the 22.04 validation reference > 0.25."""
import argparse, json, os, time
import numpy as np, pandas as pd
import hai_lib as H
from aitcert_path import stats as S

ap = argparse.ArgumentParser(); ap.add_argument("--victim", default="baseline_s0"); args = ap.parse_args()
t0 = time.time()
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json")); F = len(prep.cols)
model = H.load_model(args.victim, F)
cv = np.load(os.path.join(H.RES, "models", f"{args.victim}_chanvar.npy"))
theta = json.load(open(os.path.join(H.RES, f"e1_{args.victim}.json")))["theta"]
tr6 = prep.to_array(H.read_files("hai-22.04", ["train6.csv"])[0]); ref = tr6[int(len(tr6) * 0.8):]
edges = [np.unique(np.quantile(ref[:, j], np.linspace(0, 1, 11))) for j in range(F)]
def psi(seg):
    vals = []
    for j in range(F):
        if prep.cols[j] in MISSING:
            continue
        e = edges[j]
        if len(e) < 3:
            continue
        r = np.histogram(np.clip(ref[:, j], e[0], e[-1]), e)[0] / len(ref)
        q = np.histogram(np.clip(seg[:, j], e[0], e[-1]), e)[0] / len(seg)
        r, q = np.clip(r, 1e-4, None), np.clip(q, 1e-4, None)
        vals.append(float(((q - r) * np.log(q / r)).sum()))
    return float(np.median(vals))
def segments(arr, mask, length):
    out = []
    for s in range(0, len(arr) - length + 1, length):
        m = mask[s:s + length]
        if m.mean() > 0.9:
            out.append(arr[s:s + length][m])
    return out
MISSING = set()
neg_arr, neg_mask = [], []
for i in range(1, 5):
    df = H.read_files("hai-22.04", [f"test{i}.csv"])[0]
    y = H.labels(df); m = np.ones(len(y), dtype=bool)
    for s, e in H.events_from_labels(y):
        m[max(0, s - 300):e + 600] = False   # exclude attacks and recovery
    neg_arr.append(prep.to_array(df)); neg_mask.append(m)
names23 = [f"hai-train{i}.csv" for i in range(1, 5)]
pos_df = H.read_files("hai-23.05", names23)
missing = [c for c in prep.cols if c not in pos_df[0].columns]
shared = np.array([c not in missing for c in prep.cols]); MISSING.update(missing)
def impute(df):  # channels renamed in 23.05 are unavailable -> reference mean (0 after standardisation)
    df = df.copy()
    for j, c in enumerate(prep.cols):
        if c not in df.columns:
            df[c] = prep.mu[j]
    return df
pos_arr = [prep.to_array(impute(df)) for df in pos_df]
# (i) validity under drift with a consistent configuration: missing inputs imputed with the reference
#     mean, score computed over shared channels only, threshold recalibrated on 22.04 validation alike
def shared_score(arr):
    a = arr.copy(); a[:, ~shared] = 0.0
    r = H.residuals(model, a)[:, shared]
    return H.smooth_fast(H.score_from_residuals(r, cv[shared]))
theta_sh = float(np.nanquantile(shared_score(ref), 0.999))
fa, hours = 0, 0.0
for arr in pos_arr:
    fa += len(H.alarm_episodes(shared_score(arr), theta_sh)); hours += len(arr) / 3600.0
fa_n, h_n = 0, 0.0
for a, m in zip(neg_arr, neg_mask):
    sc = shared_score(a); sc[~m] = np.nan
    fa_n += len(H.alarm_episodes(sc, theta_sh)); h_n += m.sum() / 3600.0
out = {"missing_columns_in_2305": missing, "n_shared": int(shared.sum()), "theta_shared": theta_sh,
       "far_2305_per_h": fa / hours, "alarms_2305": fa, "hours_2305": hours,
       "far_2204_test_normal_per_h": fa_n / h_n, "alarms_2204": fa_n, "hours_2204": h_n,
       "far_2305_ucb95": S.poisson_upper(fa, hours, 0.05), "far_2204_ucb95": S.poisson_upper(fa_n, h_n, 0.05)}
# (ii) drift monitor
for name, L in (("daily", 86400), ("hourly", 3600)):
    neg = [psi(s) for a, m in zip(neg_arr, neg_mask) for s in segments(a, m, L)]
    pos = [psi(s) for a in pos_arr for s in segments(a, np.ones(len(a), dtype=bool), L)]
    tp = int(sum(p > 0.25 for p in pos)); fp = int(sum(n > 0.25 for n in neg))
    out[name] = {"n_pos": len(pos), "n_neg": len(neg), "tp": tp, "fp": fp,
                 "tpr": tp / len(pos) if pos else None, "fpr": fp / len(neg) if neg else None,
                 "tpr_lcb95": S.cp_lower(tp, len(pos), 0.05) if pos else None,
                 "fpr_ucb95": S.cp_upper(fp, len(neg), 0.05) if neg else None,
                 "psi_pos_median": float(np.median(pos)) if pos else None, "psi_neg_median": float(np.median(neg)) if neg else None,
                 "psi_pos": pos, "psi_neg": neg}
    print(name, out[name]["n_pos"], out[name]["n_neg"], out[name]["tpr"], out[name]["fpr"])
# (iii) exploratory: model-based residual monitor (hourly median of the shared-channel score);
#       threshold = maximum of sliding hourly medians on the 22.04 validation reference
def hourly_medians(sc, mask=None, stride=3600):
    vals = []
    for st in range(0, len(sc) - 3600 + 1, stride):
        seg = sc[st:st + 3600]
        if mask is not None and mask[st:st + 3600].mean() <= 0.9:
            continue
        if mask is not None:
            seg = seg[mask[st:st + 3600]]
        vals.append(float(np.nanmedian(seg)))
    return vals
ref_sc = shared_score(ref)
thr_med = float(np.max(hourly_medians(ref_sc, stride=300)))
negm = [v for a, m in zip(neg_arr, neg_mask) for v in hourly_medians(shared_score(a), m)]
posm = [v for a in pos_arr for v in hourly_medians(shared_score(a))]
tp = int(sum(v > thr_med for v in posm)); fp = int(sum(v > thr_med for v in negm))
out["residual_monitor_hourly"] = {"threshold": thr_med, "n_pos": len(posm), "n_neg": len(negm), "tp": tp, "fp": fp,
    "tpr": tp / len(posm), "fpr": fp / len(negm), "tpr_lcb95": S.cp_lower(tp, len(posm), 0.05),
    "fpr_ucb95": S.cp_upper(fp, len(negm), 0.05), "exploratory": True, "pos_values": posm, "neg_values": negm,
    "ref_values": hourly_medians(ref_sc, stride=300)}
print("residual monitor", out["residual_monitor_hourly"]["tpr"], out["residual_monitor_hourly"]["fpr"])
out["runtime_s"] = time.time() - t0
H.jdump(out, "e5_drift.json")
print("FAR on 23.05:", out["far_2305_per_h"])
