"""E5b supplement: can the eight P1 tags of HAI 22.04 that are absent from HAI 23.05 be mapped to the eight
new tag names of 23.05? Evidence only; the mapping is NOT used for the main E5b result.

Evidence (training files of both campaigns, all normal operation):
  1. univariate statistics in raw units (range, quantiles, distinct values, change frequency);
  2. tracking partner: the shared channel with the highest absolute correlation and the median absolute
     difference to it (a setpoint should track its controlled process variable in the same unit);
  3. correlation fingerprint: correlations of each tag with the shared channels (10-s subsample); the
     fingerprint similarity of a 22.04 tag and a 23.05 tag is the Pearson correlation of their
     fingerprints over channels that vary in both campaigns;
  4. distribution distance: 1-Wasserstein distance between the 22.04 and 23.05 values, divided by the
     22.04 standard deviation, and overlap of the 1-99% ranges.
Pre-specified acceptance rule for a pair: mutual best match on fingerprint similarity AND overlapping
1-99% ranges. Optional (--sensitivity): baseline_s0 (68 inputs, unchanged threshold) evaluated on 23.05
with accepted pairs renamed and any unmatched input imputed with its 22.04 training mean, under the E5
convention. Writes results/e5b_tag_mapping.json."""
import argparse, json, os, time
import numpy as np, pandas as pd
from scipy import stats as st
from scipy.optimize import linear_sum_assignment
import hai_lib as H
import revml_lib as R

ap = argparse.ArgumentParser(); ap.add_argument("--sensitivity", action="store_true"); args = ap.parse_args()
t0 = time.time()
OLD = ["P1_B2004", "P1_B2016", "P1_B3004", "P1_B3005", "P1_B4002", "P1_B4005", "P1_B400B", "P1_B4022"]
DOC = {"P1_B2004": "0-10 bar, heat-exchanger outlet pressure setpoint", "P1_B2016": "0-10 bar, pressure demand for thermal power output control",
       "P1_B3004": "0-720 mm, water level setpoint (return water tank)", "P1_B3005": "0-2500 l/h, discharge flowrate setpoint (return water tank)",
       "P1_B4002": "0-100 degC, heat-exchanger outlet temperature setpoint", "P1_B4005": "0-100 %, temperature PID control output",
       "P1_B400B": "0-2500 l/h, water outflow rate setpoint (heating water tank)", "P1_B4022": "0-40 degC, temperature demand for thermal power output control"}
d22 = H.read_files("hai-22.04", [f"train{i}.csv" for i in range(1, 7)])
d23 = H.read_files("hai-23.05", [f"hai-train{i}.csv" for i in range(1, 5)])
c22, c23 = list(d22[0].columns), list(d23[0].columns)
NEW = [c for c in c23 if c not in c22 and c != "timestamp"]
SHARED = [c for c in c22 if c in c23 and c != "timestamp"]
cat22 = pd.concat([d[OLD + SHARED] for d in d22], ignore_index=True).astype(np.float64)
cat23 = pd.concat([d[NEW + SHARED] for d in d23], ignore_index=True).astype(np.float64)
del d22, d23

def describe(x: pd.Series):
    v = x.to_numpy()
    q = np.quantile(v, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    vals, cnt = np.unique(v, return_counts=True)
    ch = np.count_nonzero(np.diff(v))
    dv = np.abs(np.diff(v)); dv = dv[dv > 0]
    return {"min": q[0], "q01": q[1], "q05": q[2], "q25": q[3], "median": q[4], "q75": q[5], "q95": q[6], "q99": q[7], "max": q[8],
            "mean": float(v.mean()), "sd": float(v.std()), "n_distinct": int(len(vals)), "modal_value": float(vals[cnt.argmax()]),
            "modal_share": float(cnt.max() / len(v)), "share_integer_valued": float(np.mean(np.isclose(v, np.round(v)))),
            "changes_per_hour": float(ch / (len(v) / 3600.0)), "median_nonzero_step": float(np.median(dv)) if len(dv) else 0.0}

stats22 = {c: describe(cat22[c]) for c in OLD}
stats23 = {c: describe(cat23[c]) for c in NEW}
s22 = cat22.iloc[::10]; s23 = cat23.iloc[::10]
var_sh = [c for c in SHARED if s22[c].std() > 1e-6 and s23[c].std() > 1e-6]
def fingerprint(df, c):
    x = df[c].to_numpy()
    if x.std() < 1e-12:
        return np.full(len(var_sh), np.nan)
    return np.array([np.corrcoef(x, df[s].to_numpy())[0, 1] for s in var_sh])
fp22 = {c: fingerprint(s22, c) for c in OLD}; fp23 = {c: fingerprint(s23, c) for c in NEW}
F = np.full((len(OLD), len(NEW)), np.nan)
for i, a in enumerate(OLD):
    for j, b in enumerate(NEW):
        ok = np.isfinite(fp22[a]) & np.isfinite(fp23[b])
        if ok.sum() > 5:
            F[i, j] = np.corrcoef(fp22[a][ok], fp23[b][ok])[0, 1]
Wd = np.full((len(OLD), len(NEW)), np.nan); OV = np.zeros((len(OLD), len(NEW)), dtype=bool)
for i, a in enumerate(OLD):
    sd = max(stats22[a]["sd"], 1e-9)
    for j, b in enumerate(NEW):
        Wd[i, j] = st.wasserstein_distance(s22[a].to_numpy(), s23[b].to_numpy()) / sd
        OV[i, j] = not (stats22[a]["q99"] < stats23[b]["q01"] or stats23[b]["q99"] < stats22[a]["q01"])
def partner(df, c, fp):
    if not np.isfinite(fp).any():
        return None
    k = int(np.nanargmax(np.abs(fp))); p = var_sh[k]
    return {"channel": p, "corr": float(fp[k]), "median_abs_diff_raw": float(np.median(np.abs(df[c] - df[p]))),
            "partner_sd": float(df[p].std())}
part22 = {c: partner(s22, c, fp22[c]) for c in OLD}; part23 = {c: partner(s23, c, fp23[c]) for c in NEW}
Fz = np.nan_to_num(F, nan=-2.0)
rows, cols = linear_sum_assignment(-Fz)
pairs = []
for i, j in zip(rows, cols):
    a, b = OLD[i], NEW[j]
    mutual = bool(np.argmax(Fz[i]) == j and np.argmax(Fz[:, j]) == i)
    srt = np.sort(Fz[i])[::-1]
    pairs.append({"old_2204": a, "new_2305": b, "fingerprint_similarity": None if np.isnan(F[i, j]) else float(F[i, j]),
                  "margin_to_second_best": float(srt[0] - srt[1]), "mutual_best": mutual, "ranges_overlap": bool(OV[i, j]),
                  "wasserstein_over_sd2204": float(Wd[i, j]), "same_tracking_partner": bool(part22[a] and part23[b] and part22[a]["channel"] == part23[b]["channel"]),
                  "accepted": bool(mutual and OV[i, j])})
out = {"old_tags_2204": OLD, "new_tags_2305": NEW, "documentation_2204": DOC, "n_shared_varying_channels": len(var_sh),
       "stats_2204": stats22, "stats_2305": stats23, "tracking_partner_2204": part22, "tracking_partner_2305": part23,
       "fingerprint_similarity": {a: {b: (None if np.isnan(F[i, j]) else float(F[i, j])) for j, b in enumerate(NEW)} for i, a in enumerate(OLD)},
       "wasserstein_over_sd2204": {a: {b: float(Wd[i, j]) for j, b in enumerate(NEW)} for i, a in enumerate(OLD)},
       "automatic_rule_fingerprint": {"rule": "Hungarian assignment on fingerprint similarity; accept mutual best match with overlapping 1-99% ranges (fixed before inspecting the results)",
                                      "pairs": pairs,
                                      "verdict": "fails for the temperature group: in 22.04 P1_B4005, P1_B400B and P1_FT02Z are almost collinear (r = 0.998), so their fingerprints are indistinguishable and the rule pairs P1_B4005 (%, 0-100) with x1003_10_SETPOINT_OUT (l/h, 6-3190)"}}
# ---- post-hoc evidence classification (defined after the automatic rule failed; reported as such)
x = cat22["P1_B4005"].to_numpy(); yb = cat22["P1_B400B"].to_numpy()
b4005_fit = np.polyfit(yb, x, 1); r_ = x - np.polyval(b4005_fit, yb)
out["b4005_vs_b400b_2204"] = {"corr": float(np.corrcoef(x, yb)[0, 1]), "fit_B4005_eq_a_B400B_plus_b": [float(b4005_fit[0]), float(b4005_fit[1])],
                              "r2": float(1 - r_.var() / x.var()), "resid_sd": float(r_.std())}
rw, cw = linear_sum_assignment(np.nan_to_num(Wd, nan=1e9))
def group_ok(a, b):
    return b.startswith("x100") and int(b[4]) + 1 == int(a[4])        # x1001 <-> B2xxx, x1002 <-> B3xxx, x1003 <-> B4xxx
ev_rows = []
for i, j in zip(rw, cw):
    a, b = OLD[i], NEW[j]
    p22, p23 = part22[a], part23[b]
    e1 = bool(p22 and p23 and p22["channel"] == p23["channel"] and abs(p22["corr"]) >= 0.9 and abs(p23["corr"]) >= 0.9)
    e2 = bool(Wd[i, j] <= 1.0)
    e3 = bool(np.isfinite(F[i, j]) and F[i, j] >= 0.8)
    rc = stats23[b]["changes_per_hour"] / max(stats22[a]["changes_per_hour"], 1e-9)
    rs = stats23[b]["median_nonzero_step"] / max(stats22[a]["median_nonzero_step"], 1e-12)
    e5 = bool(0.5 <= rc <= 2 and 0.5 <= rs <= 2)
    g = group_ok(a, b); score = int(e1) + int(e2) + int(e3) + int(e5)
    conf = "none" if not g else ("strong" if score >= 3 else "moderate" if score == 2 else "weak")
    ev_rows.append({"old_2204": a, "new_2305": b, "doc_2204": DOC[a], "group_consistent_name": g,
                    "E1_same_tracked_pv_abs_corr_ge_0.9": e1, "E2_wasserstein_over_sd_le_1": e2, "E3_fingerprint_ge_0.8": e3,
                    "E5_same_dynamics_factor_2": e5, "wasserstein_over_sd2204": float(Wd[i, j]),
                    "fingerprint_similarity": None if np.isnan(F[i, j]) else float(F[i, j]),
                    "changes_per_hour_ratio": rc, "median_step_ratio": rs,
                    "range_2204": [stats22[a]["min"], stats22[a]["max"]], "range_2305": [stats23[b]["min"], stats23[b]["max"]],
                    "evidence_score_0_4": score, "confidence": conf})
out["evidence_classification"] = {
    "proposer": "Hungarian assignment minimising the 1-Wasserstein distance in raw units (divided by the 22.04 SD)",
    "criteria": "E1 same tracked process variable with |r| >= 0.9 in both campaigns; E2 W/SD <= 1; E3 fingerprint similarity >= 0.8; "
                "E5 change frequency and median step within a factor of 2; naming group x1001<->B2xxx, x1002<->B3xxx, x1003<->B4xxx. "
                "strong: group and >= 3 of E1,E2,E3,E5; moderate: group and 2; weak: group and <= 1; none: group inconsistent. Post hoc.",
    "pairs": ev_rows}
for p in ev_rows:
    print(p["old_2204"], "->", p["new_2305"], p["confidence"], "score", p["evidence_score_0_4"], "W/sd", round(p["wasserstein_over_sd2204"], 2),
          "fp", p["fingerprint_similarity"])

if args.sensitivity:
    del cat22, cat23, s22, s23
    prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
    model = H.load_model("baseline_s0", len(prep.cols)); cv = np.load(os.path.join(H.RES, "models", "baseline_s0_chanvar.npy"))
    e1 = json.load(open(os.path.join(H.RES, "e1_baseline_s0.json"))); theta = e1["theta"]
    # baseline_s0 on 22.04 with its real inputs (stored E1 scores), E5 convention, for the ratio
    z = np.load(os.path.join(H.RES, "models", "baseline_s0_scores.npz"))
    test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
    fa22, h22 = 0, 0.0
    for i, df in enumerate(test, start=1):
        c, h = R.e5_far(z[f"test{i}"].astype(np.float64), R.e5_normal_mask(H.labels(df)), theta); fa22 += c; h22 += h
    good = {p["old_2204"]: p["new_2305"] for p in ev_rows if p["confidence"] in ("strong", "moderate")}
    grp = {p["old_2204"]: p["new_2305"] for p in ev_rows if p["group_consistent_name"]}
    variants = {"S1_strong_moderate_mapped_rest_mean": (good, None),
                "S2_group_consistent_mapped_B4005_mean": (grp, None),
                "S3_group_consistent_mapped_B4005_from_B400B": (grp, "P1_B4005")}
    out["sensitivity_baseline_s0_mapped"] = {"theta": theta, "far_2204_e5_convention_real_inputs": R.rate_summary(fa22, h22, 0.05),
                                             "note": "sensitivity only (not the main result): the original 68-input detector with renamed inputs restored; "
                                                     "tags without an accepted counterpart are imputed in 23.05 only (S1, S2), which is not a like-for-like configuration",
                                             "variants": {}}
    d23 = H.read_files("hai-23.05", [f"hai-train{i}.csv" for i in range(1, 5)])
    for vname, (mp, recon) in variants.items():
        fa, hours = 0, 0.0
        for df0 in d23:
            df = df0.copy()
            for j, c in enumerate(prep.cols):
                if c in df.columns:
                    continue
                if c in mp:
                    df[c] = df[mp[c]]
                elif recon == c and "P1_B400B" in mp:
                    df[c] = np.polyval(b4005_fit, df[mp["P1_B400B"]].to_numpy())
                else:
                    df[c] = prep.mu[j]
            sc = H.smooth_fast(H.score_from_residuals(H.residuals(model, prep.to_array(df)), cv))
            fa += len(H.alarm_episodes(sc, theta)); hours += len(sc) / 3600.0
        rs = R.rate_summary(fa, hours, 0.05)
        out["sensitivity_baseline_s0_mapped"]["variants"][vname] = {
            "mapping_used": mp, "reconstructed": recon, "imputed_with_2204_mean": [c for c in OLD if c not in mp and c != recon],
            "far_2305_e5_convention": rs, "ratio_vs_2204_e5_convention": R.rate_ratio_exact(fa, hours, fa22, h22, 0.05)}
        print(vname, rs["count"], round(rs["rate_per_h"], 3), flush=True)
out["meta"] = R.run_meta("e5b_tag_mapping.py", t0, {"none": None}, {"subsample_for_correlations_s": 10})
p = R.save_json(out, "e5b_tag_mapping.json")
print("wrote", p, "runtime", round(time.time() - t0, 1))
