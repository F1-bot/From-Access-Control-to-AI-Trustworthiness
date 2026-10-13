"""E5b (major revision): drift between HAI 22.04 and HAI 23.05 WITHOUT the imputation confound.

e5_drift.py fed the eight P1 setpoint tags that were renamed in 23.05 to the 22.04 detector as their
22.04 training means. Here the detector is RETRAINED on the channels that exist under the same name in
both campaigns (the eight renamed tags are excluded), with the recipe and seed of baseline_s0
(e1_baseline.py as run by run_all.sh: seed 0, 12 epochs x 250000 windows, no corruption; validation =
last 20% of train6; threshold = 99.9th percentile of the smoothed validation score).
(a) training; a 68-channel control trained through the same code path must reproduce baseline_s0.
(b) HAI 22.04 test files, E1 conventions (58 attacks; false alarms per hour of normal operation).
(c) HAI 23.05 normal data (the four training files, 249 h; secondary: normal periods of the two 23.05
    test files): false alarms per hour with exact Poisson bounds, under the E1 convention and the E5
    convention (22.04 normal periods exclude 5 min before and 10 min after each attack); rate ratios.
    Decomposition of the E5 confound: baseline_s0 with real inputs, baseline_s0 with the E5 imputation
    (recomputed), and the retrained detector, under both conventions.
(d) Hourly residual monitor exactly as in E5 (hourly median of the smoothed score; threshold = maximum
    of sliding hourly medians on validation), TPR/FPR with exact bounds and the ROC AUC of hourly
    medians (drifted vs in-distribution hours); AUC of the original E5 monitor from e5_drift.json.
Writes results/e5b_drift_retrained.json and results/models/e5b_shared60_s0.* (new files only).
Usage: python e5b_drift_retrained.py [--no_control]"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from safetensors.torch import save_file
import hai_lib as H
import revml_lib as R
from aitcert_path import stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=R.E1_RECIPE["seed"])
ap.add_argument("--epochs", type=int, default=R.E1_RECIPE["epochs"])
ap.add_argument("--per_epoch", type=int, default=R.E1_RECIPE["per_epoch"])
ap.add_argument("--no_control", action="store_true")
args = ap.parse_args()
t0 = time.time()
A = 0.05
H.set_seed(args.seed)
train = H.read_files("hai-22.04", [f"train{i}.csv" for i in range(1, 7)])
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
y22 = [H.labels(df) for df in test]
prep68 = H.fit_prep(train)
stored = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
assert prep68.cols == stored.cols and np.allclose(prep68.mu, stored.mu) and np.allclose(prep68.sd, stored.sd)
names23 = [f"hai-train{i}.csv" for i in range(1, 5)]
d23 = H.read_files("hai-23.05", names23)
cols23 = set(d23[0].columns)
assert all(set(df.columns) == cols23 for df in d23)
shared = [c for c in prep68.cols if c in cols23]
excluded = [c for c in prep68.cols if c not in cols23]
sel = np.array([c in cols23 for c in prep68.cols])
prep60 = H.Prep(shared, prep68.mu[sel], prep68.sd[sel])
out = {"recipe": {**R.E1_RECIPE, "seed": args.seed, "epochs": args.epochs, "per_epoch": args.per_epoch},
       "n_features": len(shared), "features": shared, "excluded_renamed_tags": excluded}
print("features", len(shared), "excluded", excluded, flush=True)

def train_detector(prep, tag):
    arrs = [prep.to_array(df) for df in train]
    cut = int(len(arrs[-1]) * 0.8)
    val = [arrs[-1][cut:]]
    tr = arrs[:-1] + [arrs[-1][:cut]]
    log = []
    model = H.train_forecaster(tr, val, seed=args.seed, epochs=args.epochs, per_epoch=args.per_epoch, corrupt=0.0, log=log)
    res_val = H.residuals(model, val[0])
    cv = np.nanmean(res_val, axis=0) + 1e-6
    s_val = H.smooth_fast(H.score_from_residuals(res_val, cv))
    thetas = {q: float(np.nanquantile(s_val, q)) for q in (0.995, 0.999, 0.9995)}
    return {"model": model, "cv": cv, "s_val": s_val, "val": val[0], "thetas": thetas, "theta": thetas[0.999], "log": log,
            "train_windows": int(sum(len(a) - H.W for a in tr))}

def score(model, cv, arr):
    res = H.residuals(model, arr)
    return H.smooth_fast(H.score_from_residuals(res, cv)), res

def eval_2204(model, cv, prep, theta):
    evs, scs, contrib = [], {}, []
    for i, (df, y) in enumerate(zip(test, y22), start=1):
        sc, res = score(model, cv, prep.to_array(df))
        scs[f"test{i}"] = sc
        evs.append(H.evaluate_file(sc, y, theta))
        m = R.e5_normal_mask(y); m[:H.W] = False
        contrib.append(res[m] / cv[None, :])
    agg = H.aggregate(evs)
    agg["detected_vector"] = [int(d) for e in evs for d in e["detected"]]
    return agg, scs, np.concatenate(contrib)

# ---------------------------------------------------------------- (a) retrain on the 60 shared channels
D = train_detector(prep60, "e5b_shared60")
model, cv, theta = D["model"], D["cv"], D["theta"]
mdir = os.path.join(H.RES, "models")
tag = f"e5b_shared60_s{args.seed}"
save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(mdir, f"{tag}.safetensors"))
np.save(os.path.join(mdir, f"{tag}_chanvar.npy"), cv)
prep60.save(os.path.join(mdir, "e5b_prep_shared60.json"))
val_eps = H.alarm_episodes(D["s_val"], theta)
out["training"] = {"theta": theta, "thetas": D["thetas"], "train_windows": D["train_windows"], "train_log": D["log"],
                   "validation_episodes": [[int(a), int(b)] for a, b in val_eps],
                   "validation_alarms_per_h": R.rate_summary(len(val_eps), len(D["val"]) / 3600.0, A)}
print("theta", theta, "val episodes", len(val_eps), flush=True)

# ---------------------------------------------------------------- (b) 22.04 test, E1 conventions
agg, sc22, contrib22 = eval_2204(model, cv, prep60, theta)
by_q = {}
for q, th in D["thetas"].items():
    evs = [H.evaluate_file(sc22[f"test{i}"], y22[i - 1], th) for i in range(1, 5)]
    by_q[str(q)] = H.aggregate(evs)
np.savez_compressed(os.path.join(mdir, f"{tag}_scores.npz"), **{k: v.astype(np.float32) for k, v in sc22.items()}, val=D["s_val"].astype(np.float32))
e1b0 = json.load(open(os.path.join(H.RES, "e1_baseline_s0.json")))
vec0 = np.array(e1b0["detected_vector"]); vec = np.array(agg["detected_vector"])
out["test_2204_e1_convention"] = {
    "aggregate": agg, "by_quantile": by_q,
    "recall": R.prop_summary(agg["events_detected"], agg["events"], A),
    "recall_accept_tau_0.50": S.accept_proportion(agg["events_detected"], agg["events"], 0.50, ">=", A),
    "far": R.rate_summary(agg["false_alarms"], agg["normal_hours"], A),
    "far_accept_tau_0.25": S.accept_rate(agg["false_alarms"], agg["normal_hours"], 0.25, A),
    "vs_baseline_s0": {"baseline_s0_detected": int(vec0.sum()), "baseline_s0_false_alarms": e1b0["aggregate"]["false_alarms"],
                       "only_baseline": int(((vec0 == 1) & (vec == 0)).sum()), "only_retrained": int(((vec0 == 0) & (vec == 1)).sum()),
                       "far_ratio_retrained_over_baseline": R.rate_ratio_exact(agg["false_alarms"], agg["normal_hours"],
                                                                               e1b0["aggregate"]["false_alarms"], e1b0["aggregate"]["normal_hours"], A)}}
fa_e5c, h_e5c = 0, 0.0
for i, y in enumerate(y22, start=1):
    c, h = R.e5_far(sc22[f"test{i}"], R.e5_normal_mask(y), theta); fa_e5c += c; h_e5c += h
out["test_2204_e5_convention"] = R.rate_summary(fa_e5c, h_e5c, A)
print("22.04 recall", agg["events_detected"], "/", agg["events"], "FA", agg["false_alarms"], "h", round(agg["normal_hours"], 3),
      "| E5 conv FA", fa_e5c, "h", round(h_e5c, 3), flush=True)

# ---------------------------------------------------------------- (c) 23.05 normal data
per_file, n_fa, h_e1, h_e5, posm, contrib23, hourly23 = [], 0, 0.0, 0.0, [], [], []
for name, df in zip(names23, d23):
    sc, res = score(model, cv, prep60.to_array(df))
    eps = H.alarm_episodes(sc, theta)
    hm = R.hourly_medians(sc)
    posm += hm
    c, _ = R.hourly_bins(len(sc), np.ones(len(sc), bool), [a for a, _ in eps])
    hourly23 += c.tolist()
    contrib23.append(np.nanmean(res[H.W:] / cv[None, :], axis=0))
    per_file.append({"file": name, "seconds": int(len(sc)), "false_alarms": len(eps), "hours_e5": len(sc) / 3600.0,
                     "hours_e1": (len(sc) - H.W) / 3600.0, "rate_per_h_e5": len(eps) / (len(sc) / 3600.0),
                     "median_hourly_median_score": float(np.median(hm))})
    n_fa += len(eps); h_e1 += (len(sc) - H.W) / 3600.0; h_e5 += len(sc) / 3600.0
out["normal_2305_train_files"] = {"per_file": per_file, "e1_convention": R.rate_summary(n_fa, h_e1, A),
                                  "e5_convention": R.rate_summary(n_fa, h_e5, A),
                                  "hourly_counts_dispersion": R.dispersion_test(hourly23, [1.0] * len(hourly23), B=20000, seed=0)}
print("23.05 FA", n_fa, "h", h_e5, "rate", n_fa / h_e5, flush=True)
# secondary: normal periods of the two 23.05 test files (labels in separate files)
sec = {"files": [], "e1": [0, 0.0], "e5": [0, 0.0], "recall": [0, 0]}
posm_test = []
for tf, lf in (("hai-test1.csv", "label-test1.csv"), ("hai-test2.csv", "label-test2.csv")):
    df = H.read_files("hai-23.05", [tf])[0]
    lab = pd.read_csv(os.path.join(H.DATA, "hai-23.05", lf))
    # label-test2.csv stores timestamps at minute precision only: check row alignment at minute level
    assert len(lab) == len(df)
    t_lab = pd.to_datetime(lab["timestamp"].astype(str).str.strip()).dt.floor("min").values
    t_dat = pd.to_datetime(df["timestamp"].astype(str).str.strip()).dt.floor("min").values
    assert (t_lab == t_dat).all(), f"label rows not aligned with {tf}"
    y = lab["label"].to_numpy().astype(np.int8)
    sc, _ = score(model, cv, prep60.to_array(df))
    ev = H.evaluate_file(sc, y, theta)
    m5 = R.e5_normal_mask(y)
    c5, hh5 = R.e5_far(sc, m5, theta)
    posm_test += R.hourly_medians(sc, m5)
    sec["files"].append({"file": tf, "attacks": len(ev["events"]), "detected": int(sum(ev["detected"])),
                         "false_alarms_e1": ev["false_alarms"], "normal_hours_e1": ev["normal_hours"],
                         "false_alarms_e5": c5, "normal_hours_e5": hh5})
    sec["e1"][0] += ev["false_alarms"]; sec["e1"][1] += ev["normal_hours"]
    sec["e5"][0] += c5; sec["e5"][1] += hh5
    sec["recall"][0] += int(sum(ev["detected"])); sec["recall"][1] += len(ev["events"])
out["normal_2305_test_files_secondary"] = {
    "per_file": sec["files"], "e1_convention": R.rate_summary(sec["e1"][0], sec["e1"][1], A),
    "e5_convention": R.rate_summary(sec["e5"][0], sec["e5"][1], A),
    "event_recall_2305_attacks_supplementary": R.prop_summary(sec["recall"][0], sec["recall"][1], A)}
out["normal_2305_all"] = {"e1_convention": R.rate_summary(n_fa + sec["e1"][0], h_e1 + sec["e1"][1], A),
                          "e5_convention": R.rate_summary(n_fa + sec["e5"][0], h_e5 + sec["e5"][1], A)}
out["rate_ratios_2305_over_2204"] = {
    "e1_convention_train_files": R.rate_ratio_exact(n_fa, h_e1, agg["false_alarms"], agg["normal_hours"], A),
    "e5_convention_train_files": R.rate_ratio_exact(n_fa, h_e5, fa_e5c, h_e5c, A),
    "e1_convention_all_2305_normal": R.rate_ratio_exact(n_fa + sec["e1"][0], h_e1 + sec["e1"][1], agg["false_alarms"], agg["normal_hours"], A)}
# per-channel contribution to the normalised residual (drift localisation)
c22 = np.nanmean(contrib22, axis=0); c23 = np.average(np.array(contrib23), axis=0, weights=[f["seconds"] for f in per_file])
order = np.argsort(-c23)
out["channel_contributions"] = {"mean_normalised_residual_2204_normal": dict(zip(shared, c22.round(4).tolist())),
                                "mean_normalised_residual_2305": dict(zip(shared, c23.round(4).tolist())),
                                "top10_2305": [[shared[j], float(c23[j]), float(c22[j])] for j in order[:10]],
                                "share_of_2305_score_top5": float(c23[order[:5]].sum() / c23.sum())}

# ---------------------------------------------------------------- (d) hourly residual monitor (as E5)
ref_vals = R.hourly_medians(D["s_val"], stride=300)
thr_med = float(np.max(ref_vals))
negm = [v for i, y in enumerate(y22, start=1) for v in R.hourly_medians(sc22[f"test{i}"], R.e5_normal_mask(y))]
tp = int(sum(v > thr_med for v in posm)); fp = int(sum(v > thr_med for v in negm))
tp2 = int(sum(v > thr_med for v in posm_test))
e5 = json.load(open(os.path.join(H.RES, "e5_drift.json")))
rm = e5["residual_monitor_hourly"]
out["residual_monitor_hourly"] = {
    "threshold": thr_med, "n_ref_windows": len(ref_vals), "n_pos": len(posm), "n_neg": len(negm), "tp": tp, "fp": fp,
    "tpr": R.prop_summary(tp, len(posm), A), "fpr": R.prop_summary(fp, len(negm), A),
    "auc": R.auc_mw(posm, negm), "auc_bootstrap_ci95_iid": R.auc_bootstrap(posm, negm, B=10000, seed=0),
    "secondary_pos_2305_test_normal_hours": {"n": len(posm_test), "tp": tp2, "tpr": R.prop_summary(tp2, len(posm_test), A),
                                             "auc_vs_neg": R.auc_mw(posm_test, negm) if posm_test else None},
    "pos_values": posm, "neg_values": negm, "ref_values": ref_vals, "pos_values_2305_test_normal": posm_test,
    "original_e5_monitor": {"threshold": rm["threshold"], "n_pos": rm["n_pos"], "n_neg": rm["n_neg"], "tp": rm["tp"], "fp": rm["fp"],
                            "auc": R.auc_mw(rm["pos_values"], rm["neg_values"]),
                            "auc_bootstrap_ci95_iid": R.auc_bootstrap(rm["pos_values"], rm["neg_values"], B=10000, seed=0)},
    "exploratory": True}
print("monitor thr", thr_med, "TP", tp, "/", len(posm), "FP", fp, "/", len(negm), "AUC", out["residual_monitor_hourly"]["auc"],
      "| E5 AUC", out["residual_monitor_hourly"]["original_e5_monitor"]["auc"], flush=True)
del model; torch.cuda.empty_cache()

# ---------------------------------------------------------------- decomposition of the E5 confound (baseline_s0)
b0 = H.load_model("baseline_s0", len(stored.cols))
cv0 = np.load(os.path.join(mdir, "baseline_s0_chanvar.npy"))
th0 = e1b0["theta"]
z0 = np.load(os.path.join(mdir, "baseline_s0_scores.npz"))
shm = np.array([c in cols23 for c in stored.cols])
def shared_score(arr):  # e5_drift.py: renamed channels set to the training mean, score over shared channels
    a = arr.copy(); a[:, ~shm] = 0.0
    r = H.residuals(b0, a)[:, shm]
    return H.smooth_fast(H.score_from_residuals(r, cv0[shm]))
tr6 = stored.to_array(train[5]); ref = tr6[int(len(tr6) * 0.8):]
th_sh = float(np.nanquantile(shared_score(ref), 0.999))
dec = {"baseline_s0_real_inputs": {}, "baseline_s0_e5_imputation": {"theta_shared": th_sh, "theta_shared_e5_json": e5["theta_shared"]}}
fa1, fa5, h5, fai1, nh1, fai5 = 0, 0, 0.0, 0, 0.0, 0
for i, (df, y) in enumerate(zip(test, y22), start=1):
    s_real = z0[f"test{i}"].astype(np.float64)
    m5 = R.e5_normal_mask(y)
    c, h = R.e5_far(s_real, m5, th0); fa5 += c; h5 += h
    s_imp = shared_score(stored.to_array(df))
    ev = H.evaluate_file(s_imp, y, th_sh); fai1 += ev["false_alarms"]; nh1 += ev["normal_hours"]
    c, _ = R.e5_far(s_imp, m5, th_sh); fai5 += c
dec["baseline_s0_real_inputs"] = {"e1_convention_2204": R.rate_summary(e1b0["aggregate"]["false_alarms"], e1b0["aggregate"]["normal_hours"], A),
                                  "e5_convention_2204": R.rate_summary(fa5, h5, A),
                                  "note": "no 23.05 value: the eight renamed inputs are unavailable without imputation or mapping"}
fa23i, h23i = 0, 0.0
for df in d23:
    dfi = df.copy()
    for j, c in enumerate(stored.cols):
        if c not in dfi.columns:
            dfi[c] = stored.mu[j]
    s = shared_score(stored.to_array(dfi)); fa23i += len(H.alarm_episodes(s, th_sh)); h23i += len(s) / 3600.0
dec["baseline_s0_e5_imputation"].update({
    "e1_convention_2204": R.rate_summary(fai1, nh1, A), "e5_convention_2204": R.rate_summary(fai5, h5, A),
    "e5_convention_2305": R.rate_summary(fa23i, h23i, A),
    "reproduces_e5_json": {"alarms_2204": [fai5, e5["alarms_2204"]], "alarms_2305": [fa23i, e5["alarms_2305"]],
                           "identical": fai5 == e5["alarms_2204"] and fa23i == e5["alarms_2305"]},
    "ratio_2305_over_2204_e5_convention": R.rate_ratio_exact(fa23i, h23i, fai5, h5, A)})
dec["e5b_retrained_shared60"] = {"e1_convention_2204": out["test_2204_e1_convention"]["far"],
                                 "e5_convention_2204": out["test_2204_e5_convention"],
                                 "e1_convention_2305": out["normal_2305_train_files"]["e1_convention"],
                                 "e5_convention_2305": out["normal_2305_train_files"]["e5_convention"]}
out["imputation_decomposition"] = dec
print("decomp: b0 real E5conv", fa5, round(h5, 3), "| imputed E1conv", fai1, round(nh1, 3), "E5conv", fai5, "23.05", fa23i, flush=True)
del b0; torch.cuda.empty_cache()

# ---------------------------------------------------------------- control: 68 channels through the same code path
if not args.no_control:
    C = train_detector(prep68, "control68")
    aggc, _, _ = eval_2204(C["model"], C["cv"], prep68, C["theta"])
    out["control68"] = {"theta": C["theta"], "theta_baseline_s0": e1b0["theta"],
                        "theta_identical": bool(abs(C["theta"] - e1b0["theta"]) < 1e-9),
                        "events_detected": aggc["events_detected"], "false_alarms": aggc["false_alarms"],
                        "detected_vector_identical": aggc["detected_vector"] == e1b0["detected_vector"],
                        "false_alarms_identical": aggc["false_alarms"] == e1b0["aggregate"]["false_alarms"]}
    print("control68", out["control68"], flush=True)
out["meta"] = R.run_meta("e5b_drift_retrained.py", t0, {"training": args.seed, "auc_bootstrap": 0, "dispersion_mc": 0},
                         {"model_files": [f"results/models/{tag}.safetensors", f"results/models/{tag}_chanvar.npy",
                                          f"results/models/{tag}_scores.npz", "results/models/e5b_prep_shared60.json"]})
p = R.save_json(out, "e5b_drift_retrained.json")
print("wrote", p, "runtime", round(time.time() - t0, 1))
