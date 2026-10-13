"""ID-3 re-analysis with independent units (major revision; complements the pre-registered E3).

(a) Poison detection by the PRE-REGISTERED sanitiser of E3 (trusted model on train1-2, seed 100,
    windows above the 99.5th percentile of its validation window scores are removed) at beta = 1%,
    with the poisoned SOURCE TRACE (7 attacks of test1; each replicated 6 times) as the unit:
    per-trace removal fractions, a cluster bootstrap over traces (B = 10000, seed 0), the trace-level
    exact bounds, and the clean-window removal fraction (pooled, per training array, hourly-cluster
    bootstrap). The trusted model is retrained with the E3 recipe; the script checks that it reproduces
    the stored E3 threshold and removal counts.
(b) Pre-registered degradation criterion pdeg <= 0.10 at beta = 1%: paired comparison of a clean model
    trained with the E3 recipe (seed 0, 10 epochs x 200k windows; the counterfactual of the poisoned
    model) and the E3 poisoned models on the 51 attacks of test2-4. Exact one-sided 95% upper bound of
    the recall degradation; decision accept / reject / inconclusive. Secondary references: baseline_s0
    (E1) and the E3b clean model; seed-to-seed discordance of the five E1 baselines as a noise reference.
Writes results/id3_units.json.
Usage: python id3_units.py [--retrain_poisoned]"""
import argparse, json, os, time
import numpy as np, pandas as pd
import hai_lib as H
import revml_lib as R
from aitcert_path import stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--beta", type=float, default=0.01)
ap.add_argument("--B", type=int, default=10000)
ap.add_argument("--retrain_poisoned", action="store_true", help="also retrain the E3 poisoned (simple) model as a reproducibility check")
args = ap.parse_args()
t0 = time.time()
EP, PE = R.E3_RECIPE["epochs"], R.E3_RECIPE["per_epoch"]
A = 0.05
e3 = json.load(open(os.path.join(H.RES, "e3_poison_beta0.01.json")))
E1B0 = json.load(open(os.path.join(H.RES, "e1_baseline_s0.json")))   # test1 attacks are the first 7 entries

# ---------------------------------------------------------------- data exactly as in e3_poison.py
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
train = H.read_files("hai-22.04", [f"train{i}.csv" for i in range(1, 7)])
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrs = [prep.to_array(df) for df in train]
cut = int(len(arrs[-1]) * 0.8)
val = [arrs[-1][cut:]]
clean = arrs[:-1] + [arrs[-1][:cut]]
n_clean_windows = sum(len(a) - H.W for a in clean)
t1 = prep.to_array(test[0]); ev1 = H.events_from_labels(H.labels(test[0]))
segs = [t1[max(0, s - 30 - H.W):min(len(t1), e + 90)] for s, e in ev1]
seg_windows = sum(len(sg) - H.W for sg in segs)
reps = int(np.ceil(args.beta * n_clean_windows / seg_windows))
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
ids1 = attacks[attacks.test_file == "test1"].attack_id.tolist()
assert len(ids1) == len(segs) == 7
def frog(seg, gamma):
    base = np.repeat(seg[:H.W + 30].mean(0, keepdims=True), len(seg), 0)
    return (base + gamma * (seg - base)).astype(np.float32)
gammas = np.linspace(0.1, 1.0, reps)
poison = {"simple": [sg for _ in range(reps) for sg in segs],
          "frog": [frog(sg, g) for g in gammas for sg in segs]}
# poison list index j -> source segment j % 7, replica j // 7 (both kinds)

def calibrate(model):
    rv = H.residuals(model, val[0]); cv = np.nanmean(rv, 0) + 1e-6
    sv = H.smooth_fast(H.score_from_residuals(rv, cv))
    return cv, float(np.nanquantile(sv, 0.999))

def fit(arrays, idx=None, seed=0):
    m = H.train_forecaster(arrays, val, seed=seed, epochs=EP, per_epoch=PE, train_idx=idx)
    cv, th = calibrate(m)
    return m, cv, th

def evaluate(model, chan_var, theta):
    evs = []
    for i in (2, 3, 4):
        arr = prep.to_array(test[i - 1]); res = H.residuals(model, arr)
        sc = H.smooth_fast(H.score_from_residuals(res, chan_var))
        evs.append(H.evaluate_file(sc, H.labels(test[i - 1]), theta))
    a = H.aggregate(evs)
    a["detected_vector"] = [int(d) for e in evs for d in e["detected"]]
    return a

out = {"beta": args.beta, "replicas": reps, "n_traces": len(segs), "trace_ids": ids1,
       "trace_windows_per_replica": [int(len(sg) - H.W) for sg in segs], "clean_windows": int(n_clean_windows)}

# ---------------------------------------------------------------- (a) trusted sanitiser (pre-registered)
H.set_seed(0)
trusted, tcv, _ = fit(clean[:2], seed=R.E3_RECIPE["trusted_seed"])
tv = H.window_scores(trusted, val, H.window_index(val), tcv)
t_thr = float(np.quantile(tv, 0.995))
# score clean + poison together, in the same order and batches as e3_poison.py
scored = {}
for kind in ("simple", "frog"):
    arrays = clean + poison[kind]
    idx = H.window_index(arrays)
    scored[kind] = (idx, H.window_scores(trusted, arrays, idx, tcv))
idx_s, sc_s = scored["simple"]
is_clean = idx_s[:, 0] < len(clean)
clean_idx = idx_s[is_clean]
clean_removed = sc_s[is_clean] > t_thr
idx_f, sc_f = scored["frog"]
clean_removed_frog_pass = sc_f[idx_f[:, 0] < len(clean)] > t_thr
ref = e3["runs"]["simple"]["sanitiser"]
repro = {"threshold_stored": ref["threshold"], "threshold_retrained": t_thr,
         "threshold_identical": bool(abs(t_thr - ref["threshold"]) < 1e-9),
         "clean_removed_stored": ref["clean_removed"], "clean_removed_retrained": int(clean_removed.sum()),
         "clean_removed_retrained_frog_pass": int(clean_removed_frog_pass.sum())}
# clean windows: pooled, per array, and cluster bootstrap over clock hours of each array
per_arr = []
for a in range(len(clean)):
    sel = clean_idx[:, 0] == a
    per_arr.append({"array": ["train1", "train2", "train3", "train4", "train5", "train6[:80%]"][a],
                    "windows": int(sel.sum()), "removed": int(clean_removed[sel].sum()),
                    "fraction": float(clean_removed[sel].mean())})
hour_id = clean_idx[:, 0] * 10_000 + clean_idx[:, 1] // 3600
uh, inv = np.unique(hour_id, return_inverse=True)
h_rem = np.bincount(inv, weights=clean_removed.astype(float)); h_n = np.bincount(inv).astype(float)
rng = np.random.default_rng(0)
bs = np.empty(args.B)
for b in range(args.B):
    k = rng.integers(0, len(uh), len(uh)); bs[b] = h_rem[k].sum() / h_n[k].sum()
out["clean_removal"] = {"removed": int(clean_removed.sum()), "windows": int(len(clean_removed)),
                        "fraction": float(clean_removed.mean()), "per_array": per_arr,
                        "hour_cluster_bootstrap": {"n_clusters": int(len(uh)), "B": args.B, "seed": 0,
                                                   "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                                   "lcb95_one_sided": float(np.quantile(bs, 0.05))},
                        "tau_fpr": 0.05}
out["sanitiser"] = {"threshold": t_thr, "reproduction_check": repro}
print("trusted threshold", t_thr, "stored", ref["threshold"], "clean removed", int(clean_removed.sum()), "stored", ref["clean_removed"], flush=True)

def cluster_bootstrap(removed, windows, B, seed=0):
    """Resample traces with replacement; pooled (window-weighted) and unweighted mean removal."""
    removed = np.asarray(removed, float); windows = np.asarray(windows, float)
    rng = np.random.default_rng(seed)
    pooled = np.empty(B); mean = np.empty(B)
    n = len(removed)
    for b in range(B):
        k = rng.integers(0, n, n)
        pooled[b] = removed[k].sum() / windows[k].sum(); mean[b] = (removed[k] / windows[k]).mean()
    q = lambda v, p: float(np.quantile(v, p))
    return {"B": B, "seed": seed, "method": "percentile bootstrap over the 7 source traces",
            "pooled_ci95": [q(pooled, 0.025), q(pooled, 0.975)], "pooled_lcb95_one_sided": q(pooled, 0.05),
            "mean_ci95": [q(mean, 0.025), q(mean, 0.975)], "mean_lcb95_one_sided": q(mean, 0.05),
            "n_distinct_resamples_possible": "C(13,7) = 1716"}

out["traces"] = {}
for kind in ("simple", "frog"):
    idx_k, sc_k = scored[kind]
    isp = idx_k[:, 0] >= len(clean)
    pidx = idx_k[isp].copy(); pidx[:, 0] -= len(clean)          # index into the poison list
    prem = sc_k[isp] > t_thr
    src = pidx[:, 0] % len(segs); rep = pidx[:, 0] // len(segs)
    # phase of each poison window: target time t (absolute, in test1) relative to the attack [s, e]
    seg_start = np.array([max(0, s - 30 - H.W) for s, e in ev1])
    t_abs = seg_start[src] + pidx[:, 1]
    s_arr = np.array([s for s, e in ev1])[src]; e_arr = np.array([e for s, e in ev1])[src]
    phase = np.where(t_abs < s_arr, 0, np.where(t_abs <= e_arr + H.W, 1, 2))   # 0 pre-attack context, 1 window touches the attack, 2 post tail
    rows = []
    for s in range(len(segs)):
        sel = src == s
        per_rep = [float(prem[sel & (rep == r)].mean()) for r in range(reps)]
        ph = {name: {"windows": int((sel & (phase == k)).sum()), "removed": int(prem[sel & (phase == k)].sum()),
                     "fraction": float(prem[sel & (phase == k)].mean()) if (sel & (phase == k)).any() else None}
              for k, name in enumerate(("pre_attack_context", "attack_containing", "post_attack_tail"))}
        rows.append({"attack_id": ids1[s], "windows": int(sel.sum()), "removed": int(prem[sel].sum()),
                     "fraction": float(prem[sel].mean()), "per_replica_fraction": per_rep,
                     "replicas_identical": bool(np.ptp(per_rep) == 0), "by_phase": ph,
                     "detected_by_baseline_s0_e1": int(E1B0["detected_vector"][s])})
    rem = [r["removed"] for r in rows]; win = [r["windows"] for r in rows]
    fr = np.array([r["fraction"] for r in rows])
    rem_a = [r["by_phase"]["attack_containing"]["removed"] for r in rows]; win_a = [r["by_phase"]["attack_containing"]["windows"] for r in rows]
    st_ = ref if kind == "simple" else e3["runs"]["frog"]["sanitiser"]
    out["traces"][kind] = {
        "per_trace": rows,
        "pooled_fraction": float(sum(rem) / sum(win)), "mean_of_trace_fractions": float(fr.mean()),
        "median_trace_fraction": float(np.median(fr)), "min_trace_fraction": float(fr.min()), "max_trace_fraction": float(fr.max()),
        "poison_removed_retrained": int(sum(rem)), "poison_removed_stored": st_["poison_removed"],
        "poison_windows": int(sum(win)), "poison_windows_stored": st_["poison_windows"],
        "window_level_cp_lcb95_as_reported": S.cp_lower(st_["poison_removed"], st_["poison_windows"], A),
        "cluster_bootstrap": cluster_bootstrap(rem, win, args.B, 0),
        "trace_level": {f"traces_removed_ge_{thr}": R.prop_summary(int((fr >= thr).sum()), len(fr), A) for thr in (0.5, 0.8)},
        "best_possible_trace_level_lcb95": S.cp_lower(len(fr), len(fr), A),
        "traces_needed_for_lcb_ge_0.80_with_zero_misses": S.n_zero_failure(A, 0.20),
        "phase_pooled": {name: {"windows": int((phase == k).sum()), "removed": int(prem[phase == k].sum()),
                                "fraction": float(prem[phase == k].mean())}
                         for k, name in enumerate(("pre_attack_context", "attack_containing", "post_attack_tail"))},
        "oracle_max_pdet_window_level": float((phase == 1).mean()),
        "oracle_note": "a sanitiser that removes exactly the windows whose input or target touches the attack interval reaches this pdet",
        "attack_containing_cluster_bootstrap": cluster_bootstrap(rem_a, win_a, args.B, 0)}
    print(kind, "pooled", out["traces"][kind]["pooled_fraction"], "per trace", np.round(fr, 3).tolist(),
          "bootstrap", out["traces"][kind]["cluster_bootstrap"]["pooled_ci95"], flush=True)
out["sanitiser"]["reproduction_check"]["poison_removed_identical"] = all(
    out["traces"][k]["poison_removed_retrained"] == out["traces"][k]["poison_removed_stored"] for k in ("simple", "frog"))
del trusted

# ---------------------------------------------------------------- (b) degradation criterion
m0, cv0, th0 = fit(clean, seed=0)
ev_clean = evaluate(m0, cv0, th0)
print("clean (E3 recipe)", ev_clean["events_detected"], ev_clean["false_alarms"], flush=True)
del m0
vec_clean = ev_clean["detected_vector"]
e1 = json.load(open(os.path.join(H.RES, "e1_baseline_s0.json")))
n1 = len([1 for s, e in ev1])
vec_b0 = e1["detected_vector"][n1:]
e3b = json.load(open(os.path.join(H.RES, "e3b_poison_dose.json")))
vec_e3b = e3b["clean"]["detected_vector"]
deg = {"clean_reference_models": {
           "clean_e3_recipe": {"recipe": "seed 0, 10 epochs x 200000 windows, clean training arrays (counterfactual of the E3 poisoned model)",
                               "events_detected": ev_clean["events_detected"], "event_recall": ev_clean["event_recall"],
                               "false_alarms": ev_clean["false_alarms"], "normal_hours": ev_clean["normal_hours"],
                               "theta": th0, "detected_vector": vec_clean},
           "baseline_s0_e1": {"recipe": "E1, seed 0, 12 epochs x 250000 windows", "events_detected": int(sum(vec_b0)), "detected_vector": vec_b0},
           "e3b_clean": {"recipe": "E3b, seed 0, 8 epochs x 150000 windows", "events_detected": int(sum(vec_e3b)), "detected_vector": vec_e3b}},
       "tests": {}}
for kind in ("simple", "frog"):
    for state in ("no_sanitisation", "sanitised"):
        vec_p = e3["runs"][kind][state]["detected_vector"]
        for cname, vc in (("clean_e3_recipe", vec_clean), ("baseline_s0_e1", vec_b0), ("e3b_clean", vec_e3b)):
            deg["tests"][f"{kind}/{state}/vs_{cname}"] = R.paired_degradation(vc, vec_p, tau=0.10, alpha=A)
prim = deg["tests"]["simple/no_sanitisation/vs_clean_e3_recipe"]
deg["primary"] = {"comparison": "simple/no_sanitisation vs clean_e3_recipe", **prim}
# noise reference: discordance between independently seeded clean E1 baselines on test2-4
seeds = {}
for s in range(5):
    seeds[s] = json.load(open(os.path.join(H.RES, f"e1_baseline_s{s}.json")))["detected_vector"][n1:]
pairs = []
for i in range(5):
    for j in range(i + 1, 5):
        a, b = np.array(seeds[i]), np.array(seeds[j])
        pairs.append({"seeds": [i, j], "n10": int(((a == 1) & (b == 0)).sum()), "n01": int(((a == 0) & (b == 1)).sum())})
deg["seed_noise_reference_e1"] = {"pairs": pairs, "max_discordant": max(p["n10"] + p["n01"] for p in pairs),
                                  "recall_test2_4_by_seed": {str(s): float(np.mean(v)) for s, v in seeds.items()}}
if args.retrain_poisoned:
    arrays = clean + poison["simple"]
    mp, cvp, thp = fit(arrays, H.window_index(arrays), seed=0)
    evp = evaluate(mp, cvp, thp)
    deg["poisoned_simple_retrained_check"] = {"detected_vector": evp["detected_vector"], "events_detected": evp["events_detected"],
                                              "false_alarms": evp["false_alarms"],
                                              "identical_to_stored": evp["detected_vector"] == e3["runs"]["simple"]["no_sanitisation"]["detected_vector"]
                                              and evp["false_alarms"] == e3["runs"]["simple"]["no_sanitisation"]["false_alarms"]}
    del mp
out["degradation"] = deg
out["meta"] = R.run_meta("id3_units.py", t0, {"trusted": R.E3_RECIPE["trusted_seed"], "clean_model": 0, "bootstrap": 0},
                         {"recipe": R.E3_RECIPE, "alpha": A})
p = R.save_json(out, "id3_units.json")
print(json.dumps({"primary": {k: prim[k] for k in ("n10_clean_only", "n01_other_only", "recall_clean", "recall_other", "ucb95_one_sided_exact", "decision")},
                  "repro": out["sanitiser"]["reproduction_check"], "runtime_s": out["meta"]["runtime_s"]}, indent=1))
print("wrote", p)
