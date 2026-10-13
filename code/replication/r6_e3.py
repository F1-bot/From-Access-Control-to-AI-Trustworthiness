"""R6 (E3 of the replication): poisoning drill for data integrity (ID-3) on HAI 21.03, with the design of the frozen plan.

Poison: the traces of the 20 attacks of test2, each restricted to the attack interval plus its W-second input history (so
that every poisoned window contains attack samples), injected into the training data as 'normal' and replicated to a
budget of beta = 1% of the clean training windows (simple replication, as E3). Models use the E3 recipe (10 epochs x
200,000 windows) and the operating-point rule of the plan.
  * Screening (planned sanitiser of E3): a trusted model trained on train1 only (seed 100) scores every training window;
    windows above the 99.5th percentile of its validation window scores are removed. Poison detection = removed share of
    the poisoned windows, with the 20 source traces as units (percentile cluster bootstrap, B = 10,000, seed 0); clean
    removal = removed share of the clean windows (exact bound). Requirement: detection >= 0.80 and clean removal <= 0.05.
  * Degradation: the poisoned pipeline (no screening) against a clean pipeline with the same recipe, three seeds each; an
    attack counts as detected by a pipeline if at least two of its three models detect it (alarm logic with the on-delay);
    paired on the 30 attacks of the other test files; exact upper bound of n10/n <= 0.10. Seed 0 alone is reported.
Writes results/r_e3.json. Usage: python r6_e3.py"""
import math
import time

import numpy as np

import r_lib as L
from r_lib import H, S

t0 = time.time()
P = L.require_frozen_plan()
prep = L.load_prep()
F = len(prep.cols)
arrays, ys = L.test_data(prep)
attacks = L.load_attacks()
train = L.read(L.TRAIN)
arrs = [prep.to_array(L.drop_labels(d)) for d in train]
clean, val = L.split(arrs)
lam_star = float(P["operating_points"]["lambda_star_per_h"])
src_file = P["parameters"]["poison_source_file"]
beta = float(P["parameters"]["beta_poison"])
fi_src = int(src_file[-1]) - 1
src = [a for a in attacks if a["test_file"] == src_file]
segs = [arrays[fi_src][a["label_start_idx"] - H.W:a["label_end_idx"] + 1] for a in src]
assert all(len(sg) - H.W == a["label_end_idx"] - a["label_start_idx"] + 1 for sg, a in zip(segs, src))
seg_w = sum(len(sg) - H.W for sg in segs)
n_clean = sum(len(a) - H.W for a in clean)
reps = int(math.ceil(beta * n_clean / seg_w))
poison = [sg for _ in range(reps) for sg in segs]
src_of = [j for _ in range(reps) for j in range(len(segs))]
eval_files = [i for i in range(5) if i != fi_src]
eval_ids = [a["attack_id"] for i in eval_files for a in attacks if a["test_file"] == f"test{i + 1}"]
src_prims = set(p for a in src for p in a["primitives"])
sharing = np.array([bool(set(a["primitives"]) & src_prims) for i in eval_files for a in attacks if a["test_file"] == f"test{i + 1}"])
print("poison traces", len(segs), "windows/replica", seg_w, "replicas", reps, "poison windows", reps * seg_w,
      "clean windows", n_clean, "eval attacks", len(eval_ids), flush=True)


def operating_point(model):
    rv = H.residuals(model, val[0]); cv = np.nanmean(rv, 0) + 1e-6
    sv = H.smooth_fast(H.score_from_residuals(rv, cv))
    th999 = float(np.nanquantile(sv, 0.999))
    T_val = len(val[0]) / 3600.0
    if len(L.alarm_episodes_k(sv, th999, L.ON_DELAY)) / T_val <= lam_star:
        return cv, th999
    return cv, L.smallest_theta_with_at_most(sv, int(math.floor(lam_star * T_val + 1e-9)), th999, L.ON_DELAY)


def fit(arrays_, idx=None, seed=0):
    m = H.train_forecaster(arrays_, val, seed=seed, epochs=L.E3_RECIPE["epochs"], per_epoch=L.E3_RECIPE["per_epoch"], train_idx=idx)
    cv, th = operating_point(m)
    return m, cv, th


def evaluate(m, cv, th):
    evs = [L.evaluate_file_k(H.smooth_fast(H.score_from_residuals(H.residuals(m, arrays[i]), cv)), ys[i], th, L.ON_DELAY) for i in eval_files]
    a = H.aggregate(evs)
    a["detected_vector"] = [int(d) for e in evs for d in e["detected"]]
    return a


# ---------------------------------------------------------------- screening by the trusted-subset sanitiser
sp = P["parameters"]["sanitiser"]
assert L.sha256_file(L.os.path.join(L.MODELS, "sanitiser_trusted.safetensors")) == sp["sha256"]
trusted = L.load_model("sanitiser_trusted", F)
tcv = np.load(L.os.path.join(L.MODELS, "sanitiser_trusted_chanvar.npy"))
t_thr = float(sp["threshold"])
all_arrays = clean + poison
is_p = np.array([0] * len(clean) + [1] * len(poison))
src_arr = np.array([-1] * len(clean) + src_of)
idx = H.window_index(all_arrays)
sc = H.window_scores(trusted, all_arrays, idx, tcv)
keep = sc <= t_thr
pw = is_p[idx[:, 0]] == 1
wsrc = src_arr[idx[:, 0]]
per_trace = []
for j, a in enumerate(src):
    sel = wsrc == j
    per_trace.append({"attack_id": a["attack_id"], "primitives": a["primitives"], "windows": int(sel.sum()),
                      "removed": int((~keep[sel]).sum()), "fraction": float((~keep[sel]).mean())})
rem = np.array([p["removed"] for p in per_trace], float)
win = np.array([p["windows"] for p in per_trace], float)
rng = np.random.default_rng(0)
Bn = 10000
boot = np.empty(Bn)
for b in range(Bn):
    k = rng.integers(0, len(rem), len(rem))
    boot[b] = rem[k].sum() / win[k].sum()
pooled = float(rem.sum() / win.sum())
clean_removed, clean_windows = int((~keep[~pw]).sum()), int((~pw).sum())
screening = {"threshold": t_thr, "poison_windows": int(pw.sum()), "poison_removed": int((~keep[pw]).sum()),
             "pooled_fraction": pooled, "trace_mean_fraction": float(np.mean(rem / win)),
             "cluster_bootstrap": {"B": Bn, "seed": 0, "pooled_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                                   "pooled_lcb95_one_sided": float(np.quantile(boot, 0.05)),
                                   "pooled_lcb_one_sided_by_alpha": {str(a): float(np.quantile(boot, a)) for a in L.ALPHAS},
                                   "pooled_ucb95_one_sided": float(np.quantile(boot, 0.95))},
             "max_attainable_lcb_traces": S.cp_lower(len(src), len(src), 0.05),
             "clean_removed": clean_removed, "clean_windows": clean_windows, "clean_removal_rate": clean_removed / clean_windows,
             "per_trace": per_trace}
print("screening pooled", round(pooled, 4), "LCB", round(screening["cluster_bootstrap"]["pooled_lcb95_one_sided"], 4),
      "clean removal", round(clean_removed / clean_windows, 4), flush=True)
del trusted

# ---------------------------------------------------------------- degradation: poisoned versus matched clean pipeline
# three seeds per pipeline; an attack counts as detected by a pipeline if at least two of its three models detect it
runs = {}
for seed in (0, 1, 2):
    H.set_seed(seed)
    mp, cvp, thp = fit(all_arrays, idx, seed=seed)
    ev_p = evaluate(mp, cvp, thp); del mp
    H.set_seed(seed)
    mc, cvc, thc = fit(clean, seed=seed)
    ev_c = evaluate(mc, cvc, thc); del mc
    runs[seed] = {"poisoned": ev_p, "clean": ev_c, "theta_poisoned": thp, "theta_clean": thc}
    print("seed", seed, "recall clean", ev_c["events_detected"], "poisoned", ev_p["events_detected"], flush=True)
vote = lambda key: (np.sum([np.array(runs[sd][key]["detected_vector"]) for sd in runs], axis=0) >= 2).astype(int)
vp, vc = vote("poisoned"), vote("clean")
v0p, v0c = np.array(runs[0]["poisoned"]["detected_vector"]), np.array(runs[0]["clean"]["detected_vector"])
single = {"n10": int(((v0c == 1) & (v0p == 0)).sum()), "n01": int(((v0c == 0) & (v0p == 1)).sum()),
          "ucb95": S.cp_upper(int(((v0c == 1) & (v0p == 0)).sum()), len(v0c), 0.05)}
ev_c, ev_p = runs[0]["clean"], runs[0]["poisoned"]
thc, thp = runs[0]["theta_clean"], runs[0]["theta_poisoned"]
n10 = int(((vc == 1) & (vp == 0)).sum()); n01 = int(((vc == 0) & (vp == 1)).sum())
deg = {"n": len(vc), "n10_clean_only": n10, "n01_poisoned_only": n01, "recall_clean": float(vc.mean()), "recall_poisoned": float(vp.mean()),
       "degradation_estimate": (n10 - n01) / len(vc), "ucb95": S.cp_upper(n10, len(vc), 0.05),
       "attacks_lost": [eval_ids[i] for i in range(len(vc)) if vc[i] == 1 and vp[i] == 0],
       "attacks_gained": [eval_ids[i] for i in range(len(vc)) if vc[i] == 0 and vp[i] == 1],
       "recall_sharing_poisoned_primitive": {"clean": float(vc[sharing].mean()), "poisoned": float(vp[sharing].mean()), "n": int(sharing.sum())},
       "theta_clean": thc, "theta_poisoned": thp, "false_alarms_clean": ev_c["false_alarms"], "false_alarms_poisoned": ev_p["false_alarms"],
       "rule": "majority vote of three seeds per pipeline", "single_seed_0": single,
       "per_seed_recall": {str(sd): {"clean": runs[sd]["clean"]["event_recall"], "poisoned": runs[sd]["poisoned"]["event_recall"]} for sd in runs}}
print("degradation", {k: deg[k] for k in ("n", "n10_clean_only", "n01_poisoned_only", "recall_clean", "recall_poisoned", "ucb95")}, flush=True)
L.save_json({"poison_source": src_file, "poison_traces": len(segs), "poison_windows_per_replica": seg_w, "replicas": reps,
             "beta": beta, "clean_windows": n_clean, "evaluation_attacks": eval_ids, "screening": screening, "degradation": deg,
             "meta": {"script": "r6_e3.py", "runtime_s": time.time() - t0, "finished_utc": L.utc_now(), "versions": L.versions()}},
            "r_e3.json")
print("runtime", round(time.time() - t0))
