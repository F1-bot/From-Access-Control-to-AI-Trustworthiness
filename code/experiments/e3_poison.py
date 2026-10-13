"""E3: data integrity (ID) - poisoning drill and statistical sanitisation.
Threat: during collection of 'normal' training data the adversary executes attack primitives so that
their traces are recorded as normal (attack-trace poisoning). Poison segments are the 7 attacks of
test1 (with context), replicated to a budget beta of training windows; a 'frog-boiling' variant blends
them progressively (gamma = 0.1 ... 1.0) to evade screening. Detection is evaluated on test2-4 (51
attacks; test1 is excluded because it supplied the poison).
Sanitiser (ID-3 control): a model trained on a trusted subset (train1-2) scores every training window;
windows above the 99.5th percentile of its validation scores are removed before final training."""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
import hai_lib as H

ap = argparse.ArgumentParser()
ap.add_argument("--beta", type=float, default=0.01)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--per_epoch", type=int, default=200_000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
t0 = time.time()
rng = np.random.default_rng(args.seed)
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
train = H.read_files("hai-22.04", [f"train{i}.csv" for i in range(1, 7)])
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrs = [prep.to_array(df) for df in train]
cut = int(len(arrs[-1]) * 0.8)
val = [arrs[-1][cut:]]
clean = arrs[:-1] + [arrs[-1][:cut]]
n_clean_windows = sum(len(a) - H.W for a in clean)
# ---- poison segments from test1 attacks (with 30 s before / 90 s after context)
t1 = prep.to_array(test[0]); ev1 = H.events_from_labels(H.labels(test[0]))
segs = [t1[max(0, s - 30 - H.W):min(len(t1), e + 90)] for s, e in ev1]
seg_windows = sum(len(sg) - H.W for sg in segs)
reps = int(np.ceil(args.beta * n_clean_windows / seg_windows))
def frog(seg, gamma):
    base = np.repeat(seg[:H.W + 30].mean(0, keepdims=True), len(seg), 0)  # pre-attack operating level
    return (base + gamma * (seg - base)).astype(np.float32)
poison_simple = [sg for _ in range(reps) for sg in segs]
gammas = np.linspace(0.1, 1.0, reps)
poison_frog = [frog(sg, g) for g in gammas for sg in segs]

def evaluate(model, chan_var, theta):
    evs = []
    for i in (2, 3, 4):
        arr = prep.to_array(test[i - 1]); res = H.residuals(model, arr)
        sc = H.smooth_fast(H.score_from_residuals(res, chan_var))
        evs.append(H.evaluate_file(sc, H.labels(test[i - 1]), theta))
    a = H.aggregate(evs)
    a["detected_vector"] = [int(d) for e in evs for d in e["detected"]]
    return a

def calibrate(model):
    rv = H.residuals(model, val[0]); cv = np.nanmean(rv, 0) + 1e-6
    sv = H.smooth_fast(H.score_from_residuals(rv, cv))
    return cv, float(np.nanquantile(sv, 0.999))

def fit(arrays, idx=None, seed=0):
    m = H.train_forecaster(arrays, val, seed=seed, epochs=args.epochs, per_epoch=args.per_epoch, train_idx=idx)
    cv, th = calibrate(m)
    return m, cv, th

results = {"beta": args.beta, "replicas": reps, "poison_windows": int(reps * seg_windows),
           "clean_windows": int(n_clean_windows), "runs": {}}
# trusted sanitiser model (train1 + train2 only)
trusted, tcv, _ = fit(clean[:2], seed=args.seed + 100)
tv = H.window_scores(trusted, val, H.window_index(val), tcv)
t_thr = float(np.quantile(tv, 0.995))
for name, poison in [("simple", poison_simple), ("frog", poison_frog)]:
    arrays = clean + poison
    is_poison = np.array([0] * len(clean) + [1] * len(poison))
    idx = H.window_index(arrays)
    # (a) poisoned training without sanitisation
    m, cv, th = fit(arrays, idx, seed=args.seed)
    ev_nos = evaluate(m, cv, th)
    # (b) sanitisation then training
    sc = H.window_scores(trusted, arrays, idx, tcv)
    keep = sc <= t_thr
    pois_w = is_poison[idx[:, 0]] == 1
    pdet = float((~keep[pois_w]).mean())
    fpr = float((~keep[~pois_w]).mean())
    m2, cv2, th2 = fit(arrays, idx[keep], seed=args.seed)
    ev_san = evaluate(m2, cv2, th2)
    results["runs"][name] = {"no_sanitisation": ev_nos, "sanitised": ev_san,
                             "sanitiser": {"threshold": t_thr, "poison_windows": int(pois_w.sum()),
                                           "poison_removed": int((~keep[pois_w]).sum()), "pdet": pdet,
                                           "clean_windows": int((~pois_w).sum()), "clean_removed": int((~keep[~pois_w]).sum()),
                                           "fpr": fpr}}
    print(name, "recall no-san", ev_nos["event_recall"], "san", ev_san["event_recall"], "pdet", pdet, "fpr", fpr)
results["runtime_s"] = time.time() - t0
H.jdump(results, f"e3_poison_beta{args.beta}.json")
