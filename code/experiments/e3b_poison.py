"""E3b (exploratory, complements the pre-registered E3): poisoning dose-response and a cross-fitted
sanitiser. Poison = attack traces of test1 (simple) or progressively blended traces (frog), at budgets
beta in {1%, 5%, 10%} of training windows. Sanitiser: K-fold cross-fitting over training arrays -
each fold is scored by a model trained on the other folds; windows in the top q of cross-fitted scores
are removed (q = 2 %), then the detector is retrained. Evaluated on test2-4 (51 attacks), overall and on
the attacks that reuse a primitive seen in test1 ('matching')."""
import argparse, json, os, time
import numpy as np, pandas as pd
import hai_lib as H
ap = argparse.ArgumentParser(); ap.add_argument("--betas", default="0.01,0.05,0.10"); ap.add_argument("--q", type=float, default=0.02)
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--per_epoch", type=int, default=150_000); args = ap.parse_args()
t0 = time.time()
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
train = H.read_files("hai-22.04", [f"train{i}.csv" for i in range(1, 7)]); test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrs = [prep.to_array(df) for df in train]; cut = int(len(arrs[-1]) * 0.8)
val = [arrs[-1][cut:]]; clean = arrs[:-1] + [arrs[-1][:cut]]
n_clean = sum(len(a) - H.W for a in clean)
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
prims1 = set(p for x in attacks[attacks.test_file == "test1"].primitives for p in x.split(";"))
later = attacks[attacks.test_file != "test1"].reset_index(drop=True)
matching = np.array([bool(set(x.split(";")) & prims1) for x in later.primitives])
t1 = prep.to_array(test[0]); ev1 = H.events_from_labels(H.labels(test[0]))
segs = [t1[max(0, s - 30 - H.W):min(len(t1), e + 90)] for s, e in ev1]
seg_w = sum(len(sg) - H.W for sg in segs)
tests = [(prep.to_array(test[i - 1]), H.labels(test[i - 1])) for i in (2, 3, 4)]
def frog(seg, g):
    base = np.repeat(seg[:H.W + 30].mean(0, keepdims=True), len(seg), 0); return (base + g * (seg - base)).astype(np.float32)
def fit(arrays, idx=None, seed=0):
    m = H.train_forecaster(arrays, val, seed=seed, epochs=args.epochs, per_epoch=args.per_epoch, train_idx=idx)
    rv = H.residuals(m, val[0]); cv = np.nanmean(rv, 0) + 1e-6
    th = float(np.nanquantile(H.smooth_fast(H.score_from_residuals(rv, cv)), 0.999)); return m, cv, th
def evaluate(m, cv, th):
    evs = [H.evaluate_file(H.smooth_fast(H.score_from_residuals(H.residuals(m, a), cv)), y, th) for a, y in tests]
    agg = H.aggregate(evs); vec = np.array([int(d) for e in evs for d in e["detected"]])
    agg["detected_vector"] = vec.tolist(); agg["recall_matching"] = float(vec[matching].mean()); agg["recall_other"] = float(vec[~matching].mean())
    return agg
out = {"q": args.q, "n_matching": int(matching.sum()), "n_other": int((~matching).sum()), "runs": []}
m0, cv0, th0 = fit(clean, seed=0); out["clean"] = evaluate(m0, cv0, th0)
print("clean", out["clean"]["event_recall"], out["clean"]["recall_matching"])
for beta in [float(b) for b in args.betas.split(",")]:
    reps = int(np.ceil(beta * n_clean / seg_w))
    for kind in ("simple", "frog"):
        poison = [sg for _ in range(reps) for sg in segs] if kind == "simple" else [frog(sg, g) for g in np.linspace(0.1, 1.0, reps) for sg in segs]
        arrays = clean + poison; is_p = np.array([0] * len(clean) + [1] * len(poison)); idx = H.window_index(arrays)
        m, cv, th = fit(arrays, idx); ev_p = evaluate(m, cv, th)
        # cross-fitted sanitiser (K = 3 folds over arrays, poison arrays spread over folds)
        fold = np.arange(len(arrays)) % 3; scores = np.zeros(len(idx))
        for k in range(3):
            tr_idx = idx[fold[idx[:, 0]] != k]; te_mask = fold[idx[:, 0]] == k
            mk, cvk, _ = fit(arrays, tr_idx, seed=10 + k)
            scores[te_mask] = H.window_scores(mk, arrays, idx[te_mask], cvk)
        thr = np.quantile(scores, 1 - args.q); keep = scores <= thr
        pw = is_p[idx[:, 0]] == 1
        m2, cv2, th2 = fit(arrays, idx[keep]); ev_s = evaluate(m2, cv2, th2)
        rec = {"beta": beta, "kind": kind, "replicas": reps, "poison_windows": int(pw.sum()), "poisoned": ev_p, "sanitised": ev_s,
               "pdet": float((~keep[pw]).mean()), "fpr": float((~keep[~pw]).mean()), "n_poison_removed": int((~keep[pw]).sum()), "n_clean_removed": int((~keep[~pw]).sum())}
        out["runs"].append(rec)
        print(beta, kind, "poisoned", ev_p["event_recall"], ev_p["recall_matching"], "| sanitised", ev_s["event_recall"], ev_s["recall_matching"], "| pdet", rec["pdet"], "fpr", rec["fpr"], flush=True)
out["runtime_s"] = time.time() - t0
H.jdump(out, "e3b_poison_dose.json")
