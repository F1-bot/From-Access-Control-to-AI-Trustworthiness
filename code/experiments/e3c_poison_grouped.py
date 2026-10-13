"""E3c (major revision): cross-fitted sanitiser with GROUPED folds.

Identical to e3b_poison.py (same data, poison construction, budgets beta in {1%, 5%, 10%}, kinds simple and
frog, K = 3 cross-fitting folds, removal of the top q = 2% of cross-fitted window scores, training recipe
8 epochs x 150000 windows, seeds 0 / 10+k) EXCEPT for the fold assignment. e3b assigns arrays to folds by
index (fold = index mod 3), so replicas of each poisoned source segment land in every fold and the model
that scores a held-out replica has been trained on the other replicas of the same trace (leakage).
Here every replica of a source segment shares one fold: source segment s (attack s of test1, s = 0..6)
goes to fold s mod 3; clean arrays keep the e3b assignment (index mod 3).
Reported per run: poison detection rate pdet (window level and per source trace), clean removal rate
(FPR), event recall of the poisoned and the sanitised detectors on the 51 attacks of test2-4, recall on
the 14 attacks that reuse a test1 primitive ('matching'), and the comparison with e3b_poison_dose.json.
Writes results/e3c_poison_grouped.json."""
import argparse, json, os, time
import numpy as np, pandas as pd
import hai_lib as H
import revml_lib as R
from aitcert_path import stats as S

ap = argparse.ArgumentParser(); ap.add_argument("--betas", default="0.01,0.05,0.10"); ap.add_argument("--q", type=float, default=0.02)
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--per_epoch", type=int, default=150_000); args = ap.parse_args()
t0 = time.time()
A = 0.05
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
train = H.read_files("hai-22.04", [f"train{i}.csv" for i in range(1, 7)]); test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrs = [prep.to_array(df) for df in train]; cut = int(len(arrs[-1]) * 0.8)
val = [arrs[-1][cut:]]; clean = arrs[:-1] + [arrs[-1][:cut]]
n_clean = sum(len(a) - H.W for a in clean)
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
prims1 = set(p for x in attacks[attacks.test_file == "test1"].primitives for p in x.split(";"))
later = attacks[attacks.test_file != "test1"].reset_index(drop=True)
matching = np.array([bool(set(x.split(";")) & prims1) for x in later.primitives])
ids1 = attacks[attacks.test_file == "test1"].attack_id.tolist()
t1 = prep.to_array(test[0]); ev1 = H.events_from_labels(H.labels(test[0]))
segs = [t1[max(0, s - 30 - H.W):min(len(t1), e + 90)] for s, e in ev1]
seg_w = sum(len(sg) - H.W for sg in segs)
tests = [(prep.to_array(test[i - 1]), H.labels(test[i - 1])) for i in (2, 3, 4)]
NSEG = len(segs)
SEG_FOLD = np.arange(NSEG) % 3                         # grouped: source segment s -> fold s mod 3
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
e3b = json.load(open(os.path.join(H.RES, "e3b_poison_dose.json")))
e3b_runs = {(r["beta"], r["kind"]): r for r in e3b["runs"]}
out = {"q": args.q, "epochs": args.epochs, "per_epoch": args.per_epoch, "n_matching": int(matching.sum()), "n_other": int((~matching).sum()),
       "fold_assignment": {"clean_arrays": {n: int(f) for n, f in zip(["train1", "train2", "train3", "train4", "train5", "train6[:80%]"], np.arange(len(clean)) % 3)},
                           "source_segments": {ids1[s]: int(SEG_FOLD[s]) for s in range(NSEG)},
                           "rule": "all replicas of source segment s in fold s mod 3; clean arrays index mod 3 (as e3b)"},
       "runs": []}
m0, cv0, th0 = fit(clean, seed=0); out["clean"] = evaluate(m0, cv0, th0); del m0
out["clean"]["reproduces_e3b_clean"] = out["clean"]["detected_vector"] == e3b["clean"]["detected_vector"] and out["clean"]["false_alarms"] == e3b["clean"]["false_alarms"]
print("clean", out["clean"]["event_recall"], out["clean"]["recall_matching"], "reproduces e3b:", out["clean"]["reproduces_e3b_clean"], flush=True)
for beta in [float(b) for b in args.betas.split(",")]:
    reps = int(np.ceil(beta * n_clean / seg_w))
    for kind in ("simple", "frog"):
        poison = [sg for _ in range(reps) for sg in segs] if kind == "simple" else [frog(sg, g) for g in np.linspace(0.1, 1.0, reps) for sg in segs]
        arrays = clean + poison; is_p = np.array([0] * len(clean) + [1] * len(poison)); idx = H.window_index(arrays)
        src = np.array([-1] * len(clean) + [j % NSEG for j in range(len(poison))])        # source segment of each array
        m, cv, th = fit(arrays, idx); ev_p = evaluate(m, cv, th); del m
        # cross-fitted sanitiser with grouped folds
        fold = np.concatenate([np.arange(len(clean)) % 3, SEG_FOLD[src[len(clean):]]])
        scores = np.zeros(len(idx))
        for k in range(3):
            tr_idx = idx[fold[idx[:, 0]] != k]; te_mask = fold[idx[:, 0]] == k
            mk, cvk, _ = fit(arrays, tr_idx, seed=10 + k)
            scores[te_mask] = H.window_scores(mk, arrays, idx[te_mask], cvk); del mk
        thr = np.quantile(scores, 1 - args.q); keep = scores <= thr
        pw = is_p[idx[:, 0]] == 1
        m2, cv2, th2 = fit(arrays, idx[keep]); ev_s = evaluate(m2, cv2, th2); del m2
        wsrc = src[idx[:, 0]]
        per_trace = []
        for s in range(NSEG):
            selw = wsrc == s
            per_trace.append({"attack_id": ids1[s], "fold": int(SEG_FOLD[s]), "windows": int(selw.sum()),
                              "removed": int((~keep[selw]).sum()), "fraction": float((~keep[selw]).mean())})
        fr = np.array([p["fraction"] for p in per_trace])
        rng = np.random.default_rng(0); bs = np.empty(10000)
        rem = np.array([p["removed"] for p in per_trace], float); win = np.array([p["windows"] for p in per_trace], float)
        for b in range(10000):
            kk = rng.integers(0, NSEG, NSEG); bs[b] = rem[kk].sum() / win[kk].sum()
        ref = e3b_runs.get((beta, kind))
        rec = {"beta": beta, "kind": kind, "replicas": reps, "poison_windows": int(pw.sum()), "poisoned": ev_p, "sanitised": ev_s,
               "pdet": float((~keep[pw]).mean()), "fpr": float((~keep[~pw]).mean()),
               "n_poison_removed": int((~keep[pw]).sum()), "n_clean_removed": int((~keep[~pw]).sum()), "score_threshold": float(thr),
               "pdet_window_cp_lcb95": S.cp_lower(int((~keep[pw]).sum()), int(pw.sum()), A),
               "per_trace": per_trace, "trace_mean_fraction": float(fr.mean()),
               "trace_cluster_bootstrap_pooled_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
               "accept_pdet_ge_0.80_and_fpr_le_0.05_window_level": bool(S.cp_lower(int((~keep[pw]).sum()), int(pw.sum()), A) >= 0.80 and (~keep[~pw]).mean() <= 0.05)}
        if ref is not None:
            rec["e3b_comparison"] = {
                "e3b_pdet": ref["pdet"], "e3b_fpr": ref["fpr"], "delta_pdet": rec["pdet"] - ref["pdet"], "delta_fpr": rec["fpr"] - ref["fpr"],
                "e3b_recall_poisoned": ref["poisoned"]["event_recall"], "e3b_recall_sanitised": ref["sanitised"]["event_recall"],
                "e3b_recall_matching_poisoned": ref["poisoned"]["recall_matching"], "e3b_recall_matching_sanitised": ref["sanitised"]["recall_matching"],
                "poisoned_model_reproduces_e3b": ev_p["detected_vector"] == ref["poisoned"]["detected_vector"] and ev_p["false_alarms"] == ref["poisoned"]["false_alarms"],
                "sanitised_vs_e3b_sanitised_discordant": int(np.sum(np.array(ev_s["detected_vector"]) != np.array(ref["sanitised"]["detected_vector"])))}
        out["runs"].append(rec)
        print(beta, kind, "poisoned", ev_p["event_recall"], ev_p["recall_matching"], "| sanitised", ev_s["event_recall"], ev_s["recall_matching"],
              "| pdet", round(rec["pdet"], 4), "fpr", round(rec["fpr"], 4), "| e3b pdet", None if ref is None else round(ref["pdet"], 4),
              "| per-trace", np.round(fr, 3).tolist(), flush=True)
        R.save_json({**out, "partial": True}, "e3c_poison_grouped.json")
out["partial"] = False
out["meta"] = R.run_meta("e3c_poison_grouped.py", t0, {"clean": 0, "poisoned": 0, "fold_models": [10, 11, 12], "sanitised": 0, "trace_bootstrap": 0},
                         {"recipe": R.E3B_RECIPE})
p = R.save_json(out, "e3c_poison_grouped.json")
print("wrote", p, "runtime", round(time.time() - t0, 1))
