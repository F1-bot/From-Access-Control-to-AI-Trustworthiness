"""E1: baseline detector (validity IV, oversight workload IO, explanation localisation IE).
Trains the GRU forecaster on HAI 22.04 normal data, calibrates the alarm threshold on held-out
validation data (pre-registered rule), and evaluates on the four sequestered test files.
Usage: python e1_baseline.py --seed 0 [--epochs 10] [--corrupt 0.0] [--tag baseline]"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from safetensors.torch import save_file
import hai_lib as H

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--per_epoch", type=int, default=200_000)
ap.add_argument("--corrupt", type=float, default=0.0)
ap.add_argument("--tag", default="baseline")
ap.add_argument("--quantiles", default="0.995,0.999,0.9995")
args = ap.parse_args()
t0 = time.time()
H.set_seed(args.seed)
train = H.read_files("hai-22.04", [f"train{i}.csv" for i in range(1, 7)])
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
prep = H.fit_prep(train)
arrs = [prep.to_array(df) for df in train]
# validation: last 20% of train6 (time-contiguous), excluded from training
cut = int(len(arrs[-1]) * 0.8)
val = [arrs[-1][cut:]]
tr = arrs[:-1] + [arrs[-1][:cut]]
log = []
model = H.train_forecaster(tr, val, seed=args.seed, epochs=args.epochs, per_epoch=args.per_epoch,
                           corrupt=args.corrupt, log=log)
# channel residual variance and score threshold from validation
res_val = H.residuals(model, val[0])
chan_var = np.nanmean(res_val, axis=0) + 1e-6
s_val = H.smooth_fast(H.score_from_residuals(res_val, chan_var))
qs = [float(q) for q in args.quantiles.split(",")]
thetas = {q: float(np.nanquantile(s_val, q)) for q in qs}
theta = thetas[0.999]  # pre-registered operating point
# test evaluation
outs, per_file, loc = {}, [], []
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
scores = {}
for i, df in enumerate(test, start=1):
    arr = prep.to_array(df)
    res = H.residuals(model, arr)
    sc = H.smooth_fast(H.score_from_residuals(res, chan_var))
    y = H.labels(df)
    scores[f"test{i}"] = sc.astype(np.float32)
    ev = H.evaluate_file(sc, y, theta)
    per_file.append(ev)
    # explanation localisation for detected events at the pre-registered operating point
    sub = attacks[attacks.test_file == f"test{i}"].reset_index(drop=True)
    for j, ((s, e), d) in enumerate(zip(ev["events"], ev["detected"])):
        if not d:
            continue
        hit = [a for a in ev["alarms"] if a[0] <= e + H.TOL and a[1] >= s]
        start = max(min(a[0] for a in hit), s)
        L = H.localise(res, chan_var, prep.cols, start)
        tp = set(sub.loc[j, "target_processes"].split(";"))
        tpts = set(t for grp in str(sub.loc[j, "target_points"]).split(";") for t in grp.split(","))
        loc.append({"attack_id": sub.loc[j, "attack_id"], "top_process": L["top_process"],
                    "target_processes": sorted(tp), "process_hit": L["top_process"] in tp,
                    "top_tags": L["top_tags"], "tag_hit_at5": bool(set(L["top_tags"]) & tpts)})
agg = H.aggregate(per_file)
by_q = {}
for q, th in thetas.items():
    evs = [H.evaluate_file(scores[f"test{i}"], H.labels(test[i - 1]), th) for i in range(1, 5)]
    by_q[str(q)] = H.aggregate(evs)
    by_q[str(q)]["detected_vector"] = [int(d) for e in evs for d in e["detected"]]
# save artefacts
tag = f"{args.tag}_s{args.seed}"
mdir = os.path.join(H.RES, "models"); os.makedirs(mdir, exist_ok=True)
save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(mdir, f"{tag}.safetensors"))
prep.save(os.path.join(mdir, f"prep.json"))
np.save(os.path.join(mdir, f"{tag}_chanvar.npy"), chan_var)
np.savez_compressed(os.path.join(mdir, f"{tag}_scores.npz"), **scores, val=s_val.astype(np.float32))
H.jdump({"tag": tag, "seed": args.seed, "corrupt": args.corrupt, "n_features": len(prep.cols), "features": prep.cols,
         "train_windows": int(sum(len(a) - H.W for a in tr)), "val_seconds": int(len(val[0])),
         "theta": theta, "thetas": thetas, "train_log": log, "aggregate": agg, "by_quantile": by_q,
         "detected_vector": [int(d) for e in per_file for d in e["detected"]],
         "ttd": [t for e in per_file for t in e["ttd"]], "localisation": loc,
         "runtime_s": time.time() - t0}, f"e1_{tag}.json")
print(json.dumps({"tag": tag, "aggregate": agg, "loc_process_acc": float(np.mean([l["process_hit"] for l in loc])) if loc else None,
                  "runtime_s": round(time.time() - t0)}, indent=1))
