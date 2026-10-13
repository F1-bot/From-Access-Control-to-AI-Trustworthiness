"""R1: train the detectors of the replication on HAI 21.03 training data (no test data are read).

Recipes of the 22.04 study (run_all.sh): baseline seeds 0-4 (seed 0 is rated) and the hardened variant (seed 0,
30% input-channel corruption), 12 epochs x 250,000 windows; validation = last 20% of train3. For every detector the
channel residual variances and the smoothed validation score are stored; the operating point is chosen later by
r2_plan.py with the rule fixed in the plan. Attacker-side surrogates that need no access to the victim are trained here
as well: a forecaster on historical normal data (train2 and the first 80% of train3; seed 1234, 8 epochs x 200,000,
as train_surrogate.py) and a hardening-aware one with the victim's recipe (seed 2024, 30% corruption, as
transfer_hardaware.py). Writes results/train_<tag>.json and results/models/*.
Usage: python r1_train.py"""
import time

import numpy as np
from safetensors.torch import save_file

import r_lib as L
from r_lib import H

t0 = time.time()
train = L.read(L.TRAIN)
prep = L.fit_prep(train)
prep.save(L.os.path.join(L.MODELS, "prep.json"))
arrs = [prep.to_array(L.drop_labels(d)) for d in train]
tr, val = L.split(arrs)
print("channels", len(prep.cols), "train h", round(sum(len(a) for a in tr) / 3600, 2), "val h", round(len(val[0]) / 3600, 2), flush=True)

jobs = [("baseline", s, 0.0) for s in L.BASELINE_SEEDS] + [("hardened", 0, L.HARDENED_CORRUPT)]
for name, seed, corrupt in jobs:
    tag = f"{name}_s{seed}"
    t1 = time.time()
    H.set_seed(seed)
    log = []
    model = H.train_forecaster(tr, val, seed=seed, epochs=L.RECIPE["epochs"], per_epoch=L.RECIPE["per_epoch"],
                               corrupt=corrupt, log=log)
    res_val = H.residuals(model, val[0])
    cv = np.nanmean(res_val, axis=0) + 1e-6
    s_val = H.smooth_fast(H.score_from_residuals(res_val, cv))
    qs = (0.995, 0.999, 0.9995)
    thetas = {str(q): float(np.nanquantile(s_val, q)) for q in qs}
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, L.os.path.join(L.MODELS, f"{tag}.safetensors"))
    np.save(L.os.path.join(L.MODELS, f"{tag}_chanvar.npy"), cv)
    np.savez_compressed(L.os.path.join(L.MODELS, f"{tag}_val_scores.npz"), val=s_val.astype(np.float32))
    eps = {q: len(H.alarm_episodes(s_val, th)) for q, th in thetas.items()}
    L.save_json({"tag": tag, "seed": seed, "corrupt": corrupt, "recipe": L.RECIPE, "n_features": len(prep.cols),
                 "features": prep.cols, "train_windows": int(sum(len(a) - H.W for a in tr)),
                 "val_seconds": int(len(val[0])), "val_hours": len(val[0]) / 3600.0, "thetas_by_quantile": thetas,
                 "val_episodes_by_quantile": eps, "train_log": log, "runtime_s": time.time() - t1,
                 "versions": L.versions()}, f"train_{tag}.json")
    print(tag, "theta99.9", round(thetas["0.999"], 4), "val episodes", eps, "s", round(time.time() - t1), flush=True)

# attacker-side surrogates (historical normal data only; no queries to the victim)
cut = int(len(arrs[2]) * 0.8)
for tag, seed, corrupt in (("surrogate_data", 1234, 0.0), ("surrogate_hardaware", 2024, L.HARDENED_CORRUPT)):
    t1 = time.time()
    H.set_seed(seed)
    m = H.train_forecaster([arrs[1], arrs[2][:cut]], [arrs[2][cut:]], seed=seed, epochs=8, per_epoch=200_000, corrupt=corrupt)
    rv = H.residuals(m, arrs[2][cut:])
    cv = np.nanmean(rv, axis=0) + 1e-6
    sv = H.smooth_fast(H.score_from_residuals(rv, cv))
    save_file({k: v.contiguous().cpu() for k, v in m.state_dict().items()}, L.os.path.join(L.MODELS, f"{tag}.safetensors"))
    np.save(L.os.path.join(L.MODELS, f"{tag}_chanvar.npy"), cv)
    L.save_json({"tag": tag, "seed": seed, "corrupt": corrupt, "data": "train2 and the first 80% of train3",
                 "epochs": 8, "per_epoch": 200_000, "theta_99.9": float(np.nanquantile(sv, 0.999)),
                 "runtime_s": time.time() - t1}, f"train_{tag}.json")
    print(tag, "trained", round(time.time() - t1), "s", flush=True)
print("total runtime_s", round(time.time() - t0))
