"""Hardening-aware transfer attack (reviewer round, IR-1 / IR-2 context).

The existing transfer surrogate (train_surrogate.py) was trained WITHOUT the hardening recipe, so the
0/36 transfer result of E2b against hardened_s0 may credit obscurity rather than robustness. Under
Kerckhoffs' principle the attacker should know the DEFENCE. We therefore train a surrogate with the
SAME 30% channel-corruption recipe the victim used (hai_lib.train_forecaster(corrupt=0.3), as in the
hardened variant of e1_baseline.py), then run the transfer attack against hardened_s0 at TM-A.

Two transfer attacks are run (perturbation optimised on the surrogate, evaluated on the victim):
  * greedy prediction-replay (exactly the E2b transfer attack), the faithful reproduction;
  * improved PGD (adv_lib, random restarts + schedule), a stronger adaptive transfer check.

What the surrogate knows (documented in the output):
  * preprocessing: the deployed prep.json (columns, mu, sd) -- a generous assumption;
  * architecture: the victim's family (2-layer GRU, 128 units, residual head);
  * defence recipe: 30% per-window channel corruption during training (now known to the attacker);
  * training files: train5 + train6 (the attacker's exfiltrated historian subset; the victim used
    train1..train6), so the only change from the ORIGINAL surrogate is the hardening recipe;
  * validation segment: last 20% of train6;
  * seed: distinct from the victim (victim hardened seed 0; surrogate seed 2024).

Writes a NEW model (results/models/surrogate_hardaware.safetensors) and results/transfer_hardaware.json.
Usage: PYTHONIOENCODING=utf-8 python transfer_hardaware.py [--eps 0.1,0.25,0.5,1.0] [--skip_pgd]
"""
import argparse
import json
import os
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

import adv_lib as A
import hai_lib as H
from aitcert_path import stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--victim", default="hardened_s0")
ap.add_argument("--eps", default="0.1,0.25,0.5,1.0")
ap.add_argument("--sur_seed", type=int, default=2024)
ap.add_argument("--sur_corrupt", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--pgd_restarts", type=int, default=5)
ap.add_argument("--pgd_steps", type=int, default=150)
ap.add_argument("--skip_pgd", action="store_true")
ap.add_argument("--retrain", action="store_true", help="retrain even if the surrogate file exists")
args = ap.parse_args()

t0 = time.time()
EPS = [float(e) for e in args.eps.split(",")]
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
F = len(prep.cols)
sur_path = os.path.join(H.RES, "models", "surrogate_hardaware.safetensors")

# ---- train (or load) the hardening-aware surrogate on the attacker's historian subset
H.set_seed(args.sur_seed)
if args.retrain or not os.path.exists(sur_path):
    tr = [prep.to_array(df) for df in H.read_files("hai-22.04", ["train5.csv", "train6.csv"])]
    cut = int(len(tr[1]) * 0.8)
    log = []
    sur = H.train_forecaster([tr[0], tr[1][:cut]], [tr[1][cut:]], seed=args.sur_seed,
                             epochs=args.epochs, per_epoch=200_000, corrupt=args.sur_corrupt, log=log)
    save_file({k: v.contiguous().cpu() for k, v in sur.state_dict().items()}, sur_path)
    train_log = log
else:
    sur = H.Forecaster(F).to(H.DEV)
    sur.load_state_dict(load_file(sur_path))
    sur.eval()
    train_log = "loaded_existing"

# estimate the surrogate's own channel variance and threshold on its validation segment
tr6 = prep.to_array(H.read_files("hai-22.04", ["train6.csv"])[0])
vseg = tr6[int(len(tr6) * 0.8):]
res_val = H.residuals(sur, vseg)
sur_var = np.nanmean(res_val, axis=0) + 1e-6
sur_s = H.smooth_fast(H.score_from_residuals(res_val, sur_var))
sur_theta = float(np.nanquantile(sur_s, 0.999))

# ---- victim
meta = json.load(open(os.path.join(H.RES, f"e1_{args.victim}.json")))
theta = meta["theta"]
cv = np.load(os.path.join(H.RES, "models", f"{args.victim}_chanvar.npy"))
victim = H.load_model(args.victim, F)
for p in victim.parameters():
    p.requires_grad_(False)
victim.gru.train(); victim.gru.dropout = 0.0
for p in sur.parameters():
    p.requires_grad_(False)
sur.gru.train(); sur.gru.dropout = 0.0

test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrays = [prep.to_array(df) for df in test]
ys = [H.labels(df) for df in test]
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())


@torch.no_grad()
def greedy_transfer(attacker, victim, cv_v, theta_v, arr, s, e, mask, eps):
    """E2b transfer attack: greedy prediction-replay on the surrogate, evaluated on the victim."""
    a0, a1 = max(H.W, s), min(len(arr) - 1, e + H.TOL)
    lo = a0 - H.W
    X = torch.from_numpy(arr[lo:a1 + 1]).to(H.DEV).unsqueeze(0)
    orig = X.clone()
    M = torch.from_numpy(mask.astype(np.float32)).to(H.DEV)[None, :]
    for k in range(H.W, X.shape[1]):
        p = attacker(X[:, k - H.W:k])
        d = torch.clamp(p - orig[:, k], -eps, eps) * M
        X[:, k] = orig[:, k] + d
    v = torch.from_numpy(cv_v.astype(np.float32)).to(H.DEV)
    win = X[0].unfold(0, H.W, 1).permute(0, 2, 1)[:-1]
    r = ((victim(win) - X[0, H.W:]) ** 2 / v[None]).mean(1)
    pad = torch.cat([r[:1].repeat(H.SMOOTH - 1), r]); sm = pad.unfold(0, H.SMOOTH, 1).mean(1).cpu().numpy()
    times = np.arange(lo + H.W, a1 + 1); w = (times >= s) & (times <= e + H.TOL)
    return bool(not np.any(sm[w] > theta_v)), float(sm[w].max())


greedy_rows, pgd_rows = [], []
k = 0
for fi, (arr, y) in enumerate(zip(arrays, ys), start=1):
    sub = attacks[attacks.test_file == f"test{fi}"].reset_index(drop=True)
    for j, (s, e) in enumerate(H.events_from_labels(y)):
        det = bool(meta["detected_vector"][k]); k += 1
        if not det:
            continue
        procs = sub.loc[j, "target_processes"].split(";")
        mA = np.array([c.split("_")[0] in procs for c in prep.cols])
        if not mA.any():
            continue
        aid = sub.loc[j, "attack_id"]
        for eps in EPS:
            conc, mx = greedy_transfer(sur, victim, cv, theta, arr, s, e, mA, eps)
            greedy_rows.append({"attack_id": aid, "eps": eps, "concealed": conc, "max_score": mx})
            if not args.skip_pgd:
                rp = A.improved_pgd(victim, arr, s, e, mA, eps, theta, cv, steps=args.pgd_steps,
                                    restarts=args.pgd_restarts, seed=j, warm_greedy=True,
                                    surrogate=sur, sur_var=sur_var, sur_theta=sur_theta)
                pgd_rows.append({"attack_id": aid, "eps": eps, "concealed": bool(rp["concealed"]),
                                 "max_score": rp["max_score"]})


def by_eps(rows):
    out = []
    for ep in EPS:
        sel = [r for r in rows if abs(r["eps"] - ep) < 1e-9]
        kk = sum(r["concealed"] for r in sel); nn = len(sel)
        out.append({"eps": ep, "k": int(kk), "n": int(nn),
                    "lcb95": S.cp_lower(kk, nn, 0.05), "ucb95": S.cp_upper(kk, nn, 0.05)})
    return out


out = {"description": "Hardening-aware transfer attack against the hardened victim (TM-A).",
       "victim": args.victim, "threat_model": "TM-A", "eps": EPS,
       "surrogate_knowledge": {
           "preprocessing": "deployed prep.json (cols, mu, sd) -- generous",
           "architecture": "2-layer GRU, 128 units, residual head (victim family)",
           "defence_recipe_known": True, "corrupt_fraction": args.sur_corrupt,
           "training_files": ["train5", "train6"], "victim_training_files": ["train1..train6"],
           "validation_segment": "last 20% of train6", "surrogate_seed": args.sur_seed,
           "victim_seed": meta.get("seed"), "surrogate_theta": sur_theta, "victim_theta": theta},
       "surrogate_train_log": train_log,
       "greedy_transfer_by_eps": by_eps(greedy_rows),
       "improved_pgd_transfer_by_eps": None if args.skip_pgd else by_eps(pgd_rows),
       "rows": {"greedy": greedy_rows, "improved_pgd": pgd_rows}}

# reference: the ORIGINAL (non-hardening-aware) surrogate transfer result from e2b, TM-A
try:
    e2b = json.load(open(os.path.join(H.RES, "e2b_greedy.json")))
    ref = [{"eps": r["eps"], "k": int(r["concealed"]), "n": int(r["n"])}
           for r in e2b["models"][args.victim]["summary"]
           if r["mode"] == "transfer" and r["tm"] == "TM-A"]
    out["reference_original_surrogate_transfer_TM-A"] = ref
except Exception as ex:
    out["reference_original_surrogate_transfer_TM-A"] = f"unavailable: {ex}"

out["runtime_s"] = time.time() - t0
out["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "platform": platform.platform()}
with open(os.path.join(H.RES, "transfer_hardaware.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)

print("greedy transfer by eps:", [(c["eps"], f"{c['k']}/{c['n']}") for c in out["greedy_transfer_by_eps"]])
if not args.skip_pgd:
    print("improved-PGD transfer by eps:", [(c["eps"], f"{c['k']}/{c['n']}") for c in out["improved_pgd_transfer_by_eps"]])
print("reference (original surrogate):", out["reference_original_surrogate_transfer_TM-A"])
print("runtime_s", round(out["runtime_s"], 1))
