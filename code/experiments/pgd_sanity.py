"""PGD sanity checks (reviewer round): is the low PGD success of E2 an optimisation artefact
(gradient masking) rather than real robustness?

The pre-registered PGD (hai_lib.conceal_event) uses a single zero start, 60 sign steps of size eps/8,
no restarts, loss summed over the window. Reviewers noted that its success is NON-MONOTONE in eps
(baseline TM-A: 5, 4, 0, 0 of 38 for eps 0.1/0.25/0.5/1.0), a classic sign of a weak attack.

Here we run the improved white-box PGD from adv_lib (random restarts in the ball, a cosine step
schedule, many steps, optional greedy warm start) against TM-A for both variants:
  (a) an effectively unbounded budget (eps = 5 SD): success should approach the feasibility ceiling
      if the optimiser works;
  (b) the improved PGD at eps in {0.1, 0.25, 0.5}; we check whether success becomes monotone in eps.

Only NEW files are written (results/pgd_sanity.json). Nothing existing is modified.
Usage: PYTHONIOENCODING=utf-8 python pgd_sanity.py [--limit N] [--restarts 5] [--steps 200]
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

import adv_lib as A
import hai_lib as H
from aitcert_path import stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="baseline_s0,hardened_s0")
ap.add_argument("--bounded_eps", default="0.1,0.25,0.5")
ap.add_argument("--unbounded_eps", type=float, default=5.0)
ap.add_argument("--restarts", type=int, default=5)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--limit", type=int, default=0, help="debug: cap number of events per variant")
ap.add_argument("--out", default="pgd_sanity.json")
args = ap.parse_args()

t0 = time.time()
H.set_seed(0)
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
F = len(prep.cols)
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrays = [prep.to_array(df) for df in test]
ys = [H.labels(df) for df in test]
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
BOUNDED = [float(e) for e in args.bounded_eps.split(",")]

out = {"description": "Improved white-box PGD sanity check against TM-A (adaptive-attack guideline).",
       "threat_model": "TM-A", "bounded_eps": BOUNDED, "unbounded_eps": args.unbounded_eps,
       "restarts": args.restarts, "steps": args.steps, "variants": {}}

for tag in args.variants.split(","):
    meta = json.load(open(os.path.join(H.RES, f"e1_{tag}.json")))
    theta = meta["theta"]
    chan_var = np.load(os.path.join(H.RES, "models", f"{tag}_chanvar.npy"))
    model = H.load_model(tag, F)
    for p in model.parameters():
        p.requires_grad_(False)
    model.gru.train()          # cuDNN RNN backward needs training mode
    model.gru.dropout = 0.0    # but no stochasticity

    # recompute the pre-registered PGD-by-eps for THIS run (sanity vs. e2_evasion.json)
    pre_rows, imp_rows, unb_rows = [], [], []
    k = 0
    done = 0
    for fi, (arr, y) in enumerate(zip(arrays, ys), start=1):
        ev = H.events_from_labels(y)
        sub = attacks[attacks.test_file == f"test{fi}"].reset_index(drop=True)
        for j, (s, e) in enumerate(ev):
            detected = bool(meta["detected_vector"][k]); k += 1
            if not detected:
                continue
            if args.limit and done >= args.limit:
                break
            done += 1
            procs = sub.loc[j, "target_processes"].split(";")
            mA = np.array([c.split("_")[0] in procs for c in prep.cols])
            if not mA.any():
                continue
            aid = sub.loc[j, "attack_id"]
            # (a) unbounded feasibility ceiling
            ru = A.improved_pgd(model, arr, s, e, mA, args.unbounded_eps, theta, chan_var,
                                steps=max(80, args.steps // 2), restarts=args.restarts,
                                seed=1000 + j, warm_greedy=True)
            unb_rows.append({"attack_id": aid, "eps": args.unbounded_eps, **ru})
            # (b) improved PGD at each bounded eps + pre-registered PGD for comparison
            for eps in BOUNDED:
                ri = A.improved_pgd(model, arr, s, e, mA, eps, theta, chan_var,
                                    steps=args.steps, restarts=args.restarts, seed=j, warm_greedy=True)
                imp_rows.append({"attack_id": aid, "eps": eps, **ri})
                rp = H.conceal_event(model, arr, s, e, mA, eps, theta, chan_var, steps=60)
                pre_rows.append({"attack_id": aid, "eps": eps, "concealed": bool(rp["concealed"]),
                                 "max_score": rp["max_score"]})
        if args.limit and done >= args.limit:
            break

    def summarise(rows, eps):
        sel = [r for r in rows if abs(r["eps"] - eps) < 1e-9]
        kk = sum(r["concealed"] for r in sel); nn = len(sel)
        return {"eps": eps, "k": int(kk), "n": int(nn),
                "lcb95": S.cp_lower(kk, nn, 0.05), "ucb95": S.cp_upper(kk, nn, 0.05)}

    imp_by_eps = [summarise(imp_rows, ep) for ep in BOUNDED]
    pre_by_eps = [summarise(pre_rows, ep) for ep in BOUNDED]
    unb = summarise(unb_rows, args.unbounded_eps)
    ks_imp = [c["k"] for c in imp_by_eps]
    ks_pre = [c["k"] for c in pre_by_eps]
    monotone = all(ks_imp[i] <= ks_imp[i + 1] for i in range(len(ks_imp) - 1))
    out["variants"][tag] = {
        "theta": theta,
        "improved_pgd_by_eps": imp_by_eps,
        "prereg_pgd_by_eps_recomputed": pre_by_eps,
        "unbounded": unb,
        "improved_monotone_in_eps": bool(monotone),
        "improved_counts": ks_imp, "prereg_counts": ks_pre,
        "rows": {"improved": imp_rows, "prereg": pre_rows, "unbounded": unb_rows},
    }
    print(f"{tag}: improved {ks_imp} (monotone={monotone}); prereg-recomputed {ks_pre}; "
          f"unbounded eps={args.unbounded_eps} -> {unb['k']}/{unb['n']}  "
          f"[{round(time.time()-t0)} s]", flush=True)

out["runtime_s"] = time.time() - t0
out["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "device": str(H.DEV),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "platform": platform.platform()}
with open(os.path.join(H.RES, args.out), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote", args.out, "runtime_s", round(out["runtime_s"], 1))
