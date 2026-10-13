"""E2: adversarial robustness (IR) - white-box concealment attacks against the detector.
For every attack event detected at the pre-registered operating point, the attacker optimises a
bounded perturbation (L_inf <= eps, standardised units) of the readings it controls during
[attack start, attack end + 60 s] to keep the smoothed score below the alarm threshold.
Threat models: TM-ALL (all channels spoofable: no channel integrity), TM-A (channels of the attacked
process(es) only: cross-process channels integrity-protected, cf. legacy CV/NV services),
TM-B (only the attack's target points).
Usage: python e2_evasion.py --tags baseline_s0,hardened_s0"""
import argparse, json, os, re, time
import numpy as np, pandas as pd, torch
import hai_lib as H

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="baseline_s0,hardened_s0")
ap.add_argument("--eps", default="0.1,0.25,0.5,1.0")
ap.add_argument("--steps", type=int, default=60)
args = ap.parse_args()
t0 = time.time()
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrays = [prep.to_array(df) for df in test]
ys = [H.labels(df) for df in test]
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
colnorm = {norm(c): i for i, c in enumerate(prep.cols)}
F = len(prep.cols)
out = {"eps": [float(e) for e in args.eps.split(",")], "models": {}}
for tag in args.tags.split(","):
    meta = json.load(open(os.path.join(H.RES, f"e1_{tag}.json")))
    theta = meta["theta"]
    chan_var = np.load(os.path.join(H.RES, "models", f"{tag}_chanvar.npy"))
    model = H.load_model(tag, F)
    for p in model.parameters():
        p.requires_grad_(False)
    model.gru.train()  # cuDNN RNN backward requires training mode; dropout disabled below
    model.gru.dropout = 0.0
    rows = []
    k = 0
    for fi, (arr, y) in enumerate(zip(arrays, ys), start=1):
        ev = H.events_from_labels(y)
        sub = attacks[attacks.test_file == f"test{fi}"].reset_index(drop=True)
        for j, (s, e) in enumerate(ev):
            detected = bool(meta["detected_vector"][k]); k += 1
            if not detected:
                continue
            procs = sub.loc[j, "target_processes"].split(";")
            mA = np.array([c.split("_")[0] in procs for c in prep.cols])
            tpts = [norm(t) for grp in str(sub.loc[j, "target_points"]).split(";") for t in grp.split(",")]
            mB = np.zeros(F, dtype=bool)
            for t in tpts:
                if t in colnorm:
                    mB[colnorm[t]] = True
            masks = {"TM-ALL": np.ones(F, dtype=bool), "TM-A": mA, "TM-B": mB}
            for tm, m in masks.items():
                if not m.any():
                    continue
                for eps in out["eps"]:
                    r = H.conceal_event(model, arr, s, e, m, eps, theta, chan_var, steps=args.steps)
                    rows.append({"attack_id": sub.loc[j, "attack_id"], "tm": tm, "eps": eps, "n_channels": int(m.sum()), **r})
    df = pd.DataFrame(rows)
    summ = (df.groupby(["tm", "eps"]).agg(n=("concealed", "size"), concealed=("concealed", "sum")).reset_index())
    out["models"][tag] = {"theta": theta, "rows": rows, "summary": summ.to_dict(orient="records")}
    print(tag); print(summ.to_string())
out["runtime_s"] = time.time() - t0
H.jdump(out, "e2_evasion.json")
