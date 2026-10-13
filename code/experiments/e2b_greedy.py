"""E2b: stronger adaptive evasion - greedy prediction-replay concealment.
At every step t of the attack window the attacker replaces each reading it controls by the
detector's own forecast, limited to +-eps around the true value:  x'_t[c] = x_t[c] + clip(f(x'_{t-W:t})[c] - x_t[c], -eps, eps).
White-box: f is the victim forecaster. Transfer: f is a surrogate forecaster the attacker trained only on
historical normal data (no queries to the victim). Success = no alarm in [start, end + 60 s].
Configurations (threat model x eps) are processed as one batch."""
import argparse, json, os, re, time
import numpy as np, pandas as pd, torch
import hai_lib as H

ap = argparse.ArgumentParser()
ap.add_argument("--victims", default="baseline_s0,hardened_s0")
ap.add_argument("--eps", default="0.1,0.25,0.5,1.0")
ap.add_argument("--surrogate", default="")   # tag of an attacker-trained forecaster for transfer attacks
args = ap.parse_args()
t0 = time.time()
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json")); F = len(prep.cols)
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrays = [prep.to_array(df) for df in test]; ys = [H.labels(df) for df in test]
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper()); colnorm = {norm(c): i for i, c in enumerate(prep.cols)}
EPS = [float(e) for e in args.eps.split(",")]


@torch.no_grad()
def greedy(attacker, victim, cv_v, theta, arr, s, e, masks):
    """Run all (mask, eps) configurations in a batch; return concealed flags and max scores."""
    a0, a1 = max(H.W, s), min(len(arr) - 1, e + H.TOL)
    lo = a0 - H.W
    cfg = [(m, ep) for m in masks for ep in EPS]
    B = len(cfg)
    X = torch.from_numpy(arr[lo:a1 + 1]).to(H.DEV).unsqueeze(0).repeat(B, 1, 1)   # [B, L, F]
    orig = X.clone()
    M = torch.stack([torch.from_numpy(m.astype(np.float32)) for m, _ in cfg]).to(H.DEV)  # [B, F]
    E = torch.tensor([ep for _, ep in cfg], device=H.DEV)[:, None]                       # [B, 1]
    for k in range(H.W, X.shape[1]):
        p = attacker(X[:, k - H.W:k])                      # [B, F]
        delta = torch.clamp(p - orig[:, k], -E, E) * M
        X[:, k] = orig[:, k] + delta
    # victim scores on manipulated sequences
    v = torch.from_numpy(cv_v.astype(np.float32)).to(H.DEV)
    out = []
    for b in range(B):
        win = X[b].unfold(0, H.W, 1).permute(0, 2, 1)[:-1]
        r = ((victim(win) - X[b, H.W:]) ** 2 / v[None]).mean(1)
        pad = torch.cat([r[:1].repeat(H.SMOOTH - 1), r]); sm = pad.unfold(0, H.SMOOTH, 1).mean(1).cpu().numpy()
        times = np.arange(lo + H.W, a1 + 1); w = (times >= s) & (times <= e + H.TOL)
        out.append((not bool(np.any(sm[w] > theta)), float(sm[w].max())))
    return cfg, out


res = {"eps": EPS, "models": {}}
sur = H.load_model(args.surrogate, F) if args.surrogate else None
for tag in args.victims.split(","):
    meta = json.load(open(os.path.join(H.RES, f"e1_{tag}.json"))); theta = meta["theta"]
    cv = np.load(os.path.join(H.RES, "models", f"{tag}_chanvar.npy"))
    victim = H.load_model(tag, F)
    rows, k = [], 0
    for fi, (arr, y) in enumerate(zip(arrays, ys), start=1):
        sub = attacks[attacks.test_file == f"test{fi}"].reset_index(drop=True)
        for j, (s, e) in enumerate(H.events_from_labels(y)):
            det = bool(meta["detected_vector"][k]); k += 1
            if not det:
                continue
            procs = sub.loc[j, "target_processes"].split(";")
            mA = np.array([c.split("_")[0] in procs for c in prep.cols])
            mB = np.zeros(F, dtype=bool)
            for grp in str(sub.loc[j, "target_points"]).split(";"):
                for t in grp.split(","):
                    if norm(t) in colnorm:
                        mB[colnorm[norm(t)]] = True
            masks = {"TM-ALL": np.ones(F, dtype=bool), "TM-A": mA}
            if mB.any():
                masks["TM-B"] = mB
            for mode, attacker in (("white-box", victim),) + ((("transfer", sur),) if sur is not None else ()):
                cfg, out = greedy(attacker, victim, cv, theta, arr, s, e, list(masks.values()))
                names = [n for n in masks for _ in EPS]
                for (m, ep), name, (conc, mx) in zip(cfg, names, out):
                    rows.append({"attack_id": sub.loc[j, "attack_id"], "mode": mode, "tm": name, "eps": ep, "concealed": conc, "max_score": mx})
    df = pd.DataFrame(rows)
    summ = df.groupby(["mode", "tm", "eps"]).agg(n=("concealed", "size"), concealed=("concealed", "sum")).reset_index()
    res["models"][tag] = {"theta": theta, "rows": rows, "summary": summ.to_dict(orient="records")}
    print(tag); print(summ.to_string())
res["runtime_s"] = time.time() - t0
H.jdump(res, "e2b_greedy.json")
