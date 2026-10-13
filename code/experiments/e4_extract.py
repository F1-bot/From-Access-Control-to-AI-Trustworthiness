"""E4: inference-attack protection (IP) - model extraction through the detector's query interface and
the use of the extracted surrogate for transfer concealment attacks (ATLAS chain AML.T0005 ->
AML.T0043 -> AML.T0015).
Interfaces: 'score' (continuous anomaly score returned) vs 'label' (alarm bit only = IP-1 output
minimisation). Query budgets Q (enforced by quotas, legacy service DR-1) are varied.
Fidelity = balanced agreement between surrogate and victim alarm decisions on test windows."""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import hai_lib as H

ap = argparse.ArgumentParser()
ap.add_argument("--victim", default="baseline_s0")
ap.add_argument("--budgets", default="1000,10000,100000")
ap.add_argument("--eps", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
t0 = time.time()
rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json")); F = len(prep.cols)
victim = H.load_model(args.victim, F)
cv = np.load(os.path.join(H.RES, "models", f"{args.victim}_chanvar.npy"))
meta = json.load(open(os.path.join(H.RES, f"e1_{args.victim}.json"))); theta_s = meta["theta"]
sv = np.load(os.path.join(H.RES, "models", f"{args.victim}_scores.npz"))["val"]
pool = [prep.to_array(H.read_files("hai-22.04", ["train5.csv"])[0])]
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
tarr = [prep.to_array(df) for df in test]; ys = [H.labels(df) for df in test]
cvt = torch.from_numpy(cv.astype(np.float32)).to(H.DEV)

@torch.no_grad()
def victim_score(segs):  # segs [N, W+1, F] -> unsmoothed score of the last sample
    out = []
    for b in range(0, len(segs), 4096):
        x = torch.from_numpy(segs[b:b + 4096]).to(H.DEV)
        r = (victim(x[:, :H.W]) - x[:, H.W]) ** 2 / cvt[None]
        out.append(r.mean(1).cpu().numpy())
    return np.concatenate(out)

def make_queries(n):
    t = rng.integers(H.W, len(pool[0]) - 1, n)
    segs = np.stack([pool[0][i - H.W:i + 1] for i in t]).astype(np.float32)
    pert = rng.random(n) < 0.5
    for i in np.where(pert)[0]:
        ch = rng.random(F) < rng.uniform(0.1, 0.3)
        k = rng.integers(1, 21); sig = rng.uniform(0.5, 3.0)
        segs[i][-k:, ch] += rng.normal(0, sig, size=(k, ch.sum()))
    return segs

class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(F, 128, 5, padding=2), nn.ReLU(), nn.Conv1d(128, 128, 5, padding=2, dilation=1), nn.ReLU(),
                                 nn.Conv1d(128, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 * 2 + F, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, seg):  # seg [B, W+1, F]
        h = self.net(seg.transpose(1, 2))
        z = torch.cat([h.mean(2), h.amax(2), seg[:, -1] - seg[:, -2]], 1)
        return self.head(z).squeeze(1)

def train_sur(q, y, kind, epochs=30):
    m = Surrogate().to(H.DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    X = torch.from_numpy(q); Y = torch.from_numpy(y.astype(np.float32))
    n = len(X); bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            i = perm[b:b + bs]; x = X[i].to(H.DEV); t = Y[i].to(H.DEV)
            o = m(x)
            loss = nn.functional.mse_loss(o, torch.log(t + 1e-3)) if kind == "score" else nn.functional.binary_cross_entropy_with_logits(o, t)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m

# point threshold (unsmoothed) for the label interface: 99.9th percentile of validation point scores
val_arr = [prep.to_array(H.read_files("hai-22.04", ["train6.csv"])[0])]
cut = int(len(val_arr[0]) * 0.8); vseg = val_arr[0][cut:]
vidx = np.arange(H.W, len(vseg) - 1, 3)
vq = np.stack([vseg[i - H.W:i + 1] for i in vidx]).astype(np.float32)
theta_pt = float(np.quantile(victim_score(vq), 0.999))
# evaluation windows: half from attack periods, half from normal periods
ev_segs, ev_lab = [], []
for arr, y in zip(tarr, ys):
    pos = np.where(y[H.W:-1] == 1)[0] + H.W; neg = np.where(y[H.W:-1] == 0)[0] + H.W
    for idxs, n in ((pos, 2500), (neg, 2500)):
        sel = rng.choice(idxs, size=min(n, len(idxs)), replace=False)
        ev_segs.append(np.stack([arr[i - H.W:i + 1] for i in sel]))
ev_segs = np.concatenate(ev_segs).astype(np.float32)
v_lab = victim_score(ev_segs) > theta_pt

def fidelity(sur, kind):
    with torch.no_grad():
        o = np.concatenate([sur(torch.from_numpy(ev_segs[b:b + 4096]).to(H.DEV)).cpu().numpy() for b in range(0, len(ev_segs), 4096)])
    s_lab = (np.exp(np.clip(o, -30, 30)) > theta_pt) if kind == "score" else (o > 0)
    tpr = float((s_lab & v_lab).sum() / max(v_lab.sum(), 1)); tnr = float((~s_lab & ~v_lab).sum() / max((~v_lab).sum(), 1))
    return {"balanced_agreement": (tpr + tnr) / 2, "agree_pos": tpr, "agree_neg": tnr,
            "n_eval": int(len(v_lab)), "victim_pos": int(v_lab.sum()), "n_pos_agree": int((s_lab & v_lab).sum()),
            "n_neg_agree": int((~s_lab & ~v_lab).sum())}

def transfer_attack(sur, kind, eps):
    """Craft concealment perturbations on the surrogate (TM-A), evaluate on the victim."""
    attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))
    det = meta["detected_vector"]; k = 0; res = []
    for p in sur.parameters():
        p.requires_grad_(False)
    for fi, (arr, y) in enumerate(zip(tarr, ys), start=1):
        sub = attacks[attacks.test_file == f"test{fi}"].reset_index(drop=True)
        for j, (s, e) in enumerate(H.events_from_labels(y)):
            d = det[k]; k += 1
            if not d:
                continue
            procs = sub.loc[j, "target_processes"].split(";")
            m = torch.from_numpy(np.array([c.split("_")[0] in procs for c in prep.cols], dtype=np.float32)).to(H.DEV)
            a0, a1 = max(H.W, s), min(len(arr) - 2, e + H.TOL); lo = a0 - H.W
            seg = torch.from_numpy(arr[lo:a1 + 1]).to(H.DEV)
            pm = torch.zeros_like(seg); pm[(a0 - lo):] = m[None]
            delta = torch.zeros_like(seg, requires_grad=True)
            for _ in range(60):
                x = seg + delta * pm
                win = x.unfold(0, H.W + 1, 1).permute(0, 2, 1)
                o = sur(win)
                sc = torch.exp(o.clamp(-30, 30)) if kind == "score" else torch.sigmoid(o) * 2 * theta_s
                pad = torch.cat([sc[:1].repeat(H.SMOOTH - 1), sc]); sm = pad.unfold(0, H.SMOOTH, 1).mean(1)
                loss = torch.relu(sm - 0.9 * theta_s).sum()
                if loss.item() == 0:
                    break
                g, = torch.autograd.grad(loss, delta)
                with torch.no_grad():
                    delta -= eps / 8 * torch.sign(g); delta.clamp_(-eps, eps)
            with torch.no_grad():
                x = (seg + delta * pm).cpu().numpy()
            full = arr.copy(); full[lo:a1 + 1] = x
            rv = H.residuals(victim, full[max(0, lo - H.W - H.SMOOTH):a1 + 1])
            scv = H.smooth_fast(H.score_from_residuals(rv, cv))
            off = max(0, lo - H.W - H.SMOOTH)
            times = np.arange(off, a1 + 1); win_m = (times >= s) & (times <= e + H.TOL)
            res.append({"attack_id": sub.loc[j, "attack_id"], "concealed": bool(not np.any(np.nan_to_num(scv[win_m], nan=0) > theta_s))})
    return res

out = {"victim": args.victim, "theta_point": theta_pt, "theta_smoothed": theta_s, "eps": args.eps, "runs": []}
for kind in ("score", "label"):
    for Q in [int(b) for b in args.budgets.split(",")]:
        q = make_queries(Q)
        vs = victim_score(q)
        y = vs if kind == "score" else (vs > theta_pt).astype(np.float32)
        sur = train_sur(q, y, kind, epochs=30 if Q <= 10000 else 12)
        fid = fidelity(sur, kind)
        tr = transfer_attack(sur, kind, args.eps)
        asr = float(np.mean([r["concealed"] for r in tr])) if tr else None
        out["runs"].append({"interface": kind, "Q": Q, "query_pos_rate": float((vs > theta_pt).mean()), **fid,
                            "transfer_asr": asr, "transfer_n": len(tr), "transfer_concealed": int(sum(r["concealed"] for r in tr))})
        print(kind, Q, fid["balanced_agreement"], asr)
out["runtime_s"] = time.time() - t0
H.jdump(out, "e4_extract.json")
