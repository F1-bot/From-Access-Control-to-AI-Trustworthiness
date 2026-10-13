"""E4b: model-extraction fidelity through the detector interface (IP-1/IP-2), properly trained surrogates.
Interfaces: 'score' (continuous anomaly score) vs 'label' (alarm bit only - output minimisation).
Budgets Q are enforced per identity by quotas (legacy DR-1). For each (interface, Q, seed) a 1-D CNN
surrogate is trained on the query/response pairs (90/10 split, early stopping) and its alarm decisions are
compared with the victim's on (i) fresh attacker-distribution windows and (ii) sequestered test windows
(half from attack periods). Fidelity = balanced agreement (mean of agreement on victim-positive and
victim-negative windows)."""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
import hai_lib as H
from aitcert_path import stats as S

ap = argparse.ArgumentParser(); ap.add_argument("--victim", default="baseline_s0")
ap.add_argument("--budgets", default="1000,3000,10000,30000,100000"); ap.add_argument("--seeds", default="0,1,2"); args = ap.parse_args()
t0 = time.time()
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json")); F = len(prep.cols)
victim = H.load_model(args.victim, F); cv = torch.from_numpy(np.load(os.path.join(H.RES, "models", f"{args.victim}_chanvar.npy")).astype(np.float32)).to(H.DEV)
pool = prep.to_array(H.read_files("hai-22.04", ["train5.csv"])[0])
tr6 = prep.to_array(H.read_files("hai-22.04", ["train6.csv"])[0]); vseg = tr6[int(len(tr6) * 0.8):]
test = [(prep.to_array(df), H.labels(df)) for df in H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])]

@torch.no_grad()
def vscore(segs):
    out = []
    for b in range(0, len(segs), 4096):
        x = torch.from_numpy(segs[b:b + 4096]).to(H.DEV)
        out.append((((victim(x[:, :H.W]) - x[:, H.W]) ** 2) / cv[None]).mean(1).cpu().numpy())
    return np.concatenate(out)

theta_pt = float(np.quantile(vscore(np.stack([vseg[i - H.W:i + 1] for i in range(H.W, len(vseg) - 1, 3)]).astype(np.float32)), 0.999))

def queries(n, rng):
    t = rng.integers(H.W, len(pool) - 1, n)
    segs = np.stack([pool[i - H.W:i + 1] for i in t]).astype(np.float32)
    for i in np.where(rng.random(n) < 0.5)[0]:
        ch = rng.random(F) < rng.uniform(0.1, 0.3); k = rng.integers(1, 21); sig = rng.uniform(0.5, 3.0)
        segs[i][-k:, ch] += rng.normal(0, sig, size=(k, ch.sum()))
    return segs

rng_e = np.random.default_rng(999)
eval_attacker = queries(20000, rng_e)
ev_test = []
for arr, y in test:
    pos = np.where(y[H.W:-1] == 1)[0] + H.W; neg = np.where(y[H.W:-1] == 0)[0] + H.W
    for idxs in (pos, neg):
        sel = rng_e.choice(idxs, size=min(2500, len(idxs)), replace=False)
        ev_test.append(np.stack([arr[i - H.W:i + 1] for i in sel]))
ev_test = np.concatenate(ev_test).astype(np.float32)
lab_att, lab_test = vscore(eval_attacker) > theta_pt, vscore(ev_test) > theta_pt

class Sur(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(F, 128, 5, padding=2), nn.ReLU(), nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.Conv1d(128, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + F, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, s):
        h = self.net(s.transpose(1, 2)); return self.head(torch.cat([h.mean(2), h.amax(2), s[:, -1] - s[:, -2]], 1)).squeeze(1)

def train(q, tgt, kind, seed):
    torch.manual_seed(seed); n = len(q); nv = max(100, n // 10)
    Xtr, Ytr, Xva, Yva = q[nv:], tgt[nv:], q[:nv], tgt[:nv]
    m = Sur().to(H.DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = (lambda o, t: nn.functional.mse_loss(o, torch.log(t + 1e-3))) if kind == "score" else nn.functional.binary_cross_entropy_with_logits
    best, state, bad, steps = 1e9, None, 0, 0
    Xv, Yv = torch.from_numpy(Xva).to(H.DEV), torch.from_numpy(Yva.astype(np.float32)).to(H.DEV)
    while steps < 30000:
        perm = np.random.default_rng(seed + steps).permutation(len(Xtr))
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]; x = torch.from_numpy(Xtr[i]).to(H.DEV); t = torch.from_numpy(Ytr[i].astype(np.float32)).to(H.DEV)
            loss = lossf(m(x), t); opt.zero_grad(); loss.backward(); opt.step(); steps += 1
        with torch.no_grad():
            vl = float(lossf(m(Xv), Yv))
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 5:
            break
    m.load_state_dict(state); m.eval(); return m

def fidelity(m, kind, X, vl):
    with torch.no_grad():
        o = np.concatenate([m(torch.from_numpy(X[b:b + 4096]).to(H.DEV)).cpu().numpy() for b in range(0, len(X), 4096)])
    sl = (np.exp(np.clip(o, -30, 30)) > theta_pt) if kind == "score" else (o > 0)
    tp, tn = int((sl & vl).sum()), int((~sl & ~vl).sum()); npos, nneg = int(vl.sum()), int((~vl).sum())
    bal = 0.5 * (tp / max(npos, 1) + tn / max(nneg, 1))
    ucb = 0.5 * (S.cp_upper(tp, npos, 0.025) + S.cp_upper(tn, nneg, 0.025))
    return {"balanced": bal, "ucb95": ucb, "agree_pos": tp / max(npos, 1), "agree_neg": tn / max(nneg, 1), "n_pos": npos, "n_neg": nneg}

out = {"victim": args.victim, "theta_point": theta_pt, "victim_pos_rate_attacker_eval": float(lab_att.mean()), "victim_pos_rate_test_eval": float(lab_test.mean()), "runs": []}
for kind in ("score", "label"):
    for Q in [int(b) for b in args.budgets.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            rng = np.random.default_rng(seed * 7919 + Q)
            q = queries(Q, rng); vs = vscore(q)
            tgt = vs if kind == "score" else (vs > theta_pt).astype(np.float32)
            m = train(q, tgt, kind, seed)
            r = {"interface": kind, "Q": Q, "seed": seed, "query_pos_rate": float((vs > theta_pt).mean()),
                 "attacker_dist": fidelity(m, kind, eval_attacker, lab_att), "test_dist": fidelity(m, kind, ev_test, lab_test)}
            out["runs"].append(r)
            print(kind, Q, seed, round(r["attacker_dist"]["balanced"], 3), round(r["test_dist"]["balanced"], 3), flush=True)
out["runtime_s"] = time.time() - t0
H.jdump(out, "e4b_extract.json")
