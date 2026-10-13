"""IR-1 query-based black-box chain (reviewer round).

Faithful re-implementation of the E4 extraction-then-transfer chain (e4_extract.py: ATLAS
AML.T0005 -> AML.T0043 -> AML.T0015) so it can be run against BOTH victims WITHOUT overwriting the
existing results/e4_extract.json. The attacker queries the detector (score or label interface) with
synthetic bursts, trains a 1-D CNN surrogate, and uses it to craft transfer concealment perturbations
(TM-A) that are evaluated on the victim. The IR-1 metric is the transfer attack success rate.

For the baseline we also echo the stored results/e4_extract.json (0/38) and attach exact one-sided
95% Clopper-Pearson bounds; running the chain here reproduces it as a cross-check. The same chain is
then run against hardened_s0 (36 detected attacks).

Writes results/ir1_query_chain.json. Nothing existing is modified.
Usage: PYTHONIOENCODING=utf-8 python ir1_query_chain.py [--victims baseline_s0,hardened_s0] [--budgets 1000,10000,100000]
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import hai_lib as H
from aitcert_path import stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--victims", default="baseline_s0,hardened_s0")
ap.add_argument("--budgets", default="1000,10000,100000")
ap.add_argument("--eps", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

t0 = time.time()
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
F = len(prep.cols)
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
tarr = [prep.to_array(df) for df in test]
ys = [H.labels(df) for df in test]
pool = prep.to_array(H.read_files("hai-22.04", ["train5.csv"])[0])
attacks = pd.read_csv(os.path.join(H.ROOT, "data/processed/hai2204_attacks.csv"))


class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(F, 128, 5, padding=2), nn.ReLU(),
                                 nn.Conv1d(128, 128, 5, padding=2, dilation=1), nn.ReLU(),
                                 nn.Conv1d(128, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 * 2 + F, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, seg):
        h = self.net(seg.transpose(1, 2))
        z = torch.cat([h.mean(2), h.amax(2), seg[:, -1] - seg[:, -2]], 1)
        return self.head(z).squeeze(1)


def run_victim(victim_tag):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    victim = H.load_model(victim_tag, F)
    cv = np.load(os.path.join(H.RES, "models", f"{victim_tag}_chanvar.npy"))
    cvt = torch.from_numpy(cv.astype(np.float32)).to(H.DEV)
    meta = json.load(open(os.path.join(H.RES, f"e1_{victim_tag}.json")))
    theta_s = meta["theta"]

    @torch.no_grad()
    def victim_score(segs):
        o = []
        for b in range(0, len(segs), 4096):
            x = torch.from_numpy(segs[b:b + 4096]).to(H.DEV)
            o.append((((victim(x[:, :H.W]) - x[:, H.W]) ** 2) / cvt[None]).mean(1).cpu().numpy())
        return np.concatenate(o)

    def make_queries(n):
        t = rng.integers(H.W, len(pool) - 1, n)
        segs = np.stack([pool[i - H.W:i + 1] for i in t]).astype(np.float32)
        for i in np.where(rng.random(n) < 0.5)[0]:
            ch = rng.random(F) < rng.uniform(0.1, 0.3)
            k = rng.integers(1, 21); sig = rng.uniform(0.5, 3.0)
            segs[i][-k:, ch] += rng.normal(0, sig, size=(k, ch.sum()))
        return segs

    def train_sur(q, y, kind, epochs):
        m = Surrogate().to(H.DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
        X = torch.from_numpy(q); Y = torch.from_numpy(y.astype(np.float32)); n = len(X); bs = 256
        for ep in range(epochs):
            perm = torch.randperm(n)
            for b in range(0, n, bs):
                i = perm[b:b + bs]; x = X[i].to(H.DEV); tt = Y[i].to(H.DEV); o = m(x)
                loss = nn.functional.mse_loss(o, torch.log(tt + 1e-3)) if kind == "score" \
                    else nn.functional.binary_cross_entropy_with_logits(o, tt)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); return m

    # point threshold on validation (train6 last 20%)
    val_arr = prep.to_array(H.read_files("hai-22.04", ["train6.csv"])[0])
    vseg = val_arr[int(len(val_arr) * 0.8):]
    vidx = np.arange(H.W, len(vseg) - 1, 3)
    vq = np.stack([vseg[i - H.W:i + 1] for i in vidx]).astype(np.float32)
    theta_pt = float(np.quantile(victim_score(vq), 0.999))

    # evaluation windows (half attack, half normal)
    ev_segs = []
    for arr, y in zip(tarr, ys):
        pos = np.where(y[H.W:-1] == 1)[0] + H.W; neg = np.where(y[H.W:-1] == 0)[0] + H.W
        for idxs, nn_ in ((pos, 2500), (neg, 2500)):
            sel = rng.choice(idxs, size=min(nn_, len(idxs)), replace=False)
            ev_segs.append(np.stack([arr[i - H.W:i + 1] for i in sel]))
    ev_segs = np.concatenate(ev_segs).astype(np.float32)
    v_lab = victim_score(ev_segs) > theta_pt

    def fidelity(sur, kind):
        with torch.no_grad():
            o = np.concatenate([sur(torch.from_numpy(ev_segs[b:b + 4096]).to(H.DEV)).cpu().numpy()
                                for b in range(0, len(ev_segs), 4096)])
        s_lab = (np.exp(np.clip(o, -30, 30)) > theta_pt) if kind == "score" else (o > 0)
        tp = int((s_lab & v_lab).sum()); tn = int((~s_lab & ~v_lab).sum())
        npos = int(v_lab.sum()); nneg = int((~v_lab).sum())
        return {"balanced_agreement": 0.5 * (tp / max(npos, 1) + tn / max(nneg, 1)),
                "agree_pos": tp / max(npos, 1), "agree_neg": tn / max(nneg, 1),
                "n_eval": int(len(v_lab)), "victim_pos": npos}

    def transfer_attack(sur, kind, eps):
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
                off = max(0, lo - H.W - H.SMOOTH)
                rv = H.residuals(victim, full[off:a1 + 1])
                scv = H.smooth_fast(H.score_from_residuals(rv, cv))
                times = np.arange(off, a1 + 1); win_m = (times >= s) & (times <= e + H.TOL)
                res.append({"attack_id": sub.loc[j, "attack_id"],
                            "concealed": bool(not np.any(np.nan_to_num(scv[win_m], nan=0) > theta_s))})
        return res

    runs = []
    for kind in ("score", "label"):
        for Q in [int(b) for b in args.budgets.split(",")]:
            q = make_queries(Q); vs = victim_score(q)
            y = vs if kind == "score" else (vs > theta_pt).astype(np.float32)
            sur = train_sur(q, y, kind, epochs=30 if Q <= 10000 else 12)
            fid = fidelity(sur, kind)
            tr = transfer_attack(sur, kind, args.eps)
            kk = int(sum(r["concealed"] for r in tr)); nn_ = len(tr)
            runs.append({"interface": kind, "Q": Q, "query_pos_rate": float((vs > theta_pt).mean()),
                         **fid, "transfer_k": kk, "transfer_n": nn_,
                         "transfer_asr": kk / nn_ if nn_ else None,
                         "transfer_lcb95": S.cp_lower(kk, nn_, 0.05), "transfer_ucb95": S.cp_upper(kk, nn_, 0.05)})
            print(f"  {victim_tag} {kind} Q={Q}: fid={fid['balanced_agreement']:.3f} "
                  f"transfer {kk}/{nn_} (UCB {S.cp_upper(kk, nn_, 0.05):.3f})", flush=True)
    return {"theta_smoothed": theta_s, "theta_point": theta_pt, "n_detected": int(sum(meta["detected_vector"])), "runs": runs}


out = {"description": "IR-1 extraction-then-transfer chain (ATLAS AML.T0005->T0043->T0015); "
                      "transfer ASR with exact one-sided 95% bounds.",
       "eps": args.eps, "tau_ir2_reference": 0.20, "victims": {}}
for tag in args.victims.split(","):
    print(tag)
    out["victims"][tag] = run_victim(tag)

# echo the stored baseline e4_extract.json with bounds attached
try:
    e4 = json.load(open(os.path.join(H.RES, "e4_extract.json")))
    stored = []
    for r in e4["runs"]:
        kk = int(r["transfer_concealed"]); nn_ = int(r["transfer_n"])
        stored.append({"interface": r["interface"], "Q": r["Q"], "transfer_k": kk, "transfer_n": nn_,
                       "transfer_asr": r["transfer_asr"],
                       "transfer_lcb95": S.cp_lower(kk, nn_, 0.05), "transfer_ucb95": S.cp_upper(kk, nn_, 0.05)})
    out["stored_e4_extract_baseline"] = stored
except Exception as ex:
    out["stored_e4_extract_baseline"] = f"unavailable: {ex}"

out["runtime_s"] = time.time() - t0
out["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "platform": platform.platform()}
with open(os.path.join(H.RES, "ir1_query_chain.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("runtime_s", round(out["runtime_s"], 1))
