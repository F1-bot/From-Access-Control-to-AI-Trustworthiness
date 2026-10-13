"""E4c (exploratory, post hoc): label-only extraction against the DEPLOYED alarm logic.

Unlike E4/E4b, whose oracle is the unsmoothed one-step point score at a separate threshold theta_pt,
here the oracle is the alarm the operator actually sees: the 10-s causally smoothed score at the END
of each query window compared with the deployed threshold theta (episode merging does not change the
label of a single window). Queries are built ONLY from data a plausible attacker holds -- historical
normal data (train1..train5) plus process-consistent deviations:
  * replayed segments (verbatim historical windows);
  * bounded per-channel offsets or freezes on a random channel subset, kept within the observed range;
  * slow linear drifts on a random channel subset, within the observed range.
A label-only 1-D CNN surrogate is trained for Q in {1e3, 1e4, 1e5} and 3 seeds. Fidelity on
operational test windows is the balanced agreement with the deployed alarm (from the stored smoothed
scores), with a moving-block bootstrap (block ~600 s, per file) 95% interval and the agreement on
victim-negative and victim-positive windows reported separately.

This analysis is POST HOC and EXPLORATORY; it was not pre-registered. Writes
results/e4c_extract_deployed.json and a NEW surrogate is kept only in memory.
Usage: PYTHONIOENCODING=utf-8 python e4c_extract_deployed.py [--budgets 1000,10000,100000] [--seeds 0,1,2]
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import hai_lib as H
from aitcert_path import stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--victim", default="baseline_s0")
ap.add_argument("--budgets", default="1000,10000,100000")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--block_s", type=int, default=600)
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--eval_stride", type=int, default=1, help="stride for operational test windows (1 => 600-s blocks)")
args = ap.parse_args()

t0 = time.time()
L = H.W + H.SMOOTH          # context needed to form the smoothed score at the window end
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
F = len(prep.cols)
victim = H.load_model(args.victim, F)
cv = torch.from_numpy(np.load(os.path.join(H.RES, "models", f"{args.victim}_chanvar.npy")).astype(np.float32)).to(H.DEV)
meta = json.load(open(os.path.join(H.RES, f"e1_{args.victim}.json")))
theta = meta["theta"]

pools = [prep.to_array(df) for df in H.read_files("hai-22.04", [f"train{i}.csv" for i in range(1, 6)])]
obs_lo = np.min(np.concatenate(pools), axis=0)
obs_hi = np.max(np.concatenate(pools), axis=0)
sizes = np.array([len(p) - L + 1 for p in pools])
csz = np.concatenate([[0], np.cumsum(sizes)])
swv = [np.lib.stride_tricks.sliding_window_view(p, L, axis=0) for p in pools]   # [M, F, L] each


def base_segments(n, rng):
    pick = rng.integers(0, int(csz[-1]), n)
    fidx = np.searchsorted(csz, pick, side="right") - 1
    local = pick - csz[fidx]
    segs = np.empty((n, L, F), dtype=np.float32)
    for f in range(len(pools)):
        sel = np.where(fidx == f)[0]
        if len(sel):
            segs[sel] = swv[f][local[sel]].transpose(0, 2, 1)   # [k, L, F]
    return segs


def make_queries(n, rng):
    """Process-consistent attacker queries; returns segments and a per-query deviation tag."""
    segs = base_segments(n, rng)
    tags = rng.integers(0, 4, n)   # 0 replay, 1 freeze, 2 offset, 3 drift
    for i in np.where(tags == 1)[0]:          # freeze
        ch = rng.random(F) < rng.uniform(0.1, 0.4)
        segs[i][:, ch] = segs[i][0, ch][None, :]
    for i in np.where(tags == 2)[0]:          # bounded offset
        ch = rng.random(F) < rng.uniform(0.1, 0.4)
        off = rng.uniform(-3.0, 3.0, size=int(ch.sum())).astype(np.float32)
        segs[i][:, ch] = np.clip(segs[i][:, ch] + off[None, :], obs_lo[ch], obs_hi[ch])
    ramp = np.linspace(0, 1, L, dtype=np.float32)[:, None]
    for i in np.where(tags == 3)[0]:          # slow drift
        ch = rng.random(F) < rng.uniform(0.1, 0.4)
        d = rng.uniform(-3.0, 3.0, size=int(ch.sum())).astype(np.float32)
        segs[i][:, ch] = np.clip(segs[i][:, ch] + ramp * d[None, :], obs_lo[ch], obs_hi[ch])
    return segs, tags


@torch.no_grad()
def deployed_label(segs, bs=8192):
    """Deployed alarm at each segment's end: smoothed (10-s MA) score >= theta."""
    out = np.empty(len(segs), dtype=bool)
    for b in range(0, len(segs), bs):
        x = torch.from_numpy(segs[b:b + bs]).to(H.DEV)       # [B, L, F]
        sc = []
        for k in range(H.SMOOTH):
            win = x[:, k:k + H.W]; tgt = x[:, k + H.W]
            sc.append((((victim(win) - tgt) ** 2) / cv[None]).mean(1))
        sm = torch.stack(sc, 1).mean(1)
        out[b:b + bs] = (sm >= theta).cpu().numpy()
    return out


class Sur(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(F, 128, 5, padding=2), nn.ReLU(), nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(),
                                 nn.Conv1d(128, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + F, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, s):
        h = self.net(s.transpose(1, 2))
        return self.head(torch.cat([h.mean(2), h.amax(2), s[:, -1] - s[:, -2]], 1)).squeeze(1)


def train(q, y, seed):
    torch.manual_seed(seed); n = len(q); nv = max(100, n // 10)
    Xtr, Ytr, Xva, Yva = q[nv:], y[nv:], q[:nv], y[:nv]
    m = Sur().to(H.DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    posw = torch.tensor([max(1.0, (len(Ytr) - Ytr.sum()) / max(Ytr.sum(), 1))], device=H.DEV)
    lossf = lambda o, t: nn.functional.binary_cross_entropy_with_logits(o, t, pos_weight=posw)
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
        if bad >= 6:
            break
    m.load_state_dict(state); m.eval(); return m


# operational test windows: deployed labels from the stored smoothed scores (exact deployed oracle)
scores = np.load(os.path.join(H.RES, "models", f"{args.victim}_scores.npz"))
test_arrays = [prep.to_array(df) for df in H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])]
ev_files = []
for fi, arr in enumerate(test_arrays, start=1):
    sc = scores[f"test{fi}"]
    swt = np.lib.stride_tricks.sliding_window_view(arr, L, axis=0)   # row r == arr[r:r+L], ends at t=r+L-1
    T = len(arr)
    t = np.arange(L - 1, T, args.eval_stride)
    r = t - (L - 1)
    lab = sc[t] >= theta
    ev_files.append({"rows": r, "swt": swt, "lab": lab.astype(bool), "ends": t})
lab_all = np.concatenate([f["lab"] for f in ev_files])


@torch.no_grad()
def surrogate_labels_per_file(m, bs=8192):
    outs = []
    for f in ev_files:
        rr = f["rows"]; pred = np.empty(len(rr), dtype=bool)
        for b in range(0, len(rr), bs):
            seg = f["swt"][rr[b:b + bs]].transpose(0, 2, 1)       # [B, L, F]
            o = m(torch.from_numpy(np.ascontiguousarray(seg)).to(H.DEV)).cpu().numpy()
            pred[b:b + bs] = o > 0
        outs.append(pred)
    return outs


def mbb_ci(agree_files, lab_files, block, B, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        ap, vp = [], []
        for a, v in zip(agree_files, lab_files):
            idx = S.block_bootstrap_indices(len(a), block, rng)
            ap.append(a[idx]); vp.append(v[idx])
        a = np.concatenate(ap); v = np.concatenate(vp)
        if v.any() and (~v).any():
            vals.append(0.5 * (a[v].mean() + a[~v].mean()))
    vals = np.array(vals)
    return {"lcb95": float(np.quantile(vals, 0.05)), "ucb95": float(np.quantile(vals, 0.95)),
            "lo025": float(np.quantile(vals, 0.025)), "hi975": float(np.quantile(vals, 0.975)),
            "mean": float(vals.mean()), "valid_reps": int(len(vals))}


out = {"description": "E4c exploratory label-only extraction against the DEPLOYED smoothed alarm.",
       "post_hoc": True, "pre_registered": False, "victim": args.victim, "theta": theta, "L_context": L,
       "query_recipe": "train1-5 + {replay, freeze, bounded offset, slow drift} on 10-40% of channels, "
                       "clipped to observed per-channel range",
       "deployed_oracle": "smoothed 10-s score at window end >= theta",
       "eval": {"n_test_windows": int(len(lab_all)), "deployed_pos_rate": float(lab_all.mean()),
                "n_pos": int(lab_all.sum()), "n_neg": int((~lab_all).sum()), "stride": args.eval_stride},
       "runs": []}

for Q in [int(b) for b in args.budgets.split(",")]:
    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = np.random.default_rng(seed * 104729 + Q)
        q, tags = make_queries(Q, rng)
        y = deployed_label(q)
        qpos = float(y.mean())
        m = train(q, y, seed)
        preds = surrogate_labels_per_file(m)
        agree_files = [(p == f["lab"]).astype(np.float64) for p, f in zip(preds, ev_files)]
        lab_files = [f["lab"] for f in ev_files]
        agree_all = np.concatenate(agree_files)
        pos = lab_all; neg = ~lab_all
        ap = float(agree_all[pos].mean()); an = float(agree_all[neg].mean())
        bal = 0.5 * (ap + an)
        pred_all = np.concatenate(preds)
        ci = mbb_ci(agree_files, lab_files, args.block_s, args.B, seed=seed)
        # naive CP for contrast
        tp = int((agree_all * pos).sum()); tn = int((agree_all * neg).sum())
        cp = 0.5 * (S.cp_upper(tp, int(pos.sum()), 0.025) + S.cp_upper(tn, int(neg.sum()), 0.025))
        row = {"Q": Q, "seed": seed, "query_pos_rate": qpos,
               "balanced": bal, "agree_pos": ap, "agree_neg": an,
               "surrogate_pred_pos_rate": float(pred_all.mean()),
               "mbb": ci, "naive_cp_ucb95": cp,
               "query_type_counts": {t: int((tags == i).sum()) for i, t in enumerate(["replay", "freeze", "offset", "drift"])}}
        out["runs"].append(row)
        print(f"Q={Q} s{seed}: qpos={qpos:.3f} bal={bal:.3f} (pos {ap:.3f} neg {an:.3f}) "
              f"MBB[{ci['lo025']:.3f},{ci['hi975']:.3f}] predpos={pred_all.mean():.3f}  [{round(time.time()-t0)}s]", flush=True)

out["runtime_s"] = time.time() - t0
out["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "platform": platform.platform()}
with open(os.path.join(H.RES, "e4c_extract_deployed.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("runtime_s", round(out["runtime_s"], 1))
