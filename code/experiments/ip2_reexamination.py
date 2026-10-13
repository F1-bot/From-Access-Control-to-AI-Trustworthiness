"""IP-2 re-examination (reviewer round).

(a) Documents, from the EXISTING code and results, how the E4b extraction oracle differs from the
    deployed alarm, the query distribution, and the degenerate surrogates (quantified from
    results/e4b_extract.json).
(b) Recomputes the E4b fidelity UPPER bounds with dependent-data methods. E4b treated each of the
    ~18k overlapping 1-Hz test windows as independent and used Clopper-Pearson, which understates the
    uncertainty. We retrain the E4b surrogates for the IP-2-relevant configurations (label-only and
    score, Q = 1e4, seeds 0/1/2), evaluate them on ALL operational test windows in time order, and
    recompute the one-sided 95% upper bound on balanced agreement with
      * a moving-block bootstrap over time (block ~600 s, honouring file boundaries), and
      * a cluster bootstrap over the four test files,
    alongside the naive Clopper-Pearson bound for comparison.

Writes results/ip2_reexamination.json. The victim oracle here is kept identical to E4b (unsmoothed
one-step point score > theta_pt) so the bounds are directly comparable; E4c switches to the deployed
smoothed alarm.
Usage: PYTHONIOENCODING=utf-8 python ip2_reexamination.py [--budget 10000] [--seeds 0,1,2]
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
ap.add_argument("--budget", type=int, default=10000)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--block_s", type=int, default=600)
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--stride", type=int, default=1, help="stride for the time-ordered test windows")
args = ap.parse_args()

t0 = time.time()
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
F = len(prep.cols)
victim = H.load_model(args.victim, F)
cv = torch.from_numpy(np.load(os.path.join(H.RES, "models", f"{args.victim}_chanvar.npy")).astype(np.float32)).to(H.DEV)
pool = prep.to_array(H.read_files("hai-22.04", ["train5.csv"])[0])
tr6 = prep.to_array(H.read_files("hai-22.04", ["train6.csv"])[0])
vseg = tr6[int(len(tr6) * 0.8):]
test = [(prep.to_array(df), H.labels(df)) for df in H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])]


@torch.no_grad()
def vscore(segs):
    o = []
    for b in range(0, len(segs), 4096):
        x = torch.from_numpy(segs[b:b + 4096]).to(H.DEV)
        o.append((((victim(x[:, :H.W]) - x[:, H.W]) ** 2) / cv[None]).mean(1).cpu().numpy())
    return np.concatenate(o)


theta_pt = float(np.quantile(vscore(np.stack([vseg[i - H.W:i + 1] for i in range(H.W, len(vseg) - 1, 3)]).astype(np.float32)), 0.999))
meta = json.load(open(os.path.join(H.RES, f"e1_{args.victim}.json")))
theta_sm = meta["theta"]


# ---------------- (a) documentation ----------------
e4b = json.load(open(os.path.join(H.RES, "e4b_extract.json")))
degen = []
for r in e4b["runs"]:
    td = r["test_dist"]
    npos, nneg = td["n_pos"], td["n_neg"]
    # predicted-positive rate implied by the stored agreements
    pred_pos = td["agree_pos"] * npos + (1 - td["agree_neg"]) * nneg
    pred_pos_rate = pred_pos / (npos + nneg)
    degen.append({"interface": r["interface"], "Q": r["Q"], "seed": r["seed"],
                  "balanced": td["balanced"], "agree_pos": td["agree_pos"], "agree_neg": td["agree_neg"],
                  "surrogate_pred_pos_rate": pred_pos_rate,
                  "victim_pos_rate": npos / (npos + nneg),
                  "near_degenerate": bool(pred_pos_rate > 0.95 or pred_pos_rate < 0.05)})

documentation = {
    "oracle_e4b": {"definition": "victim UNSMOOTHED one-step point score s_t > theta_pt",
                   "theta_pt": theta_pt, "note": "point threshold = 99.9th percentile of validation POINT scores"},
    "oracle_deployed": {"definition": "10-s causally SMOOTHED score tilde_s_t > theta with 30-s episode merging",
                        "theta_smoothed": theta_sm,
                        "note": "deployed alarm uses the smoothed score and merges episodes < 30 s apart; "
                                "E4b labels single windows by the unsmoothed point score, a different and noisier target"},
    "threshold_gap": {"theta_pt": theta_pt, "theta_smoothed": theta_sm, "ratio": theta_pt / theta_sm},
    "query_distribution": {"source": "train5 (historian) windows of length W+1",
                           "perturbation": "50% of queries get synthetic bursts: on a random 10-30% of channels, "
                                           "the last k in [1,20] samples get additive N(0, sigma) noise, sigma in [0.5,3.0] SD",
                           "comment": "synthetic bursts, NOT process-consistent manipulations; the attacker eval "
                                      "distribution is ~50% victim-positive by construction, unlike ~3.8% operationally"},
    "degeneracy": {"runs": degen,
                   "comment": "on operational test windows several surrogates collapse to near-constant predictions; "
                              "balanced agreement near 0.5 then reflects a degenerate classifier, not fidelity"},
    "victim_pos_rate_test_eval_e4b": e4b.get("victim_pos_rate_test_eval"),
}


# ---------------- (b) dependent-data recompute ----------------
class Sur(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(F, 128, 5, padding=2), nn.ReLU(), nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(),
                                 nn.Conv1d(128, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + F, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, s):
        h = self.net(s.transpose(1, 2))
        return self.head(torch.cat([h.mean(2), h.amax(2), s[:, -1] - s[:, -2]], 1)).squeeze(1)


def queries(n, rng):
    t = rng.integers(H.W, len(pool) - 1, n)
    segs = np.stack([pool[i - H.W:i + 1] for i in t]).astype(np.float32)
    for i in np.where(rng.random(n) < 0.5)[0]:
        ch = rng.random(F) < rng.uniform(0.1, 0.3); k = rng.integers(1, 21); sig = rng.uniform(0.5, 3.0)
        segs[i][-k:, ch] += rng.normal(0, sig, size=(k, ch.sum()))
    return segs


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


# time-ordered test windows per file, via sliding-window VIEWS (no full copy; memory-safe)
@torch.no_grad()
def vscore_rows(swv, rows, bs=8192):
    o = []
    for b in range(0, len(rows), bs):
        seg = np.ascontiguousarray(swv[rows[b:b + bs]].transpose(0, 2, 1))   # [B, W+1, F]
        x = torch.from_numpy(seg).to(H.DEV)
        o.append((((victim(x[:, :H.W]) - x[:, H.W]) ** 2) / cv[None]).mean(1).cpu().numpy())
    return np.concatenate(o)


per_file = []
for arr, y in test:
    swv = np.lib.stride_tricks.sliding_window_view(arr, H.W + 1, axis=0)     # [T-W, F, W+1], row r = arr[r:r+W+1]
    rows = np.arange(0, len(arr) - H.W - 1, args.stride)                      # target r+W <= len-2
    vlab = vscore_rows(swv, rows) > theta_pt
    per_file.append({"swv": swv, "rows": rows, "vlab": vlab, "n": len(rows)})
vlab_all = np.concatenate([pf["vlab"] for pf in per_file])


@torch.no_grad()
def predict(m, kind, swv, rows, bs=8192):
    o = []
    for b in range(0, len(rows), bs):
        seg = np.ascontiguousarray(swv[rows[b:b + bs]].transpose(0, 2, 1))
        o.append(m(torch.from_numpy(seg).to(H.DEV)).cpu().numpy())
    o = np.concatenate(o)
    return (np.exp(np.clip(o, -30, 30)) > theta_pt) if kind == "score" else (o > 0)


def balanced(agree, vlab):
    pos = vlab; neg = ~vlab
    ap = agree[pos].mean() if pos.any() else np.nan
    an = agree[neg].mean() if neg.any() else np.nan
    return 0.5 * (ap + an), ap, an


def mbb_upper(agree_files, vlab_files, block, B, seed=0):
    """Moving-block bootstrap over time, honouring file boundaries; 95th percentile of balanced agreement."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        a_parts, v_parts = [], []
        for a, v in zip(agree_files, vlab_files):
            idx = S.block_bootstrap_indices(len(a), block, rng)
            a_parts.append(a[idx]); v_parts.append(v[idx])
        a = np.concatenate(a_parts); v = np.concatenate(v_parts)
        if v.any() and (~v).any():
            vals.append(0.5 * (a[v].mean() + a[~v].mean()))
    vals = np.array(vals)
    return float(np.quantile(vals, 0.95)), int(len(vals))


def file_cluster_upper(agree_files, vlab_files, B, seed=0):
    rng = np.random.default_rng(seed)
    nf = len(agree_files); vals = []
    for _ in range(B):
        sel = rng.integers(0, nf, nf)
        a = np.concatenate([agree_files[i] for i in sel]); v = np.concatenate([vlab_files[i] for i in sel])
        if v.any() and (~v).any():
            vals.append(0.5 * (a[v].mean() + a[~v].mean()))
    return float(np.quantile(np.array(vals), 0.95)), int(len(vals))


recompute = {"budget": args.budget, "block_s": args.block_s, "B": args.B, "stride": args.stride,
             "n_test_windows": int(len(vlab_all)), "victim_pos_rate_all_windows": float(vlab_all.mean()),
             "n_pos": int(vlab_all.sum()), "n_neg": int((~vlab_all).sum()), "runs": []}
for kind in ("label", "score"):
    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = np.random.default_rng(seed * 7919 + args.budget)
        q = queries(args.budget, rng); vs = vscore(q)
        tgt = vs if kind == "score" else (vs > theta_pt).astype(np.float32)
        m = train(q, tgt, kind, seed)
        agree_files = [(predict(m, kind, pf["swv"], pf["rows"]) == pf["vlab"]).astype(np.float64) for pf in per_file]
        vlab_files = [pf["vlab"] for pf in per_file]
        agree_all = np.concatenate(agree_files)
        bal, ap, an = balanced(agree_all, vlab_all)
        npos, nneg = int(vlab_all.sum()), int((~vlab_all).sum())
        tp = int((agree_all * vlab_all).sum()); tn = int((agree_all * ~vlab_all).sum())
        cp_ucb = 0.5 * (S.cp_upper(tp, npos, 0.025) + S.cp_upper(tn, nneg, 0.025))
        mbb, nmbb = mbb_upper(agree_files, vlab_files, args.block_s, args.B, seed=seed)
        fcb, nfcb = file_cluster_upper(agree_files, vlab_files, max(500, args.B // 2), seed=seed)
        row = {"interface": kind, "Q": args.budget, "seed": seed, "balanced": bal, "agree_pos": ap, "agree_neg": an,
               "naive_cp_ucb95": cp_ucb, "moving_block_ucb95": mbb, "file_cluster_ucb95": fcb,
               "mbb_valid_reps": nmbb, "fcb_valid_reps": nfcb}
        recompute["runs"].append(row)
        print(f"{kind} Q={args.budget} s{seed}: bal={bal:.3f} (pos {ap:.3f} neg {an:.3f}) "
              f"CP-UCB {cp_ucb:.3f} | MBB-UCB {mbb:.3f} | file-UCB {fcb:.3f}", flush=True)

# echo the stored e4b UCBs for the same configs
stored = [{"interface": r["interface"], "Q": r["Q"], "seed": r["seed"],
           "test_balanced": r["test_dist"]["balanced"], "test_cp_ucb95": r["test_dist"]["ucb95"]}
          for r in e4b["runs"] if r["Q"] == args.budget]
recompute["stored_e4b_cp_ucb95_same_configs"] = stored

out = {"description": "IP-2 re-examination: oracle/query/degeneracy documentation and dependent-data "
                      "recomputation of E4b fidelity upper bounds.",
       "victim": args.victim, "documentation": documentation, "recompute": recompute,
       "runtime_s": time.time() - t0,
       "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "platform": platform.platform()}}
with open(os.path.join(H.RES, "ip2_reexamination.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("theta_pt", round(theta_pt, 3), "theta_sm", round(theta_sm, 3))
print("runtime_s", round(out["runtime_s"], 1))
