"""R5 (E4 of the replication): model extraction (IP-2) and the query-based transfer chain (IR-1) on HAI 21.03.

IP-2, final procedure of the 22.04 study (e4b_extract.py with ip2_reexamination.py), for both rated variants: the
attacker queries windows of historical normal data (train2) of length W + 1, half of them with synthetic bursts;
the label-only interface returns the victim's alarm bit (unsmoothed one-step score above the 99.9th percentile of
the validation point scores); a 1-D CNN surrogate is trained with early stopping at Q = 10^4 queries for seeds 0-2;
fidelity is the balanced agreement on ALL time-ordered operational test windows, with one-sided 95% upper bounds
from a moving-block bootstrap (600 s, file boundaries honoured, B = 2000) and a test-file cluster bootstrap
(B = 1000); the decision uses the largest of these bounds over seeds. Fidelity on the attacker's query distribution
(20,000 fresh queries) is reported. IR-1, as ir1_query_chain.py: score and label interfaces with Q = 10^3, 10^4,
10^5; the surrogate's alarm logit is attacked with 60 sign steps (eps = 0.5, TM-A), and the perturbed readings are judged
by the victim's deployed alarm logic (with the on-delay). Writes results/r_e4.json.
Usage: python r5_e4.py"""
import time

import numpy as np
import torch
import torch.nn as nn

import r_lib as L
from r_lib import H, S

t0 = time.time()
P = L.require_frozen_plan()
prep = L.load_prep()
cols = list(prep.cols)
F = len(cols)
arrays, ys = L.test_data(prep)
attacks = L.load_attacks()
by_file = {f"test{i}": [a for a in attacks if a["test_file"] == f"test{i}"] for i in range(1, 6)}
train = L.read(L.TRAIN)
arrs = [prep.to_array(L.drop_labels(d)) for d in train]
pool = arrs[1]                                   # attacker's historian data: train2
vseg = L.split(arrs)[1][0]
e1 = L.load_json("r_e1.json")
EPS_CHAIN = 0.5


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


def train_early_stopping(q, tgt, kind, seed):
    """e4b_extract.py / ip2_reexamination.py: 90/10 split, Adam 1e-3, batches of 256, stop after 5 epochs without gain."""
    torch.manual_seed(seed); n = len(q); nv = max(100, n // 10)
    Xtr, Ytr, Xva, Yva = q[nv:], tgt[nv:], q[:nv], tgt[:nv]
    m = Sur().to(H.DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = (lambda o, t: nn.functional.mse_loss(o, torch.log(t + 1e-3))) if kind == "score" else nn.functional.binary_cross_entropy_with_logits
    best, state, bad, steps = 1e9, None, 0, 0
    Xv, Yv = torch.from_numpy(Xva).to(H.DEV), torch.from_numpy(Yva.astype(np.float32)).to(H.DEV)
    while steps < 30000:
        perm = np.random.default_rng(seed + steps).permutation(len(Xtr))
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]
            x = torch.from_numpy(Xtr[i]).to(H.DEV); t = torch.from_numpy(Ytr[i].astype(np.float32)).to(H.DEV)
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


def train_fixed_epochs(q, y, kind, epochs):
    """ir1_query_chain.py: fixed number of epochs."""
    m = Sur().to(H.DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    X = torch.from_numpy(q); Y = torch.from_numpy(y.astype(np.float32)); n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, 256):
            i = perm[b:b + 256]; x = X[i].to(H.DEV); tt = Y[i].to(H.DEV); o = m(x)
            loss = nn.functional.mse_loss(o, torch.log(tt + 1e-3)) if kind == "score" else nn.functional.binary_cross_entropy_with_logits(o, tt)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def mbb_upper(agree_files, vlab_files, block, B, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        a_parts, v_parts = [], []
        for a, v in zip(agree_files, vlab_files):
            idx = S.block_bootstrap_indices(len(a), block, rng)
            a_parts.append(a[idx]); v_parts.append(v[idx])
        a = np.concatenate(a_parts); v = np.concatenate(v_parts)
        if v.any() and (~v).any():
            vals.append(0.5 * (a[v].mean() + a[~v].mean()))
    return float(np.quantile(np.array(vals), 0.95)), len(vals)


def file_cluster_upper(agree_files, vlab_files, B, seed=0):
    rng = np.random.default_rng(seed); nf = len(agree_files); vals = []
    for _ in range(B):
        sel = rng.integers(0, nf, nf)
        a = np.concatenate([agree_files[i] for i in sel]); v = np.concatenate([vlab_files[i] for i in sel])
        if v.any() and (~v).any():
            vals.append(0.5 * (a[v].mean() + a[~v].mean()))
    return float(np.quantile(np.array(vals), 0.95)), len(vals)


out = {"Q_ip2": 10000, "seeds": [0, 1, 2], "victims": {}}
for tag in L.VARIANTS:
    t1 = time.time()
    victim = L.load_model(tag, F)
    cvt = torch.from_numpy(L.chanvar(tag).astype(np.float32)).to(H.DEV)
    cv = L.chanvar(tag)
    theta_s = float(P["operating_points"][tag]["theta"])

    @torch.no_grad()
    def vscore(segs):
        o = []
        for b in range(0, len(segs), 4096):
            x = torch.from_numpy(segs[b:b + 4096]).to(H.DEV)
            o.append((((victim(x[:, :H.W]) - x[:, H.W]) ** 2) / cvt[None]).mean(1).cpu().numpy())
        return np.concatenate(o)

    theta_pt = float(np.quantile(vscore(np.stack([vseg[i - H.W:i + 1] for i in range(H.W, len(vseg) - 1, 3)]).astype(np.float32)), 0.999))

    @torch.no_grad()
    def vscore_rows(swv, rows, bs=8192):
        o = []
        for b in range(0, len(rows), bs):
            x = torch.from_numpy(np.ascontiguousarray(swv[rows[b:b + bs]].transpose(0, 2, 1))).to(H.DEV)
            o.append((((victim(x[:, :H.W]) - x[:, H.W]) ** 2) / cvt[None]).mean(1).cpu().numpy())
        return np.concatenate(o)

    @torch.no_grad()
    def predict(m, kind, swv, rows, bs=8192):
        o = []
        for b in range(0, len(rows), bs):
            o.append(m(torch.from_numpy(np.ascontiguousarray(swv[rows[b:b + bs]].transpose(0, 2, 1))).to(H.DEV)).cpu().numpy())
        o = np.concatenate(o)
        return (np.exp(np.clip(o, -30, 30)) > theta_pt) if kind == "score" else (o > 0)

    per_file = []
    for arr in arrays:
        swv = np.lib.stride_tricks.sliding_window_view(arr, H.W + 1, axis=0)
        rows = np.arange(0, len(arr) - H.W - 1)
        per_file.append({"swv": swv, "rows": rows, "vlab": vscore_rows(swv, rows) > theta_pt})
    vlab_all = np.concatenate([pf["vlab"] for pf in per_file])
    rng_e = np.random.default_rng(999)
    eval_att = queries(20000, rng_e)
    lab_att = vscore(eval_att) > theta_pt

    # ---------------- IP-2 (label-only, Q = 1e4, seeds 0-2)
    ip2 = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed * 7919 + 10000)
        q = queries(10000, rng); vs = vscore(q)
        m = train_early_stopping(q, (vs > theta_pt).astype(np.float32), "label", seed)
        agree_files = [(predict(m, "label", pf["swv"], pf["rows"]) == pf["vlab"]).astype(np.float64) for pf in per_file]
        vlab_files = [pf["vlab"] for pf in per_file]
        a_all = np.concatenate(agree_files)
        bal = 0.5 * (a_all[vlab_all].mean() + a_all[~vlab_all].mean())
        mbb, nmbb = mbb_upper(agree_files, vlab_files, 600, 2000, seed=seed)
        fcb, nfcb = file_cluster_upper(agree_files, vlab_files, 1000, seed=seed)
        with torch.no_grad():
            o = np.concatenate([m(torch.from_numpy(eval_att[b:b + 4096]).to(H.DEV)).cpu().numpy() for b in range(0, len(eval_att), 4096)])
        sl = o > 0
        att_bal = 0.5 * ((sl & lab_att).sum() / max(lab_att.sum(), 1) + (~sl & ~lab_att).sum() / max((~lab_att).sum(), 1))
        ip2.append({"seed": seed, "query_pos_rate": float((vs > theta_pt).mean()), "balanced_operational": float(bal),
                    "agree_pos": float(a_all[vlab_all].mean()), "agree_neg": float(a_all[~vlab_all].mean()),
                    "moving_block_ucb95": mbb, "file_cluster_ucb95": fcb, "mbb_valid": nmbb, "fcb_valid": nfcb,
                    "balanced_attacker_distribution": float(att_bal)})
        print(tag, "IP-2 seed", seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in ip2[-1].items()}, flush=True)

    # ---------------- IR-1 query-based chain
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    det = e1["variants"][tag]["aggregate"]["detected_vector"]
    ev_segs = []
    for arr, y in zip(arrays, ys):
        pos = np.where(y[H.W:-1] == 1)[0] + H.W; neg = np.where(y[H.W:-1] == 0)[0] + H.W
        for idxs in (pos, neg):
            sel = rng.choice(idxs, size=min(2500, len(idxs)), replace=False)
            ev_segs.append(np.stack([arr[i - H.W:i + 1] for i in sel]))
    ev_segs = np.concatenate(ev_segs).astype(np.float32)
    v_lab = vscore(ev_segs) > theta_pt

    def make_queries(n):
        t = rng.integers(H.W, len(pool) - 1, n)
        segs = np.stack([pool[i - H.W:i + 1] for i in t]).astype(np.float32)
        for i in np.where(rng.random(n) < 0.5)[0]:
            ch = rng.random(F) < rng.uniform(0.1, 0.3); k = rng.integers(1, 21); sig = rng.uniform(0.5, 3.0)
            segs[i][-k:, ch] += rng.normal(0, sig, size=(k, ch.sum()))
        return segs

    def transfer_attack(sur, kind, eps):
        for p in sur.parameters():
            p.requires_grad_(False)
        res, k = [], 0
        for fi, (arr, y) in enumerate(zip(arrays, ys), start=1):
            for (s, e), a in zip(H.events_from_labels(y), by_file[f"test{fi}"]):
                d = det[k]; k += 1
                if not d:
                    continue
                mk = torch.from_numpy(np.array([c.split("_")[0] in a["target_processes"] for c in cols], dtype=np.float32)).to(H.DEV)
                a0, a1 = max(H.W, s), min(len(arr) - 2, e + H.TOL); lo = a0 - H.W
                seg = torch.from_numpy(arr[lo:a1 + 1]).to(H.DEV)
                pm = torch.zeros_like(seg); pm[(a0 - lo):] = mk[None]
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
                off = max(0, lo - L.ON_DELAY - H.W - H.SMOOTH)
                rv = H.residuals(victim, full[off:a1 + 1])
                scv = H.smooth_fast(H.score_from_residuals(rv, cv))
                times = np.arange(off, a1 + 1); win_m = (times >= s) & (times <= e + H.TOL)
                res.append({"attack_id": a["attack_id"],
                            "concealed": not L.alarm_in_window(np.nan_to_num(scv, nan=-np.inf), theta_s, L.ON_DELAY, win_m)})
        return res

    chain = []
    for kind in ("score", "label"):
        for Q in (1000, 10000, 100000):
            q = make_queries(Q); vs = vscore(q)
            yq = vs if kind == "score" else (vs > theta_pt).astype(np.float32)
            sur = train_fixed_epochs(q, yq, kind, epochs=30 if Q <= 10000 else 12)
            with torch.no_grad():
                o = np.concatenate([sur(torch.from_numpy(ev_segs[b:b + 4096]).to(H.DEV)).cpu().numpy() for b in range(0, len(ev_segs), 4096)])
            s_lab = (np.exp(np.clip(o, -30, 30)) > theta_pt) if kind == "score" else (o > 0)
            tp = int((s_lab & v_lab).sum()); tn = int((~s_lab & ~v_lab).sum())
            tr = transfer_attack(sur, kind, EPS_CHAIN)
            kk = int(sum(r["concealed"] for r in tr))
            chain.append({"interface": kind, "Q": Q, "balanced_agreement": 0.5 * (tp / max(int(v_lab.sum()), 1) + tn / max(int((~v_lab).sum()), 1)),
                          "transfer_k": kk, "transfer_n": len(tr)})
            print(tag, "IR-1 chain", chain[-1], flush=True)
    out["victims"][tag] = {"theta_point": theta_pt, "theta_smoothed": theta_s, "n_test_windows": int(len(vlab_all)),
                           "victim_pos_rate": float(vlab_all.mean()), "ip2_runs": ip2, "ir1_chain": chain,
                           "runtime_s": time.time() - t1}
    del victim
    torch.cuda.empty_cache()
out["meta"] = {"script": "r5_e4.py", "runtime_s": time.time() - t0, "finished_utc": L.utc_now(), "versions": L.versions()}
L.save_json(out, "r_e4.json")
print("runtime", round(time.time() - t0))
