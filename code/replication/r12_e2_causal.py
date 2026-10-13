"""R12: sensitivity of E2 to a strictly causal alarm onset (post hoc; not part of the frozen plan). Identical to r4_e2.py
except that every perturbation is judged with both the alarm logic of the plan and its strictly causal variant
(r_lib.alarm_episodes_causal). Writes results/r_e2_causal_<variant>.json.

Original description of r4_e2.py follows.
R4 (E2 of the replication): adversarial robustness (IR-2) and transfer attacks (IR-1) on HAI 21.03.

Final procedure of the 22.04 study for every attack detected at the frozen operating point, threat model TM-A (the
readings of the attacked process(es)), budgets eps in {0.1, 0.25, 0.5} standard deviations, adapted to the deployed
alarm logic of the plan (smoothed score; exceedances closer than 30 s form an episode; alarm after 20 s of an episode):
  * planned PGD (zero start, 60 sign steps of eps/8, as hai_lib.conceal_event), minimising sum_t relu(score_t - 0.9 theta);
  * improved PGD (5 random restarts, 200 steps with a cosine step schedule, greedy warm start, seed j = index of the
    attack within its file, as adv_lib.improved_pgd), with the same objective and, in a second run, with an objective
    adapted to the on-delay: sum_t relu(min of the smoothed score over the 20-s window starting at t - 0.9 theta);
  * greedy prediction replay (white-box, as e2b_greedy.py).
A perturbation conceals an attack if the deployed alarm logic, evaluated on the perturbed readings with enough lead-in,
raises no alarm overlapping [start, end + 60 s]. An attack is concealed at budget eps if any attack at any budget
eps' <= eps conceals it. Reported as well: the unbounded ceiling (improved PGD, eps = 5, seed 1000 + j), greedy transfer
from the surrogate trained on historical normal data (IR-1) and, for the hardened variant, from the hardening-aware
surrogate. Writes results/r_e2.json. Usage: python r4_e2.py"""
import time

import numpy as np
import torch

import r_lib as L
from r_lib import H

import sys
ONLY = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--variant=")]
VARS = ONLY if ONLY else list(L.VARIANTS)
t0 = time.time()
P = L.require_frozen_plan()
K = int(P["alarm_logic"]["on_delay_s"])
assert K == L.ON_DELAY
prep = L.load_prep()
cols = list(prep.cols)
F = len(cols)
arrays, ys = L.test_data(prep)
attacks = L.load_attacks()
by_file = {f"test{i}": [a for a in attacks if a["test_file"] == f"test{i}"] for i in range(1, 6)}
e1 = L.load_json("r_e1.json")
EPS = (0.1, 0.25, 0.5)
UNBOUNDED = 5.0


def freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)
    m.gru.train()          # cuDNN RNN backward needs training mode
    m.gru.dropout = 0.0    # but no stochasticity
    return m


def smoothed(model, var, x):
    """Causal-smoothed score of a segment x [L, F] for targets L-W (as hai_lib.conceal_event)."""
    win = x.unfold(0, H.W, 1).permute(0, 2, 1)[:-1]
    r = (model(win) - x[H.W:]) ** 2 / var[None, :]
    sc = r.mean(1)
    pad = torch.cat([sc[:1].repeat(H.SMOOTH - 1), sc])
    return pad.unfold(0, H.SMOOTH, 1).mean(1)


def window_min(sm):
    """Minimum of the smoothed score over every on-delay window (alarm iff some window minimum exceeds theta)."""
    return sm.unfold(0, K, 1).min(1)[0] if sm.shape[0] >= K else sm.max().reshape(1)


@torch.no_grad()
def concealed(victim, cv, theta, arr, lo, a1, x_seg, s, e):
    """Deployed alarm logic on the perturbed readings, with a lead-in long enough to see exceedance runs that start earlier."""
    full = arr.copy()
    full[lo:a1 + 1] = x_seg
    off = max(0, lo - K - H.SMOOTH - H.W)
    sc = H.smooth_fast(H.score_from_residuals(H.residuals(victim, full[off:a1 + 1]), cv))
    if hasattr(victim, "gru"):
        victim.gru.train()   # hai_lib.residuals switches the model to eval mode; cuDNN RNN backward needs training mode
    times = np.arange(off, a1 + 1)
    in_win = (times >= s) & (times <= e + H.TOL)
    sc = np.nan_to_num(sc, nan=-np.inf)
    return (not L.alarm_in_window(sc, theta, K, in_win), not L.alarm_in_window_causal(sc, theta, K, in_win))


@torch.no_grad()
def greedy_delta(attacker, seg, pmask, eps):
    X = seg.clone().unsqueeze(0); orig = X.clone(); M = pmask.unsqueeze(0)
    for k in range(H.W, X.shape[1]):
        p = attacker(X[:, k - H.W:k])
        X[:, k] = orig[:, k] + torch.clamp(p - orig[:, k], -eps, eps) * M[:, k]
    return ((X[0] - orig[0]) * pmask).clamp_(-eps, eps)


def pgd(victim, cv, theta, arr, s, e, mask, eps, *, improved, seed, objective="all"):
    T = arr.shape[0]
    a0, a1 = max(H.W, s), min(T - 1, e + H.TOL)
    lo = a0 - H.W
    seg = torch.from_numpy(arr[lo:a1 + 1]).to(H.DEV)
    pmask = torch.zeros_like(seg)
    pmask[(a0 - lo):, :] = torch.from_numpy(mask.astype(np.float32)).to(H.DEV)[None, :]
    var = torch.from_numpy(cv.astype(np.float32)).to(H.DEV)
    times = np.arange(lo + H.W, a1 + 1)
    in_win = torch.from_numpy((times >= s) & (times <= e + H.TOL)).to(H.DEV)
    inits = [torch.zeros_like(seg)]
    steps = 200 if improved else 60
    if improved:
        g = torch.Generator(device="cpu").manual_seed(seed)
        inits += [(torch.rand(seg.shape, generator=g) * 2 - 1).to(H.DEV) * eps for _ in range(5)]
        inits.append(greedy_delta(victim, seg, pmask, eps))
    best_val, best_x = float("inf"), None
    for d0 in inits:
        delta = (d0 * pmask).clone().detach().requires_grad_(True)
        for t in range(steps):
            alpha = eps * (0.5 * (1 + np.cos(np.pi * t / max(1, steps - 1))) * (0.5 - 0.025) + 0.025) if improved else eps / 8
            sm = smoothed(victim, var, seg + delta * pmask)
            loss = torch.relu((window_min(sm) if objective == "window" else sm) - 0.9 * theta).sum()
            if float(loss.item()) == 0.0:
                break
            gr, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= alpha * torch.sign(gr)
                delta.clamp_(-eps, eps)
        with torch.no_grad():
            x = seg + delta * pmask
            sm_x = smoothed(victim, var, x)
            if objective == "window":
                wm = window_min(sm_x)
                ww = in_win[K - 1:][:wm.shape[0]] if wm.shape[0] > 1 else in_win[:1]
            else:
                wm, ww = sm_x, in_win
            val = float(wm[ww].max().item()) if bool(ww.any()) else float(wm.max().item())
        if val < best_val:
            best_val, best_x = val, x.cpu().numpy()
    return concealed(victim, cv, theta, arr, lo, a1, best_x, s, e)


@torch.no_grad()
def greedy(attacker, victim, cv, theta, arr, s, e, mask, eps_list):
    a0, a1 = max(H.W, s), min(len(arr) - 1, e + H.TOL)
    lo = a0 - H.W
    out = []
    for eps in eps_list:
        seg = torch.from_numpy(arr[lo:a1 + 1]).to(H.DEV)
        pmask = torch.zeros_like(seg)
        pmask[(a0 - lo):, :] = torch.from_numpy(mask.astype(np.float32)).to(H.DEV)[None, :]
        x = (seg + greedy_delta(attacker, seg, pmask, eps)).cpu().numpy()
        out.append(concealed(victim, cv, theta, arr, lo, a1, x, s, e))
    return out


sur_data = L.load_model("surrogate_data", F)
sur_hard = L.load_model("surrogate_hardaware", F)
out = {"eps": list(EPS), "unbounded_eps": UNBOUNDED, "threat_model": "TM-A", "on_delay_s": K, "variants": {}}
for tag in VARS:
    th = float(P["operating_points"][tag]["theta"])
    cv = L.chanvar(tag)
    victim = freeze(L.load_model(tag, F))
    det = e1["variants"][tag]["aggregate"]["detected_vector"]
    rows, k = [], 0
    t1 = time.time()
    for fi, (arr, y) in enumerate(zip(arrays, ys), start=1):
        for j, ((s, e), a) in enumerate(zip(H.events_from_labels(y), by_file[f"test{fi}"])):
            d = det[k]; k += 1
            if not d:
                continue
            mA = np.array([c.split("_")[0] in a["target_processes"] for c in cols])
            row = {"attack_id": a["attack_id"], "n_channels": int(mA.sum())}
            if not mA.any():
                row["no_controllable_channel"] = True
                rows.append(row)
                continue
            row["pgd_planned"] = {str(ep): pgd(victim, cv, th, arr, s, e, mA, ep, improved=False, seed=j) for ep in EPS}
            row["pgd_improved"] = {str(ep): pgd(victim, cv, th, arr, s, e, mA, ep, improved=True, seed=j) for ep in EPS}
            row["pgd_improved_window"] = {str(ep): pgd(victim, cv, th, arr, s, e, mA, ep, improved=True, seed=j, objective="window")
                                          for ep in EPS}
            row["greedy"] = dict(zip([str(ep) for ep in EPS], greedy(victim, victim, cv, th, arr, s, e, mA, EPS)))
            row["unbounded"] = pgd(victim, cv, th, arr, s, e, mA, UNBOUNDED, improved=True, seed=1000 + j)
            row["transfer_data_surrogate"] = dict(zip([str(ep) for ep in EPS], greedy(sur_data, victim, cv, th, arr, s, e, mA, EPS)))
            if tag == "hardened_s0":
                row["transfer_hardaware_surrogate"] = dict(zip([str(ep) for ep in EPS], greedy(sur_hard, victim, cv, th, arr, s, e, mA, EPS)))
            for li, lname in ((0, "plan"), (1, "causal")):
                row[f"strongest_ball_{lname}"] = {str(ep): any(row[m][str(e2)][li] for m in ("pgd_planned", "pgd_improved", "pgd_improved_window", "greedy")
                                                               for e2 in EPS if e2 <= ep + 1e-9) for ep in EPS}
            rows.append(row)
            print(tag, a["attack_id"], "plan", row["strongest_ball_plan"]["0.5"], "causal", row["strongest_ball_causal"]["0.5"],
                  "| %.0f s" % (time.time() - t1), flush=True)
    valid = [r for r in rows if not r.get("no_controllable_channel")]
    summ = {"n_attackable": len(valid)}
    for lname in ("plan", "causal"):
        summ[f"strongest_ball_{lname}"] = {str(ep): int(sum(r[f"strongest_ball_{lname}"][str(ep)] for r in valid)) for ep in EPS}
    summ["differing_attacks"] = [r["attack_id"] for r in valid if r["strongest_ball_plan"]["0.5"] != r["strongest_ball_causal"]["0.5"]]
    out["variants"][tag] = {"theta": th, "summary": summ, "rows": rows}
    print(tag, summ, flush=True)
    del victim
    torch.cuda.empty_cache()
out["meta"] = {"script": "r4_e2.py", "runtime_s": time.time() - t0, "finished_utc": L.utc_now(), "versions": L.versions()}
L.save_json(out, f"r_e2_causal_{ONLY[0] if ONLY else 'all'}.json")
print("runtime", round(time.time() - t0))
