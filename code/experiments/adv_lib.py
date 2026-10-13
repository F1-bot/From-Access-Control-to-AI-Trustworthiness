"""Helper module for the adversarial re-analyses requested in the major revision (reviewer round).
Kept separate from hai_lib.py and aitcert so no existing artefact is modified.

Contents:
  - three_way_decision: Clopper-Pearson one-sided 95% accept/reject/inconclusive for a `<=` requirement;
  - aggregate_concealed: collapse per-(attack_id) rows to k/n by an OR over admissible runs;
  - improved_pgd: a stronger white-box concealment PGD with random restarts, a step schedule and an
    optional greedy warm start (used by pgd_sanity.py and transfer_hardaware.py).

The detector / scoring semantics are imported from hai_lib; nothing here mutates it.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import hai_lib as H
from aitcert_path import stats as S


# ----------------------------------------------------------------- decision helpers
def three_way(k: int, n: int, tau: float, direction: str = "<=", alpha: float = 0.05) -> Dict:
    """Exact one-sided 95% Clopper-Pearson bounds and a three-way decision.

    For a `<=` requirement (attack success rate <= tau):
        accept        if UCB <= tau   (evidence that the system conforms),
        reject        if LCB >  tau   (evidence that it violates the requirement),
        inconclusive  otherwise.
    """
    lcb = S.cp_lower(k, n, alpha)
    ucb = S.cp_upper(k, n, alpha)
    est = k / n if n else float("nan")
    if direction == "<=":
        if ucb <= tau:
            dec = "accept"
        elif lcb > tau:
            dec = "reject"
        else:
            dec = "inconclusive"
    else:  # ">="
        if lcb >= tau:
            dec = "accept"
        elif ucb < tau:
            dec = "reject"
        else:
            dec = "inconclusive"
    return {"k": int(k), "n": int(n), "estimate": est, "lcb95": lcb, "ucb95": ucb,
            "tau": tau, "direction": direction, "alpha": alpha, "decision": dec}


def aggregate_concealed(rows: List[Dict], pred) -> Dict[str, bool]:
    """OR the `concealed` flag over all rows matching `pred(row)`, keyed by attack_id."""
    out: Dict[str, bool] = {}
    for r in rows:
        if pred(r):
            out[r["attack_id"]] = out.get(r["attack_id"], False) or bool(r["concealed"])
    return out


def kn(d: Dict[str, bool]) -> Tuple[int, int]:
    return int(sum(d.values())), int(len(d))


# ----------------------------------------------------------------- improved white-box PGD
def _smoothed_scores(model, var: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Causal-smoothed per-window score of a manipulated segment x [L, F], matching hai_lib.conceal_event
    (same unfold, same SMOOTH moving average padded with the first value)."""
    win = x.unfold(0, H.W, 1).permute(0, 2, 1)[:-1]   # windows ending before each target
    tgt = x[H.W:]
    r = (model(win) - tgt) ** 2 / var[None, :]
    sc = r.mean(1)
    pad = torch.cat([sc[:1].repeat(H.SMOOTH - 1), sc])
    return pad.unfold(0, H.SMOOTH, 1).mean(1)


@torch.no_grad()
def _greedy_delta(attacker, arr_seg: torch.Tensor, pmask: torch.Tensor, eps: float) -> torch.Tensor:
    """Greedy prediction-replay perturbation (used as a PGD warm start). arr_seg [L, F]."""
    X = arr_seg.clone().unsqueeze(0)      # [1, L, F]
    orig = X.clone()
    M = pmask.unsqueeze(0)
    for k in range(H.W, X.shape[1]):
        p = attacker(X[:, k - H.W:k])
        d = torch.clamp(p - orig[:, k], -eps, eps) * M[:, k]
        X[:, k] = orig[:, k] + d
    return ((X[0] - orig[0]) * pmask).clamp_(-eps, eps)


def improved_pgd(model, arr: np.ndarray, s: int, e: int, chan_mask: np.ndarray, eps: float,
                 theta: float, chan_var: np.ndarray, *, steps: int = 200, restarts: int = 5,
                 margin: float = 0.9, seed: int = 0, warm_greedy: bool = False,
                 surrogate=None, sur_var: Optional[np.ndarray] = None, sur_theta: Optional[float] = None,
                 device=None) -> Dict:
    """Stronger white-box (or transfer) concealment attack.

    Improvements over hai_lib.conceal_event:
      * `restarts` random initialisations uniformly in the L_inf ball (plus a zero start and,
        optionally, a greedy warm start);
      * `steps` gradient steps with a cosine step-size schedule from eps/2 down to eps/40;
      * keeps the best (lowest max in-window smoothed score) perturbation over all restarts.

    The perturbation is optimised on `surrogate` if given (transfer) and always *evaluated* on `model`
    with the true `chan_var`/`theta`, exactly as hai_lib.conceal_event evaluates.
    """
    dev = device if device is not None else H.DEV
    T = arr.shape[0]
    a0, a1 = max(H.W, s), min(T - 1, e + H.TOL)
    lo = a0 - H.W
    seg = torch.from_numpy(arr[lo:a1 + 1]).to(dev)
    pmask = torch.zeros_like(seg)
    pmask[(a0 - lo):, :] = torch.from_numpy(chan_mask.astype(np.float32)).to(dev)[None, :]

    att_model = surrogate if surrogate is not None else model
    var_opt = torch.from_numpy((sur_var if surrogate is not None else chan_var).astype(np.float32)).to(dev)
    th_opt = sur_theta if surrogate is not None else theta
    var_eval = torch.from_numpy(chan_var.astype(np.float32)).to(dev)

    times = np.arange(lo + H.W, a1 + 1)
    in_win = (times >= s) & (times <= e + H.TOL)
    in_win_t = torch.from_numpy(in_win).to(dev)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    inits = [torch.zeros_like(seg)]
    for _ in range(restarts):
        inits.append((torch.rand(seg.shape, generator=rng) * 2 - 1).to(dev) * eps)
    if warm_greedy:
        inits.append(_greedy_delta(att_model, seg, pmask, eps).to(dev))

    best_delta, best_max = None, float("inf")
    for d0 in inits:
        delta = (d0 * pmask).clone().detach().requires_grad_(True)
        for t in range(steps):
            alpha = eps * (0.5 * (1 + np.cos(np.pi * t / max(1, steps - 1))) * (0.5 - 0.025) + 0.025)
            x = seg + delta * pmask
            sm = _smoothed_scores(att_model, var_opt, x)
            loss = torch.relu(sm - margin * th_opt).sum()
            if float(loss.item()) == 0.0:
                break
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= alpha * torch.sign(g)
                delta.clamp_(-eps, eps)
        with torch.no_grad():
            x = seg + delta * pmask
            sm_eval = _smoothed_scores(model, var_eval, x)
            mx = float(sm_eval[in_win_t].max().item()) if in_win.any() else -float("inf")
            if mx < best_max:
                best_max, best_delta = mx, (delta.detach() * pmask).clone()
    concealed = bool(best_max <= theta)
    return {"concealed": concealed, "max_score": best_max,
            "linf": float(best_delta.abs().max().item()) if best_delta is not None else 0.0}
