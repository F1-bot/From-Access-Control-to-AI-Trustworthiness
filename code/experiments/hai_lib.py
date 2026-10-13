"""Shared library for the HAI 22.04 case study: data preparation, GRU forecaster, anomaly scoring,
alarm logic, event-level evaluation, explanation localisation and the white-box concealment attack.

The detector follows the common forecasting-residual design used for HAI/HAICon baselines: a GRU
predicts the next sample from a W-second window; the anomaly score is the mean per-channel squared
residual normalised by the channel's residual variance on validation data, smoothed causally.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data", "raw", "hai")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W = 60            # input window (s)
SMOOTH = 10       # causal moving-average length (s)
MERGE_GAP = 30    # merge alarm episodes separated by < 30 s
TOL = 60          # detection tolerance after attack end (s)


# ------------------------------------------------------------------ data
def read_files(version: str, names: Sequence[str]) -> List[pd.DataFrame]:
    out = []
    for n in names:
        p = os.path.join(DATA, version, n)
        if not os.path.exists(p) and os.path.exists(p + ".gz"):
            p = p + ".gz"
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        out.append(df)
    return out


@dataclass
class Prep:
    cols: List[str]
    mu: np.ndarray
    sd: np.ndarray

    def to_array(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols].to_numpy(dtype=np.float32)
        return (x - self.mu) / self.sd

    def save(self, path: str):
        json.dump({"cols": self.cols, "mu": self.mu.tolist(), "sd": self.sd.tolist()}, open(path, "w"))

    @classmethod
    def load(cls, path: str) -> "Prep":
        d = json.load(open(path))
        return cls(d["cols"], np.array(d["mu"], dtype=np.float32), np.array(d["sd"], dtype=np.float32))


def fit_prep(train: List[pd.DataFrame]) -> Prep:
    cat = pd.concat(train, ignore_index=True)
    cols = [c for c in cat.columns if c not in ("timestamp", "Attack", "time") and not c.startswith("Attack")]
    num = cat[cols].astype(np.float64)
    sd = num.std().to_numpy()
    keep = [c for c, s in zip(cols, sd) if s > 1e-6]
    mu = num[keep].mean().to_numpy().astype(np.float32)
    sd = num[keep].std().to_numpy().astype(np.float32)
    return Prep(keep, mu, np.maximum(sd, 1e-3).astype(np.float32))


def labels(df: pd.DataFrame) -> np.ndarray:
    col = "Attack" if "Attack" in df.columns else [c for c in df.columns if c.lower().startswith("attack")][0]
    return df[col].to_numpy().astype(np.int8)


def events_from_labels(y: np.ndarray) -> List[Tuple[int, int]]:
    ev, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j + 1 < n and y[j + 1] == 1:
                j += 1
            ev.append((i, j))
            i = j + 1
        else:
            i += 1
    return ev


# ------------------------------------------------------------------ model
class Forecaster(nn.Module):
    def __init__(self, f: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(f, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, f))

    def forward(self, x):  # x [B, W, F] -> [B, F]
        h, _ = self.gru(x)
        return self.head(h[:, -1]) + x[:, -1]  # residual: predict change from last observation


def window_index(arrays: List[np.ndarray], stride: int = 1) -> np.ndarray:
    idx = []
    for a, arr in enumerate(arrays):
        t = np.arange(W, len(arr), stride)
        idx.append(np.stack([np.full_like(t, a), t], 1))
    return np.concatenate(idx) if idx else np.zeros((0, 2), dtype=int)


def batch_windows(arrays: List[np.ndarray], idx: np.ndarray):
    xs = np.stack([arrays[a][t - W:t] for a, t in idx])
    ys = np.stack([arrays[a][t] for a, t in idx])
    return torch.from_numpy(xs), torch.from_numpy(ys)


def train_forecaster(train_arrays: List[np.ndarray], val_arrays: List[np.ndarray], seed: int = 0,
                     epochs: int = 10, per_epoch: int = 200_000, batch: int = 1024, lr: float = 1e-3,
                     corrupt: float = 0.0, log: Optional[list] = None,
                     train_idx: Optional[np.ndarray] = None) -> Forecaster:
    """Train the GRU forecaster. If corrupt > 0, a random fraction of input channels in each training
    window is replaced by noise (channel-corruption training, used as an IR hardening measure)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = train_arrays[0].shape[1]
    model = Forecaster(f).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_idx = window_index(train_arrays, stride=1) if train_idx is None else train_idx
    va_idx = window_index(val_arrays, stride=5)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        sel = tr_idx[rng.choice(len(tr_idx), size=min(per_epoch, len(tr_idx)), replace=False)]
        tot, nb = 0.0, 0
        for b in range(0, len(sel), batch):
            x, y = batch_windows(train_arrays, sel[b:b + batch])
            x, y = x.to(DEV), y.to(DEV)
            if corrupt > 0:
                m = (torch.rand(x.shape[0], 1, x.shape[2], device=DEV) < corrupt).float()
                noise = torch.randn_like(x) * 0.5
                x = x * (1 - m) + (x + noise) * m
            loss = torch.mean((model(x) - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        vl = residual_mse(model, val_arrays, va_idx)
        if log is not None:
            log.append({"epoch": ep, "train_mse": tot / nb, "val_mse": vl})
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def residual_mse(model, arrays, idx, batch=4096):
    model.eval()
    tot, n = 0.0, 0
    for b in range(0, len(idx), batch):
        x, y = batch_windows(arrays, idx[b:b + batch])
        r = model(x.to(DEV)) - y.to(DEV)
        tot += float((r ** 2).mean(1).sum()); n += len(r)
    return tot / max(n, 1)


@torch.no_grad()
def residuals(model, arr: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Squared residual per time step and channel, shape [T, F]; first W rows are NaN."""
    model.eval()
    T, F = arr.shape
    out = np.full((T, F), np.nan, dtype=np.float32)
    xt = torch.from_numpy(arr).to(DEV)
    for s in range(W, T, batch):
        e = min(T, s + batch)
        idx = torch.arange(s, e, device=DEV)
        win = torch.stack([xt[i - W:i] for i in range(s, e)]) if e - s < 64 else xt.unfold(0, W, 1)[s - W:e - W].permute(0, 2, 1)
        pred = model(win)
        out[s:e] = ((pred - xt[s:e]) ** 2).cpu().numpy()
    return out


def smooth_causal(x: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    y = np.where(np.isnan(x), 0.0, x)
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    out[np.isnan(x)] = np.nan
    return out


def smooth_fast(x: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    y = np.where(np.isnan(x), 0.0, x).astype(np.float64)
    c = np.cumsum(np.insert(y, 0, 0.0))
    i = np.arange(len(y))
    lo = np.maximum(0, i - k + 1)
    out = (c[i + 1] - c[lo]) / (i + 1 - lo)
    out[np.isnan(x)] = np.nan
    return out


def score_from_residuals(res: np.ndarray, chan_var: np.ndarray) -> np.ndarray:
    out = np.full(res.shape[0], np.nan)
    ok = ~np.isnan(res[:, 0])
    out[ok] = (res[ok] / chan_var[None, :]).mean(axis=1)
    return out


@torch.no_grad()
def window_scores(model, arrays, idx, chan_var, batch=4096) -> np.ndarray:
    """Unsmoothed score of the windows listed in idx (array, t)."""
    v = torch.from_numpy(chan_var.astype(np.float32)).to(DEV)
    out = []
    for b in range(0, len(idx), batch):
        x, y = batch_windows(arrays, idx[b:b + batch])
        r = (model(x.to(DEV)) - y.to(DEV)) ** 2 / v[None, :]
        out.append(r.mean(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def load_model(tag: str, f: int) -> "Forecaster":
    from safetensors.torch import load_file
    m = Forecaster(f).to(DEV)
    m.load_state_dict(load_file(os.path.join(RES, "models", f"{tag}.safetensors")))
    m.eval()
    return m


# ------------------------------------------------------------------ alarms and evaluation
def alarm_episodes(score: np.ndarray, theta: float) -> List[Tuple[int, int]]:
    a = np.nan_to_num(score, nan=-np.inf) > theta
    eps = events_from_labels(a.astype(np.int8))
    merged: List[Tuple[int, int]] = []
    for s, e in eps:
        if merged and s - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def evaluate_file(score: np.ndarray, y: np.ndarray, theta: float) -> Dict:
    ev = events_from_labels(y)
    al = alarm_episodes(score, theta)
    det, ttd = [], []
    for s, e in ev:
        hit = [a for a in al if a[0] <= e + TOL and a[1] >= s]
        det.append(bool(hit))
        if hit:
            first = max(min(a[0] for a in hit), s)
            ttd.append(first - s)
        else:
            ttd.append(None)
    # false alarms: episodes not overlapping any [s, e + TOL]
    fa = [a for a in al if not any(a[0] <= e + TOL and a[1] >= s for s, e in ev)]
    attack_time = sum(min(len(y), e + TOL + 1) - s for s, e in ev)
    normal_h = (len(y) - W - attack_time) / 3600.0
    a = np.nan_to_num(score, nan=-np.inf) > theta
    yy = y.astype(bool)
    tp, fp, fn = int((a & yy).sum()), int((a & ~yy).sum()), int((~a & yy).sum())
    return {"events": ev, "detected": det, "ttd": ttd, "alarms": al, "false_alarms": len(fa),
            "normal_hours": normal_h, "n_alarms": len(al), "hours": len(y) / 3600.0,
            "point_tp": tp, "point_fp": fp, "point_fn": fn}


def aggregate(evals: List[Dict]) -> Dict:
    det = sum(sum(e["detected"]) for e in evals)
    n = sum(len(e["events"]) for e in evals)
    fa = sum(e["false_alarms"] for e in evals)
    nh = sum(e["normal_hours"] for e in evals)
    tp = sum(e["point_tp"] for e in evals); fp = sum(e["point_fp"] for e in evals); fn = sum(e["point_fn"] for e in evals)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    ttd = [t for e in evals for t in e["ttd"] if t is not None]
    return {"events_detected": det, "events": n, "event_recall": det / n if n else float("nan"),
            "false_alarms": fa, "normal_hours": nh, "far_per_h": fa / nh,
            "alarms_per_h": sum(e["n_alarms"] for e in evals) / sum(e["hours"] for e in evals),
            "point_precision": prec, "point_recall": rec,
            "point_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "ttd_median_s": float(np.median(ttd)) if ttd else None,
            "ttd_p90_s": float(np.percentile(ttd, 90)) if ttd else None}


# ------------------------------------------------------------------ explanation localisation
def localise(res: np.ndarray, chan_var: np.ndarray, cols: List[str], alarm_start: int, horizon: int = 30, topk: int = 5):
    seg = res[alarm_start:alarm_start + horizon] / chan_var[None, :]
    contrib = np.nanmean(seg, axis=0)
    order = np.argsort(-contrib)
    procs = {}
    for c, v in zip(cols, contrib):
        p = c.split("_")[0]
        procs[p] = procs.get(p, 0.0) + float(v)
    top_proc = max(procs, key=procs.get)
    return {"top_process": top_proc, "process_scores": procs, "top_tags": [cols[i] for i in order[:topk]]}


# ------------------------------------------------------------------ concealment attack (white-box PGD)
def conceal_event(model: Forecaster, arr: np.ndarray, s: int, e: int, chan_mask: np.ndarray, eps: float,
                  theta: float, chan_var: np.ndarray, steps: int = 60, margin: float = 0.9,
                  surrogate: Optional[Forecaster] = None, sur_var: Optional[np.ndarray] = None,
                  sur_theta: Optional[float] = None) -> Dict:
    """Optimise a bounded perturbation on the attacker-controlled channels over [s, e+TOL] so that the
    smoothed score stays below the alarm threshold. If `surrogate` is given, the perturbation is
    optimised on the surrogate (transfer attack) and evaluated on `model`."""
    T = arr.shape[0]
    a0, a1 = max(W, s), min(T - 1, e + TOL)
    lo = a0 - W
    seg = torch.from_numpy(arr[lo:a1 + 1]).to(DEV)                    # [L, F]
    pmask = torch.zeros_like(seg)
    pmask[(a0 - lo):, :] = torch.from_numpy(chan_mask.astype(np.float32)).to(DEV)[None, :]
    att_model = surrogate if surrogate is not None else model
    var = torch.from_numpy((sur_var if surrogate is not None else chan_var).astype(np.float32)).to(DEV)
    th = sur_theta if surrogate is not None else theta
    delta = torch.zeros_like(seg, requires_grad=True)
    alpha = eps / 8

    def smoothed_scores(m, v, x):
        win = x.unfold(0, W, 1).permute(0, 2, 1)[:-1]                     # windows ending before each target
        tgt = x[W:]
        r = (m(win) - tgt) ** 2 / v[None, :]
        sc = r.mean(1)                                                     # score for t = lo+W .. a1
        # causal moving average over SMOOTH (pad with first value)
        pad = torch.cat([sc[:1].repeat(SMOOTH - 1), sc])
        return pad.unfold(0, SMOOTH, 1).mean(1)

    for _ in range(steps):
        x = seg + delta * pmask
        sm = smoothed_scores(att_model, var, x)
        loss = torch.relu(sm - margin * th).sum()
        if loss.item() == 0:
            break
        g, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= alpha * torch.sign(g)
            delta.clamp_(-eps, eps)
    with torch.no_grad():
        x = seg + delta * pmask
        sm = smoothed_scores(model, torch.from_numpy(chan_var.astype(np.float32)).to(DEV), x).cpu().numpy()
    # the smoothed score index k corresponds to time lo + W + k; detection window starts at s
    times = np.arange(lo + W, a1 + 1)
    in_win = (times >= s) & (times <= e + TOL)
    alarm = bool(np.any(sm[in_win] > theta))
    return {"concealed": not alarm, "max_score": float(np.max(sm[in_win])) if in_win.any() else None,
            "linf": float(delta.abs().max().item())}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def jdump(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
