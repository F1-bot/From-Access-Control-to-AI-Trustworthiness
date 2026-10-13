"""Helpers for the major-revision re-analyses (scripts e5b_*, e1c_*, e3c_*, id3_units.py).

hai_lib.py and aitcert are imported and never modified. Every result written through `save_json`
carries the run time, the random seeds and the software versions.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as st

import hai_lib as H
from aitcert_path import stats as S

# Training recipes as actually executed (run_all.sh / run_b.sh / script defaults)
E1_RECIPE = {"seed": 0, "epochs": 12, "per_epoch": 250_000, "corrupt": 0.0,
             "source": "run_all.sh: e1_baseline.py --seed 0 --epochs 12 --per_epoch 250000 --tag baseline"}
E3_RECIPE = {"seed": 0, "trusted_seed": 100, "epochs": 10, "per_epoch": 200_000,
             "source": "run_all.sh: e3_poison.py --beta 0.01 --epochs 10 --per_epoch 200000"}
E3B_RECIPE = {"epochs": 8, "per_epoch": 150_000, "q": 0.02, "fold_seeds": [10, 11, 12],
              "source": "run_b.sh: e3b_poison.py --betas 0.01,0.05,0.10 (defaults epochs 8, per_epoch 150000, q 0.02)"}


# ------------------------------------------------------------------ bookkeeping
def env_info() -> Dict:
    import pandas
    import scipy
    import torch
    d = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
         "scipy": scipy.__version__, "pandas": pandas.__version__, "torch": torch.__version__,
         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": str(H.DEV),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "cudnn_deterministic_flag": bool(torch.backends.cudnn.deterministic),
         "cudnn_benchmark_flag": bool(torch.backends.cudnn.benchmark)}
    for mod in ("safetensors", "sklearn"):
        try:
            d[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            pass
    return d


def run_meta(script: str, t0: float, seeds: Dict, extra: Optional[Dict] = None) -> Dict:
    m = {"script": script, "finished_local": datetime.datetime.now().isoformat(timespec="seconds"),
         "runtime_s": round(time.time() - t0, 1), "seeds": seeds, "software": env_info(),
         "hai_lib_constants": {"W": H.W, "SMOOTH": H.SMOOTH, "MERGE_GAP": H.MERGE_GAP, "TOL": H.TOL}}
    if extra:
        m.update(extra)
    return m


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def save_json(obj: Dict, name: str) -> str:
    path = os.path.join(H.RES, name)
    with open(path, "w", encoding="utf8") as f:
        json.dump(_clean(obj), f, indent=1, default=str)
    return path


# ------------------------------------------------------------------ rates and proportions
def rate_summary(count: int, exposure_h: float, alpha: float = 0.05) -> Dict:
    """Point estimate, exact one-sided (1-alpha) upper bound and exact two-sided (1-alpha) interval."""
    return {"count": int(count), "exposure_h": float(exposure_h), "rate_per_h": count / exposure_h,
            "ucb95_one_sided": S.poisson_upper(count, exposure_h, alpha),
            "ci95_two_sided": [S.poisson_lower(count, exposure_h, alpha / 2), S.poisson_upper(count, exposure_h, alpha / 2)]}


def prop_summary(k: int, n: int, alpha: float = 0.05) -> Dict:
    return {"k": int(k), "n": int(n), "estimate": k / n if n else None,
            "lcb95_one_sided": S.cp_lower(k, n, alpha), "ucb95_one_sided": S.cp_upper(k, n, alpha),
            "ci95_two_sided": [S.cp_lower(k, n, alpha / 2), S.cp_upper(k, n, alpha / 2)]}


def rate_ratio_exact(n1: int, t1: float, n2: int, t2: float, alpha: float = 0.05) -> Dict:
    """Ratio lambda1/lambda2 of two Poisson rates with the exact conditional (Clopper-Pearson) 1-alpha
    two-sided interval and the exact conditional one-sided p-value for H0: lambda1 <= lambda2."""
    N = n1 + n2
    est = (n1 / t1) / (n2 / t2) if n2 > 0 else float("inf")
    def to_ratio(p):
        return float("inf") if p >= 1 else p * t2 / ((1 - p) * t1)
    lo = to_ratio(S.cp_lower(n1, N, alpha / 2)); hi = to_ratio(S.cp_upper(n1, N, alpha / 2))
    p0 = t1 / (t1 + t2)
    p_greater = float(st.binom.sf(n1 - 1, N, p0))   # P(X >= n1 | equal rates)
    return {"ratio": est, "ci95_two_sided": [lo, hi], "p_one_sided_lambda1_gt_lambda2": p_greater}


def auc_mw(pos: Sequence[float], neg: Sequence[float]) -> float:
    """ROC AUC = P(pos > neg) + 0.5 P(pos = neg) (Mann-Whitney)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    r = st.rankdata(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def auc_bootstrap(pos, neg, B: int = 10000, seed: int = 0) -> List[float]:
    """Stratified i.i.d. percentile bootstrap 95% interval of the AUC (ignores serial dependence)."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = auc_mw(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))])
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


# ------------------------------------------------------------------ E1 / E5 normal-period conventions
def e1_normal_mask(y: np.ndarray) -> np.ndarray:
    """Seconds counted as normal operation by hai_lib.evaluate_file: not the first W s and not inside
    any [attack start, attack end + TOL]."""
    T = len(y)
    m = np.ones(T, dtype=bool)
    m[:H.W] = False
    for s, e in H.events_from_labels(y):
        m[s:min(T, e + H.TOL + 1)] = False
    return m


def e5_normal_mask(y: np.ndarray) -> np.ndarray:
    """Mask used by e5_drift.py: exclude [start - 300 s, end + 600 s) around every attack."""
    m = np.ones(len(y), dtype=bool)
    for s, e in H.events_from_labels(y):
        m[max(0, s - 300):e + 600] = False
    return m


def e1_false_alarm_episodes(score: np.ndarray, y: np.ndarray, theta: float) -> List[Tuple[int, int]]:
    """False-alarm episodes exactly as hai_lib.evaluate_file counts them."""
    ev = H.events_from_labels(y)
    al = H.alarm_episodes(score, theta)
    return [a for a in al if not any(a[0] <= e + H.TOL and a[1] >= s for s, e in ev)]


def e5_far(score: np.ndarray, mask: np.ndarray, theta: float) -> Tuple[int, float]:
    """Alarm episodes on the masked score and normal hours, as in e5_drift.py."""
    sc = score.astype(np.float64).copy()
    sc[~mask] = np.nan
    return len(H.alarm_episodes(sc, theta)), float(mask.sum()) / 3600.0


def hourly_bins(n: int, mask: np.ndarray, starts: Sequence[int], bin_s: int = 3600) -> Tuple[np.ndarray, np.ndarray]:
    """Clock-hour bins from the start of a file: event counts (by start time) and normal exposure (h)."""
    nb = int(math.ceil(n / bin_s))
    counts = np.zeros(nb, dtype=int)
    for s in starts:
        counts[s // bin_s] += 1
    expo = np.array([mask[b * bin_s:min(n, (b + 1) * bin_s)].sum() / 3600.0 for b in range(nb)])
    return counts, expo


def dispersion_test(counts: Sequence[int], expo: Sequence[float], B: int = 100_000, seed: int = 0) -> Dict:
    """Pearson chi-square test of the homogeneous Poisson model for counts with exposures t_i
    (Fisher's index-of-dispersion test generalised to unequal exposure), with the chi-square
    p-value and a conditional Monte Carlo p-value (multinomial given the total)."""
    c = np.asarray(counts, dtype=float); t = np.asarray(expo, dtype=float)
    keep = t > 0
    c, t = c[keep], t[keep]
    N, T = c.sum(), t.sum()
    lam = N / T
    mu = lam * t
    x2 = float(((c - mu) ** 2 / mu).sum())
    df = len(c) - 1
    rng = np.random.default_rng(seed)
    sims = rng.multinomial(int(N), t / T, size=B)
    x2s = ((sims - mu[None, :]) ** 2 / mu[None, :]).sum(1)
    out = {"n_bins": int(len(c)), "total_count": int(N), "total_exposure_h": float(T), "rate_per_h": float(lam),
           "pearson_x2": x2, "df": int(df), "dispersion_x2_over_df": x2 / df if df > 0 else None,
           "p_chi2_overdispersion": float(st.chi2.sf(x2, df)),
           "p_monte_carlo_conditional": float((1 + np.sum(x2s >= x2 - 1e-9)) / (B + 1)), "mc_B": int(B), "mc_seed": seed,
           "max_count": int(c.max()), "n_bins_zero": int((c == 0).sum())}
    if np.allclose(t, t[0]):
        out["variance_over_mean"] = float(np.var(c, ddof=1) / np.mean(c)) if np.mean(c) > 0 else None
    return out


def quasi_poisson_ucb(count: int, exposure_h: float, phi: float, alpha: float = 0.05) -> float:
    """Garwood upper bound applied to the effective count N/phi over exposure T/phi (phi >= 1)."""
    phi = max(1.0, float(phi))
    n_eff, t_eff = count / phi, exposure_h / phi
    return float(st.chi2.ppf(1 - alpha, 2 * (n_eff + 1)) / 2) / t_eff


def block_bootstrap_rate_ucb(counts: Sequence[int], expo: Sequence[float], block: int, B: int = 10000,
                             seed: int = 0, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) percentile upper bound of the rate sum(c)/sum(t) under a circular
    moving-block bootstrap over consecutive bins (aitcert.stats.block_bootstrap_indices)."""
    c = np.asarray(counts, float); t = np.asarray(expo, float)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        idx = S.block_bootstrap_indices(len(c), block, rng)
        vals[b] = c[idx].sum() / t[idx].sum()
    return float(np.quantile(vals, 1 - alpha))


# ------------------------------------------------------------------ E5 hourly residual monitor
def hourly_medians(sc: np.ndarray, mask: Optional[np.ndarray] = None, stride: int = 3600) -> List[float]:
    """Copied from e5_drift.py: hourly median of the smoothed score; with a mask, only hours with more
    than 90% normal seconds are used, and the median is taken over the normal seconds."""
    vals = []
    for s0 in range(0, len(sc) - 3600 + 1, stride):
        seg = sc[s0:s0 + 3600]
        if mask is not None and mask[s0:s0 + 3600].mean() <= 0.9:
            continue
        if mask is not None:
            seg = seg[mask[s0:s0 + 3600]]
        vals.append(float(np.nanmedian(seg)))
    return vals


# ------------------------------------------------------------------ paired detection comparison
def paired_degradation(clean: Sequence[int], other: Sequence[int], tau: float = 0.10, alpha: float = 0.05) -> Dict:
    """Recall degradation d = p_clean - p_other on paired detection vectors (same attacks).
    p10 = P(clean detects, other misses), p01 = P(clean misses, other detects), d = p10 - p01.
    Exact one-sided (1-alpha) upper bound: d <= p10, so the Clopper-Pearson upper bound of n10/n bounds d
    (conservative when n01 > 0). Conservative exact lower bound (Bonferroni):
    d >= CP-LCB(n10/n, alpha/2) - CP-UCB(n01/n, alpha/2). Exact one-sided McNemar (conditional binomial)
    p-value for H0: no degradation. Decision: accept if UCB <= tau; reject if LCB > tau; else inconclusive."""
    a = np.asarray(clean, int); b = np.asarray(other, int)
    assert a.shape == b.shape
    n = len(a)
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n11 = int(((a == 1) & (b == 1)).sum()); n00 = int(((a == 0) & (b == 0)).sum())
    d = (n10 - n01) / n
    ucb = S.cp_upper(n10, n, alpha)
    # exact lower bound on d: d = p10 - p01 >= LCB(p10) - UCB(p01) with Bonferroni split (conservative)
    lcb = S.cp_lower(n10, n, alpha / 2) - S.cp_upper(n01, n, alpha / 2)
    m = n10 + n01
    p_mcnemar = float(st.binom.sf(n10 - 1, m, 0.5)) if m > 0 else 1.0
    decision = "accept" if ucb <= tau else ("reject" if lcb > tau else "inconclusive")
    return {"n": n, "n11": n11, "n10_clean_only": n10, "n01_other_only": n01, "n00": n00, "discordant": m,
            "recall_clean": float(a.mean()), "recall_other": float(b.mean()), "degradation_estimate": d,
            "ucb95_one_sided_exact": ucb, "lcb95_one_sided_exact_conservative": lcb, "tau": tau,
            "p_mcnemar_exact_one_sided": p_mcnemar, "decision": decision,
            "method": "UCB(d) = Clopper-Pearson UCB of n10/n (d <= p10); LCB(d) = CP-LCB(n10/n, alpha/2) - CP-UCB(n01/n, alpha/2)"}
