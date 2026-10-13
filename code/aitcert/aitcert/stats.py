"""Probabilistic conformity semantics: statistical acceptance of quantitative requirements.

A quantitative requirement of a service level has the form  m (>=|<=) tau.  Given evidence from
n independent evaluation units, the requirement is ACCEPTED at assurance level G iff
    LCB_{1-alpha_G}(m) >= tau      (for '>=' requirements), or
    UCB_{1-alpha_G}(m) <= tau      (for '<=' requirements),
where LCB/UCB are one-sided confidence bounds. For binomial proportions we use the exact
Clopper-Pearson bounds, which guarantee that the probability of accepting a non-conforming
system (consumer's risk) does not exceed alpha_G.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

ALPHA_BY_G = {1: 0.10, 2: 0.10, 3: 0.10, 4: 0.05, 5: 0.05, 6: 0.01, 7: 0.01}


# ---------------------------------------------------------------- binomial bounds
def cp_lower(k: int, n: int, alpha: float) -> float:
    """One-sided (1-alpha) Clopper-Pearson lower bound for a binomial proportion."""
    if n == 0:
        return 0.0
    return 0.0 if k == 0 else float(stats.beta.ppf(alpha, k, n - k + 1))


def cp_upper(k: int, n: int, alpha: float) -> float:
    """One-sided (1-alpha) Clopper-Pearson upper bound."""
    if n == 0:
        return 1.0
    return 1.0 if k == n else float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def cp_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Two-sided (1-alpha) Clopper-Pearson interval."""
    return cp_lower(k, n, alpha / 2), cp_upper(k, n, alpha / 2)


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- Poisson rate bounds (alarms per hour)
def poisson_upper(count: int, exposure: float, alpha: float) -> float:
    """One-sided (1-alpha) exact (Garwood) upper bound of a Poisson rate count/exposure."""
    return float(stats.chi2.ppf(1 - alpha, 2 * (count + 1)) / 2) / exposure


def poisson_lower(count: int, exposure: float, alpha: float) -> float:
    return 0.0 if count == 0 else float(stats.chi2.ppf(alpha, 2 * count) / 2) / exposure


# ---------------------------------------------------------------- sample sizes
def n_hoeffding(alpha: float, delta: float, beta: Optional[float] = None) -> int:
    """Distribution-free sample size from Hoeffding's inequality.
    beta=None: n >= ln(1/alpha) / (2 delta^2), the n at which the one-sided Hoeffding margin equals delta
    (a system at tau+delta is then accepted with probability of only about 0.5).
    beta given: n >= (sqrt(ln(1/alpha)) + sqrt(ln(1/beta)))^2 / (2 delta^2), which bounds both the consumer's
    risk (alpha) and the producer's risk at tau+delta (beta) without distributional assumptions."""
    if beta is None:
        return int(math.ceil(math.log(1 / alpha) / (2 * delta * delta)))
    return int(math.ceil((math.sqrt(math.log(1 / alpha)) + math.sqrt(math.log(1 / beta))) ** 2 / (2 * delta * delta)))


def n_zero_failure(alpha: float, p_max: float) -> int:
    """Smallest n with zero observed failures that bounds the failure probability by p_max at
    confidence 1-alpha:  n >= ln(alpha) / ln(1 - p_max)  (approx. 3/p_max for alpha = 0.05)."""
    return int(math.ceil(math.log(alpha) / math.log(1 - p_max)))


def n_exact_power(tau: float, delta: float, alpha: float, beta: float = 0.2, n_max: int = 100000) -> int:
    """Smallest n such that a system with true proportion tau+delta passes the exact LCB test
    'cp_lower(X, n, alpha) >= tau' with probability >= 1-beta (producer's risk <= beta)."""
    p = tau + delta
    for n in range(1, n_max + 1):
        kmin = min_successes(n, tau, alpha)
        if kmin is None:
            continue
        if 1 - stats.binom.cdf(kmin - 1, n, p) >= 1 - beta:
            return n
    raise ValueError("n_max exceeded")


def power_exact(n: int, tau: float, delta: float, alpha: float) -> float:
    """Probability that a system with true proportion tau+delta passes the exact LCB test with n units."""
    kmin = min_successes(n, tau, alpha)
    return 0.0 if kmin is None else float(1 - stats.binom.cdf(kmin - 1, n, tau + delta))


def n_exact_power_stable(tau: float, delta: float, alpha: float, beta: float = 0.2, horizon: int = 400) -> int:
    """Smallest n0 such that the exact LCB test has power >= 1-beta at tau+delta for EVERY n >= n0.
    Binomial power is saw-toothed in n, so the first crossing (n_exact_power) can be followed by sample
    sizes with lower power; stability is checked over the next `horizon` sample sizes beyond the last dip."""
    n = n_exact_power(tau, delta, alpha, beta)
    while True:
        dips = [m for m in range(n, n + horizon) if power_exact(m, tau, delta, alpha) < 1 - beta]
        if not dips:
            return n
        n = dips[-1] + 1


def min_successes(n: int, tau: float, alpha: float) -> Optional[int]:
    """Smallest number of successes k such that cp_lower(k, n, alpha) >= tau (None if impossible)."""
    lo, hi = 0, n
    if cp_lower(n, n, alpha) < tau:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if cp_lower(mid, n, alpha) >= tau:
            hi = mid
        else:
            lo = mid + 1
    return lo


def oc_curve(p_grid: Sequence[float], n: int, tau: float, alpha: Optional[float]) -> np.ndarray:
    """Operating-characteristic curve: probability of acceptance as a function of the true
    proportion p for the LCB rule (alpha given) or the naive point-estimate rule (alpha=None)."""
    if alpha is None:
        kmin = int(math.ceil(tau * n - 1e-12))
    else:
        kmin = min_successes(n, tau, alpha)
        if kmin is None:
            return np.zeros(len(p_grid))
    return np.array([1 - stats.binom.cdf(kmin - 1, n, p) for p in p_grid])


# ---------------------------------------------------------------- acceptance decisions
def accept_proportion(k: int, n: int, tau: float, direction: str, alpha: float) -> Dict:
    """Decision for a proportion requirement p (>=|<=) tau using exact one-sided bounds."""
    est = k / n if n else float("nan")
    if direction == ">=":
        bound = cp_lower(k, n, alpha)
        ok = bound >= tau
    else:
        bound = cp_upper(k, n, alpha)
        ok = bound <= tau
    return {"estimate": est, "bound": bound, "tau": tau, "direction": direction, "alpha": alpha,
            "n": n, "k": k, "accepted": bool(ok), "naive_accepted": bool(est >= tau if direction == ">=" else est <= tau)}


def accept_rate(count: int, exposure_h: float, tau: float, alpha: float) -> Dict:
    """Decision for a rate requirement (events per hour <= tau) with an exact Poisson bound."""
    est = count / exposure_h
    ub = poisson_upper(count, exposure_h, alpha)
    return {"estimate": est, "bound": ub, "tau": tau, "direction": "<=", "alpha": alpha, "count": count,
            "exposure_h": exposure_h, "accepted": bool(ub <= tau), "naive_accepted": bool(est <= tau)}


def accept_bootstrap(samples: np.ndarray, point: float, tau: float, direction: str, alpha: float) -> Dict:
    """Decision from bootstrap replicates (percentile one-sided bound)."""
    if direction == ">=":
        bound = float(np.quantile(samples, alpha))
        ok = bound >= tau
        naive = point >= tau
    else:
        bound = float(np.quantile(samples, 1 - alpha))
        ok = bound <= tau
        naive = point <= tau
    return {"estimate": point, "bound": bound, "tau": tau, "direction": direction, "alpha": alpha,
            "B": int(len(samples)), "accepted": bool(ok), "naive_accepted": bool(naive)}


# ---------------------------------------------------------------- resampling for dependent data
def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices (Kunsch 1989; Politis and Romano 1992)."""
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=nb)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def bootstrap(stat: Callable[[np.ndarray], float], units: np.ndarray, B: int = 2000, seed: int = 0) -> np.ndarray:
    """Plain bootstrap over independent evaluation units (e.g., attack events)."""
    rng = np.random.default_rng(seed)
    n = len(units)
    return np.array([stat(units[rng.integers(0, n, n)]) for _ in range(B)])


# ---------------------------------------------------------------- agreement
def agreement_probability(level_probs: Sequence[float]) -> float:
    """Probability that two independent evaluations of the same system assign the same level."""
    p = np.asarray(level_probs, dtype=float)
    return float((p * p).sum())


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
