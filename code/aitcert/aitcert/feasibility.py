"""Pre-test feasibility and power check of an evaluation plan (assurance row TE-AI-1).

Before any test data are seen, every quantitative test of a plan is checked for four properties.

1. Decidability. Can the decision rule of Section 3.4 accept at all with the planned units? With n units
   the best attainable bound is LCB(n, n) = alpha^(1/n) for a requirement m >= tau and
   UCB(0, n) = 1 - alpha^(1/n) for m <= tau; for an event rate with exposure T it is -ln(alpha)/T.
2. Power. The probability that a conforming system at the declared margin delta is accepted, the sample
   size n_min(alpha, delta) of Section 3.4 and, without a declared margin, the minimum detectable margin
   at power 1 - beta.
3. Prediction. The acceptance probability at a pilot estimate that is available before testing, for
   example the false-alarm rate on validation data.
4. Admissibility. tau must lie beyond the value that a declared trivial baseline achieves (principle P5).

Dependent units enter through a planning design effect deff, so that n_eff = n / deff; power is computed
with floor(n_eff) units.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from scipy import stats

from . import stats as S

VERDICTS = ("inadmissible threshold", "not decidable", "underpowered", "low predicted acceptance", "feasible")


# ---------------------------------------------------------------- proportions
def best_bound(n_eff: float, direction: str, alpha: float) -> float:
    """Best attainable one-sided bound with n_eff units: alpha^(1/n) for '>=' and 1 - alpha^(1/n) for '<='."""
    if n_eff <= 0:
        return 0.0 if direction == ">=" else 1.0
    b = alpha ** (1.0 / n_eff)
    return b if direction == ">=" else 1.0 - b


def acceptance_limit(n: int, tau: float, direction: str, alpha: float) -> Optional[int]:
    """Smallest k accepted for '>=' (largest k accepted for '<='); None if no outcome is accepted."""
    if direction == ">=":
        return S.min_successes(n, tau, alpha)
    ks = [k for k in range(n + 1) if S.cp_upper(k, n, alpha) <= tau]
    return max(ks) if ks else None


def p_accept_proportion(p: float, n: int, tau: float, direction: str, alpha: float) -> float:
    """Probability that a system with true proportion p is accepted with n independent units."""
    lim = acceptance_limit(n, tau, direction, alpha)
    if lim is None:
        return 0.0
    if direction == ">=":
        return float(stats.binom.sf(lim - 1, n, p))
    return float(stats.binom.cdf(lim, n, p))


def n_min(tau: float, direction: str, delta: float, alpha: float, beta: float = 0.2) -> int:
    """Stable exact sample size of Section 3.4; '<=' requirements are mirrored."""
    t = tau if direction == ">=" else 1.0 - tau
    if t + delta >= 1.0:
        return S.n_zero_failure(alpha, 1.0 - t)
    return S.n_exact_power_stable(t, delta, alpha, beta)


def min_detectable_margin(n: int, tau: float, direction: str, alpha: float, beta: float = 0.2,
                          step: float = 0.001) -> Optional[float]:
    """Smallest margin delta at which a system at tau +/- delta is accepted with probability >= 1 - beta."""
    span = (1.0 - tau) if direction == ">=" else tau
    k = 1
    while k * step <= span + 1e-12:
        d = k * step
        p = tau + d if direction == ">=" else tau - d
        if p_accept_proportion(min(1.0, max(0.0, p)), n, tau, direction, alpha) >= 1 - beta:
            return round(d, 6)
        k += 1
    return None


def check_proportion(name: str, n: float, tau: float, direction: str, alpha: float, *, delta: Optional[float] = None,
                     deff: float = 1.0, pilot: Optional[float] = None, baseline: Optional[float] = None,
                     beta: float = 0.2, units: str = "units") -> Dict:
    """Feasibility of a proportion requirement m (>=|<=) tau planned with n units."""
    deff = max(1.0, float(deff))
    n_eff = n / deff
    n_pow = int(math.floor(n_eff + 1e-9))
    bb = best_bound(n_eff, direction, alpha)
    attainable = bb >= tau if direction == ">=" else bb <= tau
    row = {"test": name, "kind": "proportion", "tau": tau, "direction": direction, "alpha": alpha, "units": units,
           "n": n, "deff": deff, "n_eff": n_eff, "best_bound": bb, "decidable": bool(attainable)}
    flags = []
    if baseline is not None:
        row["trivial_baseline"] = baseline
        ok = baseline < tau if direction == ">=" else baseline > tau
        row["threshold_admissible"] = bool(ok)
        if not ok:
            flags.append("inadmissible threshold")
    if not attainable:
        flags.append("not decidable")
    if delta is not None:
        row["delta"] = delta
        row["n_min"] = n_min(tau, direction, delta, alpha, beta)
        p_alt = tau + delta if direction == ">=" else tau - delta
        row["power_at_delta"] = p_accept_proportion(min(1.0, max(0.0, p_alt)), n_pow, tau, direction, alpha) if n_pow > 0 else 0.0
        if attainable and row["power_at_delta"] < 1 - beta:
            flags.append("underpowered")
    row["min_detectable_margin"] = min_detectable_margin(n_pow, tau, direction, alpha, beta) if n_pow > 0 and attainable else None
    if pilot is not None:
        row["pilot"] = pilot
        row["p_accept_at_pilot"] = p_accept_proportion(min(1.0, max(0.0, pilot)), n_pow, tau, direction, alpha) if n_pow > 0 else 0.0
        if attainable and row["p_accept_at_pilot"] < 1 - beta:
            flags.append("low predicted acceptance")
    row["flags"] = flags
    row["verdict"] = next((v for v in VERDICTS if v in flags), "feasible")
    return row


# ---------------------------------------------------------------- event rates
def p_accept_rate(lam: float, exposure: float, tau: float, alpha: float) -> float:
    """Probability that a Poisson process with rate lam is accepted (Garwood UCB <= tau) after `exposure`."""
    if S.poisson_upper(0, exposure, alpha) > tau:
        return 0.0
    c = 0
    while S.poisson_upper(c + 1, exposure, alpha) <= tau:
        c += 1
    return float(stats.poisson.cdf(c, lam * exposure))


def max_rate_at_power(exposure: float, tau: float, alpha: float, beta: float = 0.2) -> Optional[float]:
    """Largest true rate that is still accepted with probability >= 1 - beta (None if none is)."""
    if p_accept_rate(0.0, exposure, tau, alpha) < 1 - beta:
        return None
    lo, hi = 0.0, tau
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if p_accept_rate(mid, exposure, tau, alpha) >= 1 - beta:
            lo = mid
        else:
            hi = mid
    return lo


def exposure_for_rate(lam: float, tau: float, alpha: float, beta: float = 0.2, step: float = 1.0,
                      t_max: float = 100000.0) -> Optional[float]:
    """Smallest exposure (in steps of `step`) after which a process with rate lam is accepted with probability
    >= 1 - beta for every longer exposure checked up to twice that value."""
    t = step
    while t <= t_max:
        if p_accept_rate(lam, t, tau, alpha) >= 1 - beta:
            if all(p_accept_rate(lam, t + j * step, tau, alpha) >= 1 - beta for j in range(1, int(t / step) + 1)):
                return t
        t += step
    return None


def check_rate(name: str, exposure: float, tau: float, alpha: float, *, pilot_rate: Optional[float] = None,
               beta: float = 0.2, units: str = "hours") -> Dict:
    """Feasibility of a rate requirement (events per unit exposure) <= tau with the planned exposure."""
    bb = S.poisson_upper(0, exposure, alpha)
    attainable = bb <= tau
    row = {"test": name, "kind": "rate", "tau": tau, "direction": "<=", "alpha": alpha, "units": units,
           "exposure": exposure, "best_bound": bb, "decidable": bool(attainable),
           "max_rate_at_power": max_rate_at_power(exposure, tau, alpha, beta) if attainable else None}
    flags = [] if attainable else ["not decidable"]
    if pilot_rate is not None:
        row["pilot"] = pilot_rate
        row["p_accept_at_pilot"] = p_accept_rate(pilot_rate, exposure, tau, alpha)
        if attainable and row["p_accept_at_pilot"] < 1 - beta:
            flags.append("low predicted acceptance")
    row["flags"] = flags
    row["verdict"] = next((v for v in VERDICTS if v in flags), "feasible")
    return row


def summarise(rows: List[Dict]) -> Dict[str, List[str]]:
    """Tests grouped by verdict."""
    out: Dict[str, List[str]] = {v: [] for v in VERDICTS}
    for r in rows:
        out[r["verdict"]].append(r["test"])
    return out
