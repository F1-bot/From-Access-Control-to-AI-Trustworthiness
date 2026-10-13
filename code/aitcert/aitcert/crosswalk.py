"""Coverage of external framework requirements by sets of criteria targets.
Cov_w(F | X) = (1/|R_F|) * sum_{r in R_F} max_{x in X} s(r, x), with s = 1 (F) or w (P), R_F excluding
units not applicable to system evaluation; a priority-weighted variant uses unit weights."""
from typing import Dict, Iterable, Optional


def unit_score(m: Optional[Dict[str, float]], X: set, w_partial: float = 0.5, strict: bool = False) -> Optional[float]:
    if m is None:
        return None
    best = 0.0
    for t, s in m.items():
        if t in X:
            v = 1.0 if s >= 1.0 else (0.0 if strict else w_partial)
            best = max(best, v)
    return best


def coverage(mapping: Dict[str, Optional[Dict[str, float]]], X: Iterable[str], w_partial: float = 0.5,
             strict: bool = False, weights: Optional[Dict[str, float]] = None) -> Dict:
    X = set(X)
    num = den = 0.0
    full = addressed = n = 0
    for r, m in mapping.items():
        s = unit_score(m, X, w_partial, strict)
        if s is None:
            continue
        w = 1.0 if weights is None else weights.get(r, 1.0)
        num += w * s; den += w; n += 1
        full += s >= 1.0; addressed += s > 0
    return {"n": n, "weighted": num / den if den else float("nan"), "full": full / n if n else float("nan"),
            "addressed": addressed / n if n else float("nan")}
