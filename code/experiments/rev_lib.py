"""Shared helpers for the major-revision re-analyses (scripts e1b_localisation.py, cluster_bootstrap.py,
im2_sign_v2.py, decision_sensitivity.py, stats_revision.py).

This module only reads the existing artefacts; it never modifies hai_lib.py, aitcert, existing result
files, model files or specifications. Result files are written only under NEW names, and a file that
already exists is overwritten only if it carries the marker of the same generating script.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "results")
MODELS = os.path.join(RES, "models")
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
ATTACKS_CSV = os.path.join(ROOT, "data", "processed", "hai2204_attacks.csv")
MARKER = "generated_by"

# Primitive identifiers as printed in the HAI 22.04 attack table. The table lists A415 with primitive
# "AP3". No primitive is written "AP3"; the scenario (P2-TC-SP2) and target point (P2_VTR02) of A415 are
# exactly those of AP33 (also used by A414) and not those of AP03 (P1-PC-SP1PV1PV2, used by A209).
# We therefore read "AP3" as "AP33" in the primary analysis and keep the literal reading (a separate
# primitive) as a sensitivity check.
PRIMITIVE_FIXES = {"AP3": "AP33"}

# Rows whose primitive / scenario / target point / target process fields are mutually inconsistent in the
# source table (HAI 22.04 technical documentation, attack table, as transcribed in hai2204_attacks.csv).
# They are flagged and used AS RECORDED in the primary analyses (as in e1/e2); never silently corrected.
INCONSISTENT_ROWS = {
    "A301": "primitive AP16 (elsewhere P1-LC-CO1, target P1_LCV01D) is listed with scenario P3-LC-CO2 and target point P2_LCV01D, a tag that does not exist",
    "A304": "scenario P3-LC-CO2PV1 (target process P3) but target points P1_LCV01D, P1_LIT01",
    "A305": "scenario P1-LC-CO1-ST of AP18 (elsewhere target P1_LCV01D) but target point P3_LCV01D",
    "A306": "primitive AP04 (elsewhere P1-PC-CO1, target P1_PCV01D) is listed with target point P1_LCV01D",
    "A415": "primitive printed as 'AP3' (read as AP33, see PRIMITIVE_FIXES)",
    "A417": "target point printed as P1_PP04_SP; the data column is P1_PP04SP",
}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def versions(extra: Iterable[str] = ()) -> Dict[str, str]:
    """Software versions of the interpreter and of the packages used."""
    import numpy, scipy, pandas  # noqa: F401
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__}
    for mod in extra:
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
            if mod == "torch":
                v["torch_cuda"] = str(m.version.cuda)
                v["cudnn"] = str(m.backends.cudnn.version()) if m.cuda.is_available() else "n/a"
                v["gpu"] = m.cuda.get_device_name(0) if m.cuda.is_available() else "cpu"
        except Exception as exc:  # pragma: no cover - informative only
            v[mod] = f"not available ({exc.__class__.__name__})"
    return v


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(o):
    """Make an object JSON-serialisable (numpy scalars/arrays, NaN -> None, sets -> sorted lists)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, set):
        return sorted(_clean(v) for v in o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if (math.isnan(x) or math.isinf(x)) else x
    return o


def jdump(obj: dict, name: str, script: str) -> str:
    """Write results/<name>. Refuses to overwrite a file that was not produced by the same script."""
    path = os.path.join(RES, name)
    if os.path.exists(path):
        try:
            old = json.load(open(path, encoding="utf8"))
            ok = isinstance(old, dict) and old.get(MARKER) == script
        except Exception:
            ok = False
        if not ok:
            raise RuntimeError(f"refusing to overwrite {path}: not produced by {script}")
    obj = dict(obj)
    obj[MARKER] = script
    with open(path, "w", encoding="utf8") as f:
        json.dump(_clean(obj), f, indent=1, ensure_ascii=False)
    return path


# ---------------------------------------------------------------- attack metadata
def load_attacks(fix_primitives: bool = True) -> List[dict]:
    """The 58 HAI 22.04 attacks in file/time order with parsed primitives, target processes and points."""
    import pandas as pd
    df = pd.read_csv(ATTACKS_CSV)
    out = []
    for _, r in df.iterrows():
        prims = [p.strip().upper() for p in str(r.primitives).split(";") if p.strip()]
        if fix_primitives:
            prims = [PRIMITIVE_FIXES.get(p, p) for p in prims]
        procs = sorted(set(p.strip() for p in str(r.target_processes).split(";") if p.strip()))
        pts = sorted(set(t.strip() for g in str(r.target_points).split(";") for t in g.split(",") if t.strip()))
        out.append({"attack_id": r.attack_id, "test_file": r.test_file, "primitives": prims,
                    "target_processes": procs, "target_points": pts,
                    "label_start_idx": int(r.label_start_idx), "label_end_idx": int(r.label_end_idx)})
    return out


def components(units: Sequence[str], prims: Dict[str, Sequence[str]]) -> List[List[str]]:
    """Connected components of the unit graph in which two units are linked if they share a primitive."""
    parent = {u: u for u in units}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for u in units:
        for p in prims[u]:
            if p in first:
                ra, rb = find(u), find(first[p])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[p] = u
    groups: Dict[str, List[str]] = {}
    for u in units:
        groups.setdefault(find(u), []).append(u)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


# ---------------------------------------------------------------- distributions
def poisson_binomial_pmf(q: Sequence[float]) -> np.ndarray:
    """Exact pmf of the number of successes of independent Bernoulli(q_i) trials (DP convolution)."""
    pmf = np.zeros(len(q) + 1)
    pmf[0] = 1.0
    for i, qi in enumerate(q):
        pmf[1:i + 2] = pmf[1:i + 2] * (1 - qi) + pmf[0:i + 1] * qi
        pmf[0] *= (1 - qi)
    return pmf


def pmf_summary(pmf: np.ndarray, n: int) -> dict:
    x = np.arange(len(pmf))
    mean = float((x * pmf).sum())
    sd = float(math.sqrt(max(0.0, ((x - mean) ** 2 * pmf).sum())))
    cdf = np.cumsum(pmf)

    def q(p):
        return int(np.searchsorted(cdf, p - 1e-12))

    return {"expected_hits": mean, "expected_accuracy": mean / n, "sd_hits": sd,
            "quantiles_hits": {"0.05": q(0.05), "0.25": q(0.25), "0.5": q(0.5), "0.75": q(0.75), "0.95": q(0.95)},
            "pmf": [float(v) for v in pmf]}


def tail_ge(pmf: np.ndarray, k: int) -> float:
    """P(X >= k)."""
    return float(pmf[k:].sum()) if k < len(pmf) else 0.0
