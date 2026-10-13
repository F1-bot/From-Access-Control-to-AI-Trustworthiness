"""Shared configuration and helpers for the confirmatory replication on HAI 21.03.

The replication applies the final procedure of the HAI 22.04 case study (code/experiments) to a detector trained
with the same recipes on HAI 21.03, a campaign that was not used to design the criteria, the evaluation plan or the
procedure. Design changes required by the pre-test feasibility check (aitcert.feasibility) are fixed, with the
procedure, in the plan replication_plan_2103.yaml, which is frozen (SHA-256 digest and UTC time in
results/plan_freeze.json) before any script reads the sensor data of the test files. Scripts that evaluate on test
data call require_frozen_plan() first.

hai_lib, adv_lib and revml_lib (code/experiments) are imported and never modified.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "experiments"))
if EXP not in sys.path:
    sys.path.insert(0, EXP)
import hai_lib as H  # noqa: E402
import aitcert_path  # noqa: E402,F401  (puts the aitcert package on the path)
from aitcert import feasibility as FZ  # noqa: E402
from aitcert import stats as S  # noqa: E402

ROOT = H.ROOT
VERSION = "hai-21.03"
TRAIN = ["train1.csv", "train2.csv", "train3.csv"]
TEST = [f"test{i}.csv" for i in range(1, 6)]
ATTACKS_CSV = os.path.join(ROOT, "data", "processed", "hai2103_attacks.csv")
RES = os.path.join(HERE, "results")
MODELS = os.path.join(RES, "models")
PLAN = os.path.join(HERE, "replication_plan_2103.yaml")
FREEZE = os.path.join(RES, "plan_freeze.json")
os.makedirs(MODELS, exist_ok=True)

RECIPE = {"epochs": 12, "per_epoch": 250_000}           # run_all.sh recipe of the rated 22.04 detectors
E3_RECIPE = {"epochs": 10, "per_epoch": 200_000}        # run_all.sh recipe of the 22.04 poisoning drill
HARDENED_CORRUPT = 0.3
BASELINE_SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ("baseline_s0", "hardened_s0")
A0 = 0.05
ALPHAS = (0.10, 0.05, 0.01)
DELTA = 0.2
BETA = 0.2
TAU = {"recall": 0.50, "far": 0.25, "load": 6.0, "asr": 0.20, "pdet": 0.80, "pdet_fpr": 0.05, "pdeg": 0.10,
       "ext": 0.90, "xfid": 0.60, "drift_tpr": 0.90, "drift_fpr": 0.05}


# ---------------------------------------------------------------- bookkeeping
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, set):
        return sorted(_clean(v) for v in o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if math.isfinite(x) else None
    return o


def save_json(obj: Dict, name: str) -> str:
    path = os.path.join(RES, name)
    with open(path, "w", encoding="utf8") as f:
        json.dump(_clean(obj), f, indent=1, ensure_ascii=False)
    return path


def load_json(name: str) -> Dict:
    with open(os.path.join(RES, name), encoding="utf8") as f:
        return json.load(f)


def versions() -> Dict[str, str]:
    import platform
    import scipy
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "torch": torch.__version__, "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


# ---------------------------------------------------------------- data
def drop_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not c.lower().startswith("attack")]]


def read(names: Sequence[str]) -> List[pd.DataFrame]:
    return H.read_files(VERSION, names)


def fit_prep(train: List[pd.DataFrame]) -> "H.Prep":
    return H.fit_prep([drop_labels(d) for d in train])


def load_prep() -> "H.Prep":
    return H.Prep.load(os.path.join(MODELS, "prep.json"))


def split(arrs: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Validation = last 20% (time-contiguous) of the last training file, as in the 22.04 study."""
    cut = int(len(arrs[-1]) * 0.8)
    return arrs[:-1] + [arrs[-1][:cut]], [arrs[-1][cut:]]


def label_vectors() -> List[np.ndarray]:
    """Attack label column of each test file (the published attack schedule); no sensor data are read."""
    out = []
    for n in TEST:
        p = os.path.join(H.DATA, VERSION, n)
        p = p if os.path.exists(p) else p + ".gz"
        out.append(pd.read_csv(p, usecols=["attack"])["attack"].to_numpy().astype(np.int8))
    return out


def load_attacks() -> List[Dict]:
    df = pd.read_csv(ATTACKS_CSV)
    out = []
    for _, r in df.iterrows():
        prims = [p.strip().upper() for p in str(r.primitives).split(";") if p.strip()]
        procs = sorted(set(p.strip() for p in str(r.target_processes).split(";") if p.strip()))
        pts = sorted(set(t.strip() for g in str(r.target_points).split(";") for t in g.split(",") if t.strip()))
        out.append({"attack_id": r.attack_id, "test_file": r.test_file, "primitives": prims, "target_processes": procs,
                    "target_points": pts, "label_start_idx": int(r.label_start_idx), "label_end_idx": int(r.label_end_idx)})
    return out


def dependency_matrix(ids: Sequence[str], prims: Dict[str, Sequence[str]]) -> np.ndarray:
    """D_ij = 1 if attacks i and j share a primitive (diagonal included)."""
    return np.array([[1.0 if set(prims[a]) & set(prims[b]) else 0.0 for b in ids] for a in ids])


def graph_deff(y: np.ndarray, D: np.ndarray) -> float:
    """Dependency-graph design effect of a mean of y (Section 3.7.5), truncated at 1."""
    r = np.asarray(y, float) - np.mean(y)
    den = float(r @ r)
    return max(1.0, float(r @ D @ r) / den) if den > 0 else 1.0


# ---------------------------------------------------------------- frozen plan
def plan() -> Dict:
    with open(PLAN, encoding="utf8") as f:
        return yaml.safe_load(f)


def require_frozen_plan() -> Dict:
    """Refuse to evaluate on test data unless the plan file matches the digest recorded when it was frozen."""
    if not os.path.exists(FREEZE):
        raise RuntimeError("the replication plan has not been frozen (results/plan_freeze.json missing)")
    fz = load_json("plan_freeze.json")
    dig = sha256_file(PLAN)
    if dig != fz["plan_sha256"]:
        raise RuntimeError(f"plan digest {dig} differs from the frozen digest {fz['plan_sha256']}")
    return plan()


def test_data(prep) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    require_frozen_plan()
    dfs = read(TEST)
    return [prep.to_array(drop_labels(d)) for d in dfs], [H.labels(d) for d in dfs]


def load_model(tag: str, f: int):
    from safetensors.torch import load_file
    m = H.Forecaster(f).to(H.DEV)
    m.load_state_dict(load_file(os.path.join(MODELS, f"{tag}.safetensors")))
    m.eval()
    return m


def chanvar(tag: str) -> np.ndarray:
    return np.load(os.path.join(MODELS, f"{tag}_chanvar.npy"))


def theta(tag: str) -> float:
    """Operating point fixed in the frozen plan."""
    return float(require_frozen_plan()["operating_points"][tag]["theta"])


# ---------------------------------------------------------------- alarm logic with an on-delay (ISA-18.2)
ON_DELAY = 20      # s; fixed in the plan (smallest on-delay in 10-s steps with which both 22.04 detectors met the false-alarm limit)


def alarm_episodes_k(score: np.ndarray, theta: float, k: int) -> List[Tuple[int, int]]:
    """Alarm episodes with an on-delay of k seconds applied to exceedance episodes: exceedances of theta closer than
    MERGE_GAP form one episode (as hai_lib.alarm_episodes); an alarm is raised once an episode has lasted k seconds and
    ends with the episode. The onset is placed at the k-th second of the episode even if the score lies in a merged gap
    at that second, which needs the knowledge that the episode continues; alarm_episodes_causal places it at the first
    exceedance from that second on. k = 1 reproduces hai_lib.alarm_episodes. This is the logic of the frozen plan."""
    return [(s + k - 1, e) for s, e in H.alarm_episodes(score, theta) if e - s + 1 >= k]


def alarm_episodes_causal(score: np.ndarray, theta: float, k: int) -> List[Tuple[int, int]]:
    """Strictly causal variant of alarm_episodes_k (sensitivity analysis, not part of the frozen plan): the alarm starts
    at the first exceedance at which the episode has lasted at least k seconds."""
    a = np.nan_to_num(score, nan=-np.inf) > theta
    out = []
    for s, e in H.alarm_episodes(score, theta):
        if e - s + 1 >= k:
            idx = np.flatnonzero(a[s + k - 1:e + 1])
            out.append((s + k - 1 + int(idx[0]), e))
    return out


def alarm_in_window_causal(sm: np.ndarray, theta: float, k: int, in_win: np.ndarray) -> bool:
    return any(bool(np.any(in_win[s:e + 1])) for s, e in alarm_episodes_causal(sm, theta, k))


def evaluate_file_k(score: np.ndarray, y: np.ndarray, theta: float, k: int) -> Dict:
    """hai_lib.evaluate_file with the on-delay alarm logic."""
    ev = H.events_from_labels(y)
    al = alarm_episodes_k(score, theta, k)
    det, ttd = [], []
    for s, e in ev:
        hit = [a for a in al if a[0] <= e + H.TOL and a[1] >= s]
        det.append(bool(hit))
        ttd.append(max(min(a[0] for a in hit), s) - s if hit else None)
    fa = [a for a in al if not any(a[0] <= e + H.TOL and a[1] >= s for s, e in ev)]
    attack_time = sum(min(len(y), e + H.TOL + 1) - s for s, e in ev)
    normal_h = (len(y) - H.W - attack_time) / 3600.0
    return {"events": ev, "detected": det, "ttd": ttd, "alarms": al, "false_alarms": len(fa), "normal_hours": normal_h,
            "n_alarms": len(al), "hours": len(y) / 3600.0, "point_tp": 0, "point_fp": 0, "point_fn": 0}


def alarm_in_window(sm: np.ndarray, theta: float, k: int, in_win: np.ndarray) -> bool:
    """True if the deployed alarm logic raises an alarm that overlaps the detection window, given the smoothed score sm
    of a segment (in_win marks the window [start, end + TOL]); episodes may begin before the window."""
    return any(bool(np.any(in_win[s:e + 1])) for s, e in alarm_episodes_k(sm, theta, k))


# ---------------------------------------------------------------- operating point
def episodes_per_threshold(s_val: np.ndarray, thetas: Sequence[float], k: int = 1) -> List[int]:
    return [len(alarm_episodes_k(s_val, th, k)) for th in thetas]


def smallest_theta_with_at_most(s_val: np.ndarray, max_episodes: int, lo: float, k: int = 1) -> float:
    """Smallest threshold >= lo whose number of validation alarm episodes (on-delay k) is at most max_episodes."""
    finite = np.sort(s_val[np.isfinite(s_val)])
    cand = np.unique(finite[finite >= lo])
    if len(alarm_episodes_k(s_val, lo, k)) <= max_episodes:
        return float(lo)
    a, b = 0, len(cand) - 1          # episodes are non-increasing in theta: binary search over candidates
    while a < b:
        m = (a + b) // 2
        if len(alarm_episodes_k(s_val, float(cand[m]), k)) <= max_episodes:
            b = m
        else:
            a = m + 1
    return float(cand[a])
