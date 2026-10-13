"""R7 (E5 of the replication): operating-envelope monitoring (IV-3) with the hourly drift monitor of the frozen plan.

Statistic: share of the readings of the channels shared with HAI 22.04 in a one-hour window that lie outside the range of
the clean HAI 21.03 training data. Threshold: maximum of the statistic over leave-one-file-out training hours and
validation hours, fixed in the plan. Positives: full hours of the six
HAI 22.04 training files (a later campaign of the same testbed). Negatives: hours of the HAI 21.03 test files with more
than 90% normal seconds (attacks excluded from 5 min before to 10 min after), using their normal seconds. Bounds: exact
Clopper-Pearson for hours as units and, for the decision, with the effective sample size n/deff, deff = (1+rho)/(1-rho)
from the lag-1 autocorrelation rho of the hourly statistic within files (at least 1). Writes results/r_e5.json.
Usage: python r7_e5.py"""
import os
import time

import numpy as np
from scipy import stats as st

import r_lib as L
from r_lib import H, S

t0 = time.time()
P = L.require_frozen_plan()
prep = L.load_prep()
arrays, ys = L.test_data(prep)
dm = P["parameters"]["drift_monitor"]
shared = dm["channels"]
thr = float(dm["threshold"])
lo, hi = np.array(dm["lo"], np.float32), np.array(dm["hi"], np.float32)
sh_idx = [prep.cols.index(c) for c in shared]


def psi_stat(seg):
    """Hourly operating-envelope statistic: share of the readings outside the range of the clean training data."""
    return float(np.mean((seg < lo) | (seg > hi)))


def e5_mask(y):
    m = np.ones(len(y), bool)
    for s, e in H.events_from_labels(y):
        m[max(0, s - 300):e + 600] = False
    return m


# negatives: 21.03 test hours
neg = []
for arr, y in zip(arrays, ys):
    m = e5_mask(y)
    neg.append([psi_stat(arr[s:s + 3600][m[s:s + 3600]][:, sh_idx]) for s in range(0, len(arr) - 3600 + 1, 3600)
                if m[s:s + 3600].mean() > 0.9])
# positives: 22.04 training hours, standardised with the 21.03 parameters of the shared channels
mu = prep.mu[sh_idx]; sd = prep.sd[sh_idx]
pos = []
for i in range(1, 7):
    df = H.read_files("hai-22.04", [f"train{i}.csv"])[0]
    a = ((df[shared].to_numpy(dtype=np.float32) - mu) / sd).astype(np.float32)
    pos.append([psi_stat(a[s:s + 3600]) for s in range(0, len(a) - 3600 + 1, 3600)])
    del df, a


def lag1(groups):
    x, y = [], []
    for g in groups:
        g = np.asarray(g, float)
        if len(g) > 1:
            x += g[:-1].tolist(); y += g[1:].tolist()
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0


def block(groups, threshold, tau, direction):
    flags = np.concatenate([np.asarray(g) > threshold for g in groups])
    n, k = len(flags), int(flags.sum())
    rho = lag1(groups)
    deff = max(1.0, (1 + rho) / (1 - rho)) if rho < 1 else float("inf")
    ne, ke = n / deff, k / deff
    lcb_e = 0.0 if ke <= 0 else float(st.beta.ppf(0.05, ke, ne - ke + 1))
    ucb_e = 1.0 if ke >= ne else float(st.beta.ppf(0.95, ke + 1, ne - ke))
    if direction == ">=":
        dec = "accept" if lcb_e >= tau else ("reject" if ucb_e < tau else "inconclusive")
    else:
        dec = "accept" if ucb_e <= tau else ("reject" if lcb_e > tau else "inconclusive")
    return {"n": n, "k": k, "rate": k / n, "lcb_exact": S.cp_lower(k, n, 0.05), "ucb_exact": S.cp_upper(k, n, 0.05),
            "lag1_rho": rho, "deff": deff, "n_eff": ne, "lcb_effn": lcb_e, "ucb_effn": ucb_e, "decision": dec}


res = {}
for name, th in (("calibrated", thr),):
    tpr = block(pos, th, L.TAU["drift_tpr"], ">=")
    fpr = block(neg, th, L.TAU["drift_fpr"], "<=")
    d = [tpr["decision"], fpr["decision"]]
    dec = "reject" if "reject" in d else ("accept" if all(x == "accept" for x in d) else "inconclusive")
    res[name] = {"threshold": th, "tpr": tpr, "fpr": fpr, "decision": dec}
    print(name, "TPR", tpr["k"], "/", tpr["n"], "LCB", round(tpr["lcb_effn"], 3), "| FPR", fpr["k"], "/", fpr["n"],
          "UCB", round(fpr["ucb_effn"], 3), "deff", round(fpr["deff"], 2), "->", dec, flush=True)
allp, alln = np.concatenate(pos), np.concatenate(neg)
r = st.rankdata(np.concatenate([allp, alln]))
auc = float((r[:len(allp)].sum() - len(allp) * (len(allp) + 1) / 2) / (len(allp) * len(alln)))
L.save_json({"threshold_calibrated": thr, "shared_channels": len(shared), "results": res, "auc": auc,
             "pos_values": [list(map(float, g)) for g in pos], "neg_values": [list(map(float, g)) for g in neg],
             "meta": {"script": "r7_e5.py", "runtime_s": time.time() - t0, "finished_utc": L.utc_now()}}, "r_e5.json")
print("AUC", round(auc, 4), "runtime", round(time.time() - t0))
