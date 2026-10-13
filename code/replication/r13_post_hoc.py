"""R13: post-hoc checks requested by the verification of the replication (not part of the frozen plan).

(a) Strictly causal alarm onset: detections, false alarms and median time to detection of all six detectors when the
    alarm starts at the first exceedance at which the episode has lasted 20 s (r_lib.alarm_episodes_causal), compared
    with the alarm logic of the plan, from the stored test scores.
(b) Planning design effect of the hourly drift test: lag-1 autocorrelation of the operating-envelope statistic over the
    training hours (scored, as in the calibration, with the range of the other training files, de-meaned within files),
    the design effect (1 + rho)/(1 - rho) and the decidability of the false-positive test with 60 hours.
Writes results/r_post_hoc.json. Usage: python r13_post_hoc.py"""
import numpy as np

import r_lib as L
from r_lib import FZ, H

P = L.plan()
prep = L.load_prep()
out = {"causal_onset": {}, "drift_planning_deff": {}}

# ---------------------------------------------------------------- (a) causal onset
ys = L.label_vectors()
for tag in [f"baseline_s{s}" for s in L.BASELINE_SEEDS] + ["hardened_s0"]:
    z = np.load(L.os.path.join(L.MODELS, f"{tag}_test_scores.npz"))
    th = float(P["operating_points"][tag]["theta"])
    rec = {}
    for name, fn in (("plan", L.alarm_episodes_k), ("causal", L.alarm_episodes_causal)):
        det = fa = 0
        ttd = []
        for i, y in enumerate(ys, start=1):
            sc = z[f"test{i}"].astype(np.float64)
            ev = H.events_from_labels(y)
            al = fn(sc, th, L.ON_DELAY)
            for s, e in ev:
                hit = [a for a in al if a[0] <= e + H.TOL and a[1] >= s]
                if hit:
                    det += 1
                    ttd.append(max(min(a[0] for a in hit), s) - s)
            fa += sum(1 for a in al if not any(a[0] <= e + H.TOL and a[1] >= s for s, e in ev))
        rec[name] = {"detected": det, "false_alarms": fa, "ttd_median_s": float(np.median(ttd))}
    out["causal_onset"][tag] = rec
    print(tag, rec, flush=True)

# ---------------------------------------------------------------- (b) planning design effect of the drift test
dm = P["parameters"]["drift_monitor"]
idx = [prep.cols.index(c) for c in dm["channels"]]
train = L.read(L.TRAIN)
arrs = [prep.to_array(L.drop_labels(d)) for d in train]
tr, _ = L.split(arrs)
trs = [a[:, idx] for a in tr]
stat = lambda seg, lo, hi: float(np.mean((seg < lo) | (seg > hi)))
hrs = lambda a: [a[s:s + 3600] for s in range(0, len(a) - 3600 + 1, 3600)]
x, y_ = [], []
for k in range(len(trs)):
    rest = np.concatenate([trs[j] for j in range(len(trs)) if j != k])
    v = np.array([stat(h, rest.min(0), rest.max(0)) for h in hrs(trs[k])])
    v = v - v.mean()
    x += v[:-1].tolist()
    y_ += v[1:].tolist()
rho = float(np.corrcoef(x, y_)[0, 1])
deff = (1 + rho) / (1 - rho)
n_neg = L.load_json("r_e5.json")["results"]["calibrated"]["fpr"]["n"]
row = FZ.check_proportion("IV-3 drift FPR (hours, planning design effect from training hours)", n_neg, L.TAU["drift_fpr"], "<=",
                          L.A0, delta=L.DELTA, deff=deff, units="hours")
out["drift_planning_deff"] = {"lag1_rho_training_hours": rho, "deff": deff, "n_hours": n_neg, "check": row}
print("drift planning deff", round(deff, 2), "n_eff", round(n_neg / deff, 1), "verdict", row["verdict"], "best bound", round(row["best_bound"], 3))
L.save_json(out, "r_post_hoc.json")
