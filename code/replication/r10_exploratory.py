"""R10: exploratory analyses after the confirmatory run (not part of the frozen plan; reported as exploratory).

(a) Operating curve of the planned sanitiser of the poisoning drill: the plan fixed its threshold at the 99.5th percentile of
    the trusted model's validation window scores, which removed far fewer clean windows than the 5% that ID-3 allows. Here
    the threshold is set at lower validation quantiles; for each, the clean removal and the poison detection (pooled, with
    the 20 poisoned traces as units, percentile cluster bootstrap) are reported.
(b) HAI 22.04 with the alarm logic of the replication (20-s on-delay at the planned thresholds), from the pilot stored in
    the frozen plan: detections and false alarms with exact bounds (post hoc for the 22.04 study).
Writes results/r_exploratory.json. Usage: python r10_exploratory.py"""
import math
import time

import numpy as np

import r_lib as L
from r_lib import H, S

t0 = time.time()
P = L.require_frozen_plan()
prep = L.load_prep()
F = len(prep.cols)
arrays, ys = L.test_data(prep)
attacks = L.load_attacks()
train = L.read(L.TRAIN)
arrs = [prep.to_array(L.drop_labels(d)) for d in train]
clean, val = L.split(arrs)
src_file = P["parameters"]["poison_source_file"]
fi = int(src_file[-1]) - 1
src = [a for a in attacks if a["test_file"] == src_file]
segs = [arrays[fi][a["label_start_idx"] - H.W:a["label_end_idx"] + 1] for a in src]
trusted = L.load_model("sanitiser_trusted", F)
tcv = np.load(L.os.path.join(L.MODELS, "sanitiser_trusted_chanvar.npy"))
v_scores = H.window_scores(trusted, val, H.window_index(val), tcv)
c_scores = H.window_scores(trusted, clean, H.window_index(clean), tcv)
p_scores = [H.window_scores(trusted, [sg], H.window_index([sg]), tcv) for sg in segs]   # one replica per trace suffices
win = np.array([len(p) for p in p_scores], float)
curve = []
for q in (0.995, 0.99, 0.98, 0.97, 0.96, 0.95):
    thr = float(np.quantile(v_scores, q))
    rem = np.array([(p > thr).sum() for p in p_scores], float)
    rng = np.random.default_rng(0)
    boot = np.empty(10000)
    for b in range(len(boot)):
        k = rng.integers(0, len(rem), len(rem))
        boot[b] = rem[k].sum() / win[k].sum()
    cr = float(np.mean(c_scores > thr))
    curve.append({"validation_quantile": q, "threshold": thr, "clean_removal": cr,
                  "clean_removal_ucb95": S.cp_upper(int(round(cr * len(c_scores))), len(c_scores), 0.05),
                  "poison_detection": float(rem.sum() / win.sum()), "poison_detection_lcb95": float(np.quantile(boot, 0.05)),
                  "traces_with_majority_removed": int(sum((rem / win) > 0.5))})
    print(curve[-1], flush=True)
pilot = P["alarm_logic"]["pilot_hai2204"][str(L.ON_DELAY)]
p2204 = {t: {**r, "far_per_h": r["false_alarms"] / r["normal_hours"], "recall": r["detected"] / r["attacks"],
             "recall_cp95": list(S.cp_interval(r["detected"], r["attacks"], 0.05))} for t, r in pilot.items()}
L.save_json({"sanitiser_operating_curve": curve, "hai2204_with_on_delay_post_hoc": p2204,
             "note": "exploratory; not part of the frozen plan",
             "meta": {"script": "r10_exploratory.py", "runtime_s": time.time() - t0, "finished_utc": L.utc_now()}}, "r_exploratory.json")
print("runtime", round(time.time() - t0))
