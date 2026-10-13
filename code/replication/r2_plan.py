"""R2: pre-test feasibility check, operating points, drift-monitor calibration, and freezing of the replication plan.

Inputs that exist before any test data are read: the trained detectors and their validation scores (r1_train.py),
the published attack schedule of HAI 21.03 (data/processed/hai2103_attacks.csv, aligned with the label column of
the test files), the label columns themselves, and the training data. For HAI 22.04 the script reads only the
column names of the training files (to find the channels shared with HAI 21.03) and, for the retrospective check
of the original plan, result files of the 22.04 study that contain pre-test information (validation alarms, the
attack schedule, the numbers of drift days and poisoned traces).

Steps
  1. Operating points. The rated recipe sets the alarm threshold at the 99.9th percentile of the smoothed validation
     score. The false-alarm test is predicted to pass with probability >= 0.8 only if the true rate is at most
     lambda* = FZ.max_rate_at_power(T_test, 0.25, 0.05). Rule fixed here: keep the 99.9th percentile if its validation
     alarm rate is at most lambda*; otherwise raise the threshold to the smallest value with at most
     floor(lambda* x T_val) validation alarm episodes.
  2. Drift monitor (IV-3). Daily windows are not decidable (fewer than 59 in-distribution days). Hourly windows:
     median over the shared channels of the per-channel PSI against the validation reference (as E5). The fixed
     threshold 0.25 is checked on the in-distribution training hours (pilot false-positive rate); the threshold used
     is the maximum hourly statistic over those hours.
  3. Poisoning drill (ID-3). Poison sources: the 20 attacks of test2 (at least 14 traces are needed for a decidable
     detection test); poison segments are restricted to the attack interval plus the W-second input history, so that
     every poisoned window contains attack samples; degradation is measured on the other 30 attacks.
  4. Feasibility rows (aitcert.feasibility) for the replication plan and, retrospectively, for the original 22.04 plan.
  5. The plan is written to replication_plan_2103.yaml, its SHA-256 and the UTC time to results/plan_freeze.json.
Usage: python r2_plan.py"""
import math
import os
import time

import numpy as np
import pandas as pd
import yaml

import r_lib as L
from r_lib import FZ, H, S

import sys
DRY = "--dry" in sys.argv
t0 = time.time()
if os.path.exists(L.FREEZE):
    raise RuntimeError("the plan is already frozen; refusing to overwrite it")
A, B = L.A0, L.BETA
prep = L.load_prep()
F = len(prep.cols)
attacks = L.load_attacks()
ys = L.label_vectors()

# ---------------------------------------------------------------- exposures from the schedule (labels only)
def e1_normal_hours(y):
    T = len(y)
    m = np.ones(T, bool); m[:H.W] = False
    for s, e in H.events_from_labels(y):
        m[s:min(T, e + H.TOL + 1)] = False
    return m.sum() / 3600.0

def e5_mask(y):
    m = np.ones(len(y), bool)
    for s, e in H.events_from_labels(y):
        m[max(0, s - 300):e + 600] = False
    return m

T_test = float(sum(e1_normal_hours(y) for y in ys))
T_total = float(sum(len(y) for y in ys) / 3600.0)
lam_star = FZ.max_rate_at_power(T_test, L.TAU["far"], A, B)
print("test normal hours", round(T_test, 3), "total", round(T_total, 3), "lambda*", round(lam_star, 4), flush=True)

# ---------------------------------------------------------------- 0. alarm on-delay from the HAI 22.04 data (pilot campaign)
# ISA-18.2 recommends on-delays against fleeting alarms. Here the on-delay applies to exceedance episodes (exceedances closer
# than 30 s merged, as in the alarm logic of the 22.04 study): an alarm is raised once an episode has lasted k seconds. k is the
# smallest value in 10-s steps with which both rated 22.04 detectors, at their planned thresholds, met the false-alarm requirement
# on the 22.04 test data.
R22 = os.path.join(L.EXP, "results")
y22 = [H.labels(df) for df in H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])]
pilot_k = {}
for k in (1, 10, 20, 30, 40, 50, 60):
    row = {}
    for t in L.VARIANTS:
        z = np.load(os.path.join(R22, "models", f"{t}_scores.npz"))
        th22 = L.json.load(open(os.path.join(R22, f"e1_{t}.json")))["theta"]
        evs = [L.evaluate_file_k(z[f"test{i}"].astype(np.float64), y22[i - 1], th22, k) for i in range(1, 5)]
        fa, nh = sum(e["false_alarms"] for e in evs), sum(e["normal_hours"] for e in evs)
        row[t] = {"detected": int(sum(sum(e["detected"]) for e in evs)), "attacks": int(sum(len(e["events"]) for e in evs)),
                  "false_alarms": fa, "normal_hours": nh, "far_ucb95": S.poisson_upper(fa, nh, A),
                  "val_episodes": len(L.alarm_episodes_k(z["val"].astype(np.float64), th22, k))}
    pilot_k[str(k)] = row
    print("on-delay pilot 22.04 k", k, {t: (r["detected"], r["false_alarms"], round(r["far_ucb95"], 3)) for t, r in row.items()}, flush=True)
k_sel = next(int(k) for k, r in pilot_k.items() if int(k) % 10 == 0 and all(x["far_ucb95"] <= L.TAU["far"] for x in r.values()))
assert k_sel == L.ON_DELAY, (k_sel, L.ON_DELAY)

# ---------------------------------------------------------------- 1. operating points
tags = [f"baseline_s{s}" for s in L.BASELINE_SEEDS] + ["hardened_s0"]
ops = {}
for tag in tags:
    tj = L.load_json(f"train_{tag}.json")
    s_val = np.load(os.path.join(L.MODELS, f"{tag}_val_scores.npz"))["val"].astype(np.float64)
    T_val = tj["val_hours"]
    th999 = tj["thetas_by_quantile"]["0.999"]
    ep999 = len(L.alarm_episodes_k(s_val, th999, L.ON_DELAY))
    ep999_nodelay = len(L.alarm_episodes_k(s_val, th999, 1))
    max_ep = int(math.floor(lam_star * T_val + 1e-9))
    if ep999 / T_val <= lam_star:
        th, rule = th999, "99.9th percentile kept (validation rate with the on-delay <= lambda*)"
    else:
        th = L.smallest_theta_with_at_most(s_val, max_ep, th999, L.ON_DELAY)
        rule = f"raised to the smallest threshold with at most {max_ep} validation alarm episodes (with the on-delay)"
    ep = len(L.alarm_episodes_k(s_val, th, L.ON_DELAY))
    al_h = ep / T_val
    ops[tag] = {"theta": float(th), "theta_99.9": float(th999), "rule_applied": rule, "val_hours": T_val, "on_delay_s": L.ON_DELAY,
                "val_episodes_99.9_without_on_delay": ep999_nodelay, "val_episodes_99.9": ep999, "val_episodes": ep, "pilot_far_per_h": al_h,
                "pilot_p_accept_far": FZ.p_accept_rate(al_h, T_test, L.TAU["far"], A),
                "pilot_p_accept_far_99.9": FZ.p_accept_rate(ep999 / T_val, T_test, L.TAU["far"], A),
                "pilot_p_accept_far_99.9_without_on_delay": FZ.p_accept_rate(ep999_nodelay / T_val, T_test, L.TAU["far"], A)}
    print(tag, ops[tag], flush=True)

# ---------------------------------------------------------------- 2. drift monitor calibration (training hours only)
cols22 = [c.strip() for c in pd.read_csv(os.path.join(H.DATA, "hai-22.04", "train1.csv"), nrows=0).columns]
shared = [c for c in prep.cols if c in cols22]
sh_idx = [prep.cols.index(c) for c in shared]
train = L.read(L.TRAIN)
arrs = [prep.to_array(L.drop_labels(d)) for d in train]
tr, val = L.split(arrs)
ref = val[0][:, sh_idx]
edges = [np.unique(np.quantile(ref[:, j], np.linspace(0, 1, 11))) for j in range(len(sh_idx))]

def psi_stat(seg):
    vals = []
    for j, e in enumerate(edges):
        if len(e) < 3:
            continue
        r = np.histogram(np.clip(ref[:, j], e[0], e[-1]), e)[0] / len(ref)
        q = np.histogram(np.clip(seg[:, j], e[0], e[-1]), e)[0] / len(seg)
        r, q = np.clip(r, 1e-4, None), np.clip(q, 1e-4, None)
        vals.append(float(((q - r) * np.log(q / r)).sum()))
    return float(np.median(vals))

cal = [psi_stat(a[s:s + 3600][:, sh_idx]) for a in tr for s in range(0, len(a) - 3600 + 1, 3600)]
psi_thr = float(np.max(cal))
pilot_fpr_fixed = float(np.mean(np.array(cal) > 0.25))
n_used_edges = sum(len(e) >= 3 for e in edges)
# hourly units available in the test files (E5 convention) and in the 22.04 drift data (full hours)
n_neg_h = sum(1 for y in ys for s in range(0, len(y) - 3600 + 1, 3600) if e5_mask(y)[s:s + 3600].mean() > 0.9)
n_neg_d = sum(1 for y in ys for s in range(0, len(y) - 86400 + 1, 86400) if e5_mask(y)[s:s + 86400].mean() > 0.9)
len22 = [sum(1 for _ in open(os.path.join(H.DATA, "hai-22.04", f"train{i}.csv"), "rb")) - 1 for i in range(1, 7)]
n_pos_h = int(sum(n // 3600 for n in len22))
print("PSI channels", len(shared), "used", n_used_edges, "cal hours", len(cal), "thr", round(psi_thr, 4),
      "pilot FPR at 0.25", round(pilot_fpr_fixed, 3), "neg hours", n_neg_h, "neg days", n_neg_d, "pos hours", n_pos_h, flush=True)
# operating-envelope monitor, which replaces the PSI monitor: share of channel readings in a one-hour window that lie
# outside the range of the clean training data; threshold = maximum over leave-one-file-out training hours and validation hours
trs = [a[:, sh_idx] for a in tr]
env_lo, env_hi = np.concatenate(trs).min(0), np.concatenate(trs).max(0)
env_stat = lambda seg, lo, hi: float(np.mean((seg < lo) | (seg > hi)))
hrs = lambda a: [a[s:s + 3600] for s in range(0, len(a) - 3600 + 1, 3600)]
lofo = []
for k in range(len(trs)):
    rest = np.concatenate([trs[j] for j in range(len(trs)) if j != k])
    lofo += [env_stat(h, rest.min(0), rest.max(0)) for h in hrs(trs[k])]
val_env = [env_stat(h, env_lo, env_hi) for h in hrs(val[0][:, sh_idx])]
env_thr = float(max(max(lofo), max(val_env)))
# drift pilot: HAI 23.05 normal data (a later campaign; not used otherwise in the replication)
cols23 = [c.strip() for c in pd.read_csv(os.path.join(H.DATA, "hai-23.05", "hai-train1.csv"), nrows=0).columns]
sh23 = [c for c in shared if c in cols23]
j23 = [shared.index(c) for c in sh23]
i23 = [prep.cols.index(c) for c in sh23]
pilot23 = []
for i in range(1, 5):
    d23 = H.read_files("hai-23.05", [f"hai-train{i}.csv"])[0]
    a23 = (d23[sh23].to_numpy(np.float32) - prep.mu[i23]) / prep.sd[i23]
    pilot23 += [env_stat(h, env_lo[j23], env_hi[j23]) for h in hrs(a23)]
    del d23, a23
pilot_tpr_23 = float(np.mean(np.array(pilot23) > env_thr))
print("envelope monitor thr", round(env_thr, 5), "LOFO hours", len(lofo), "val hours", len(val_env), "pilot 23.05 hours", len(pilot23),
      "TPR", pilot_tpr_23, "min", round(float(np.min(pilot23)), 4), flush=True)

# ---------------------------------------------------------------- 3. poisoning drill design (schedule and training data only)
# The planned sanitiser of E3 (trusted model on train1, seed 100, E3 recipe; threshold = 99.5th percentile of its validation
# window scores) is trained here, before the freeze, and saved for r6_e3.py; its removal rate on the clean training windows
# is a pre-test quantity (the clean part of the drill's training data).
from safetensors.torch import save_file
H.set_seed(100)
trusted = H.train_forecaster(tr[:1], val, seed=100, epochs=L.E3_RECIPE["epochs"], per_epoch=L.E3_RECIPE["per_epoch"])
rv_t = H.residuals(trusted, val[0]); tcv = np.nanmean(rv_t, 0) + 1e-6
t_thr = float(np.quantile(H.window_scores(trusted, val, H.window_index(val), tcv), 0.995))
sc_clean = H.window_scores(trusted, tr, H.window_index(tr), tcv)
pilot_clean_removal = float(np.mean(sc_clean > t_thr))
n_clean_windows = int(len(sc_clean))
save_file({k: v.contiguous().cpu() for k, v in trusted.state_dict().items()}, os.path.join(L.MODELS, "sanitiser_trusted.safetensors"))
np.save(os.path.join(L.MODELS, "sanitiser_trusted_chanvar.npy"), tcv)
del trusted, rv_t, sc_clean
print("sanitiser threshold", round(t_thr, 4), "clean removal", round(pilot_clean_removal, 4), "of", n_clean_windows, flush=True)
src_file = "test2"
src = [a for a in attacks if a["test_file"] == src_file]
evl = [a for a in attacks if a["test_file"] != src_file]
poison_windows = int(sum(a["label_end_idx"] - a["label_start_idx"] + 1 for a in src))
src_prims = set(p for a in src for p in a["primitives"])
n_eval_sharing = sum(1 for a in evl if set(a["primitives"]) & src_prims)

# ---------------------------------------------------------------- 4a. feasibility of the replication plan
procs = sorted(set(p for a in attacks for p in a["target_processes"]))
share = {p: sum(p in a["target_processes"] for a in attacks) / len(attacks) for p in procs}
const_proc = max(procs, key=lambda p: share[p])
cb22 = L.json.load(open(os.path.join(L.EXP, "results", "cluster_bootstrap.json"), encoding="utf8"))
deff_plan = max(cb22["summary"][t]["event_recall"]["dependency_graph_deff"] for t in L.VARIANTS)
e1_22 = L.json.load(open(os.path.join(L.EXP, "results", "e1_baseline_s0.json"), encoding="utf8"))
recall_pilot = e1_22["aggregate"]["events_detected"] / e1_22["aggregate"]["events"]
n_det_plan = int(round(len(attacks) * recall_pilot))
id3_22 = L.json.load(open(os.path.join(L.EXP, "results", "id3_units.json"), encoding="utf8"))
deg22 = id3_22["degradation"]["tests"]["simple/no_sanitisation/vs_clean_e3_recipe"]
p10_pilot = deg22["n10_clean_only"] / deg22["n"]
rows_rep = [
    FZ.check_proportion("IV-2 event recall", len(attacks), L.TAU["recall"], ">=", A, delta=L.DELTA, deff=deff_plan,
                        units="attacks (planning design effect of the 22.04 study)"),
    *[FZ.check_rate(f"IV-2 false-alarm rate ({t})", T_test, L.TAU["far"], A, pilot_rate=ops[t]["pilot_far_per_h"])
      for t in L.VARIANTS],
    FZ.check_rate("IO-2 operator workload", T_total, L.TAU["load"], A,
                  pilot_rate=max(ops[t]["pilot_far_per_h"] for t in L.VARIANTS)),
    FZ.check_proportion("IE-2 localisation", n_det_plan, L.TAU["xfid"], ">=", A, delta=L.DELTA, baseline=share[const_proc],
                        units=f"detected attacks (planning recall {recall_pilot:.3f})"),
    FZ.check_proportion("IR-2 attack success", n_det_plan, L.TAU["asr"], "<=", A, delta=L.DELTA,
                        units=f"detected attacks (planning recall {recall_pilot:.3f})"),
    FZ.check_proportion("ID-3 poison detection", len(src), L.TAU["pdet"], ">=", A, delta=L.DELTA, units="poisoned traces"),
    FZ.check_proportion("ID-3 clean removal", n_clean_windows, L.TAU["pdet_fpr"], "<=", A, pilot=pilot_clean_removal,
                        units="clean training windows; pilot = removal by the planned sanitiser (pre-test)"),
    FZ.check_proportion("ID-3 recall degradation", len(evl), L.TAU["pdeg"], "<=", A, delta=L.DELTA, pilot=p10_pilot,
                        units="attacks; pilot = loss rate of the 22.04 drill"),
    FZ.check_proportion("IV-3 drift TPR (days)", n_pos_h // 24, L.TAU["drift_tpr"], ">=", A, delta=L.DELTA, units="days"),
    FZ.check_proportion("IV-3 drift FPR (days)", n_neg_d, L.TAU["drift_fpr"], "<=", A, delta=L.DELTA, units="days"),
    FZ.check_proportion("IV-3 drift FPR (hours, PSI > 0.25)", n_neg_h, L.TAU["drift_fpr"], "<=", A, delta=L.DELTA,
                        pilot=pilot_fpr_fixed, units="hours; pilot = training hours"),
    FZ.check_proportion("IV-3 drift TPR (hours, envelope monitor)", n_pos_h, L.TAU["drift_tpr"], ">=", A, delta=L.DELTA,
                        pilot=pilot_tpr_23, units="hours; pilot = HAI 23.05 hours"),
    FZ.check_proportion("IV-3 drift FPR (hours, envelope monitor)", n_neg_h, L.TAU["drift_fpr"], "<=", A, delta=L.DELTA,
                        pilot=0.0, units="hours; pilot = leave-one-file-out training hours (calibration sample)"),
]
for r in rows_rep:
    print("REP", r["test"], "|", r["verdict"], "|", {k: r.get(k) for k in ("n", "n_eff", "best_bound", "power_at_delta",
          "min_detectable_margin", "p_accept_at_pilot", "max_rate_at_power")}, flush=True)

# ---------------------------------------------------------------- 4b. retrospective check of the original 22.04 plan
att22 = pd.read_csv(os.path.join(L.ROOT, "data", "processed", "hai2204_attacks.csv"))
share22_p1 = float(np.mean(["P1" in str(t).split(";") for t in att22.target_processes]))
e1c = L.json.load(open(os.path.join(L.EXP, "results", "e1c_false_alarm_dispersion.json"), encoding="utf8"))
v22 = e1c["b_validation"]["baseline_s0"]
e5_22 = L.json.load(open(os.path.join(L.EXP, "results", "e5_drift.json"), encoding="utf8"))
T22 = e1_22["aggregate"]["normal_hours"]
n_pois22 = len(att22[att22.test_file == "test1"])
rows_orig = [
    FZ.check_proportion("IV-2 event recall (independent units)", len(att22), L.TAU["recall"], ">=", A, delta=L.DELTA, units="attacks"),
    FZ.check_proportion("IV-2 event recall (design effect observed later)", len(att22), L.TAU["recall"], ">=", A, delta=L.DELTA,
                        deff=cb22["summary"]["baseline_s0"]["event_recall"]["dependency_graph_deff"], units="attacks"),
    FZ.check_rate("IV-2 false-alarm rate", T22, L.TAU["far"], A, pilot_rate=v22["n_episodes"] / 14.4),
    FZ.check_proportion("IE-2 localisation", len(att22), L.TAU["xfid"], ">=", A, delta=L.DELTA, baseline=share22_p1,
                        units="attacks (all scheduled)"),
    FZ.check_proportion("IR-2 attack success", len(att22), L.TAU["asr"], "<=", A, delta=L.DELTA, units="attacks (upper bound)"),
    FZ.check_proportion("ID-3 poison detection", n_pois22, L.TAU["pdet"], ">=", A, delta=L.DELTA, units="poisoned traces"),
    FZ.check_proportion("ID-3 recall degradation", len(att22) - n_pois22, L.TAU["pdeg"], "<=", A, delta=L.DELTA, units="attacks"),
    FZ.check_proportion("IV-3 drift TPR (days)", e5_22["daily"]["n_pos"], L.TAU["drift_tpr"], ">=", A, delta=L.DELTA, units="days"),
    FZ.check_proportion("IV-3 drift FPR (days)", e5_22["daily"]["n_neg"], L.TAU["drift_fpr"], "<=", A, delta=L.DELTA, units="days"),
]
orig_notes = {
    "margin_not_declared": "the original plan declared no margin delta (deviation D6); delta = 0.2 is used here",
    "evaluation_distribution_not_named": "IP-2 named no evaluation distribution (deviation D2)",
    "dependence_not_addressed": "attacks reuse primitives (58 attacks, 37 primitives) and poisoned windows are replicas (deviations D3, D7)",
    "window_level_oracle_limit_poison_detection": id3_22["traces"]["simple"]["oracle_max_pdet_window_level"],
    "validation_pilot": {"episodes": v22["n_episodes"], "hours": 14.4},
}
for r in rows_orig:
    print("ORIG", r["test"], "|", r["verdict"], "|", {k: r.get(k) for k in ("n", "n_eff", "best_bound", "power_at_delta",
          "p_accept_at_pilot", "trivial_baseline")}, flush=True)
L.save_json({"rows": rows_orig, "notes": orig_notes, "summary": FZ.summarise(rows_orig), "dry_run": DRY,
             "source": "pre-test information of the HAI 22.04 study (validation alarms, attack schedule, drift days, poisoned traces)",
             "alpha": A, "beta": B, "delta": L.DELTA, "generated_utc": L.utc_now()}, "feasibility_original_plan.json")

# ---------------------------------------------------------------- 5. write and freeze the plan
plan = {
    "title": "Confirmatory replication of the HAI case study on HAI 21.03 (final procedure, frozen before test data were read)",
    "reference_profile": "3.CDI.ICS/G-4", "alpha_G": A, "alphas_reported": list(L.ALPHAS), "beta": B, "delta": L.DELTA,
    "thresholds": L.TAU,
    "data": {"campaign": "HAI 21.03", "train": L.TRAIN[:2] + ["train3 (first 80%)"], "validation": "last 20% of train3",
             "test": L.TEST, "attack_schedule": "data/processed/hai2103_attacks.csv (technical documentation, aligned with the label column)",
             "test_normal_hours_e1": T_test, "test_hours_total": T_total, "attacks": len(attacks)},
    "detectors": {"recipe": "as the rated 22.04 detectors (GRU forecaster, W = 60 s, 12 epochs x 250,000 windows)",
                  "rated": list(L.VARIANTS), "variability": [f"baseline_s{s}" for s in L.BASELINE_SEEDS],
                  "channels": prep.cols},
    "alarm_logic": {"smoothing_s": H.SMOOTH, "merge_gap_s": H.MERGE_GAP, "on_delay_s": L.ON_DELAY,
                    "definition": "exceedances of theta closer than 30 s form an episode; an alarm is raised once the episode has lasted "
                                  "20 s and ends with it",
                    "on_delay_rule": "smallest on-delay in 10-s steps with which both rated HAI 22.04 detectors met the false-alarm "
                                     "requirement at their planned thresholds on the 22.04 test data (ISA-18.2 on-delay)",
                    "pilot_hai2204": pilot_k},
    "operating_points": {"rule": "99.9th percentile of the smoothed validation score, raised to the smallest threshold with at most "
                                 "floor(lambda* x T_val) validation alarm episodes (alarm logic with the on-delay) if its validation "
                                 "rate exceeds lambda*",
                         "lambda_star_per_h": lam_star, **ops},
    "procedure": {
        "IV-2": "event recall >= 0.50: exact Clopper-Pearson and effective-n bound with the dependency-graph design effect over attack "
                "primitives (decision on the latter); false alarms <= 0.25/h: exact Poisson bound, dispersion check and quasi-Poisson widening",
        "IO-2": "alarms per hour over the test period <= 6, exact Poisson bound",
        "IE-2": f"process localisation (sum rule) >= 0.60 and superiority over the constant explainer '{const_proc}' (most frequently "
                "targeted process in the schedule): exact one-sided McNemar test with the dependency-graph design effect of the paired differences",
        "IR-2": "TM-A, success over the ball eps <= 0.5 (0.1, 0.25, 0.5) with the strongest attack: union of the planned PGD (60 steps), "
                "the improved PGD (5 restarts, 200 steps, greedy warm start) and greedy prediction replay; both PGD variants minimise the "
                "smoothed score (as planned) and the improved PGD is also run with an objective adapted to the on-delay (minimum of the "
                "smoothed score over every 20-s window); concealment is judged with the deployed alarm logic; detected attacks as units; "
                "unbounded ceiling (eps = 5) reported",
        "IR-1": "greedy transfer from a surrogate trained on historical normal data; hardening-aware surrogate for the hardened variant; "
                "query-based chain (score and label interfaces, Q = 1e3, 1e4, 1e5); reported without threshold",
        "IP-2": "label-only extraction at Q = 1e4, seeds 0-2, 1-D CNN surrogate; fidelity (balanced agreement) on all time-ordered "
                "operational test windows with moving-block (600 s) and test-file cluster bootstrap bounds; attacker distribution reported",
        "ID-3": "poisoning drill, simple replication, beta = 1%; planned sanitiser (trusted model on train1, seed 100, 99.5th percentile "
                "of its validation window scores); detection with the poisoned traces as units (percentile cluster bootstrap, B = 10000) "
                "and clean removal <= 0.05; degradation: poisoned versus clean pipeline with the same recipe, three seeds each, an attack "
                "detected by a pipeline if at least two of its models detect it; exact paired bound",
        "IV-3": "hourly operating-envelope monitor: share of the readings of the channels shared with HAI 22.04 that lie outside the range "
                "of the clean training data; threshold fixed under parameters/drift_monitor; positives = full hours of the HAI 22.04 "
                "training files; negatives = hours of the 21.03 test files with more than 90% normal seconds (E5 mask); requirement "
                "TPR >= 0.90 and FPR <= 0.05; exact bounds with an effective sample size n/deff, deff = (1 + rho)/(1 - rho) from the "
                "lag-1 autocorrelation of the hourly statistic within files (at least 1)",
        "IM-2": "Ed25519 signature over a release manifest, verified against a trust anchor pinned outside the envelope; tests: original, "
                "one modified byte, foreign-key re-signature, replaced anchor",
        "ratings": "evidence classes as in the 22.04 study, file replication_evidence.yaml (SHA-256 "
                   + L.sha256_file(os.path.join(L.HERE, "replication_evidence.yaml")) + "); demonstrated and conditional ratings",
    },
    "design_changes_required_by_the_check": [
        {"item": "alarm on-delay", "reason": "the false-alarm failure of the 22.04 study was foreseeable from the validation data; most false "
                                              "alarms there were fleeting (median 10 s); a 20-s on-delay met the requirement for both 22.04 "
                                              "detectors", "on_delay_s": L.ON_DELAY},
        {"item": "operating point", "reason": "the 99.9th percentile is kept only if its validation false-alarm rate predicts acceptance with probability >= 0.8"},
        {"item": "drift monitor", "reason": f"daily windows are not decidable ({n_neg_d} in-distribution day; 59 needed); with hourly windows the "
                                            f"PSI monitor of the 22.04 plan (threshold 0.25) flags {pilot_fpr_fixed:.2f} of the in-distribution "
                                            f"training hours, so it is replaced by an operating-envelope monitor calibrated by leave-one-file-out",
         "psi_pilot_fpr_training_hours": pilot_fpr_fixed, "envelope_threshold": env_thr,
         "calibration": {"lofo_hours": len(lofo), "lofo_max": float(max(lofo)), "validation_hours": len(val_env),
                         "validation_max": float(max(val_env))},
         "drift_pilot_hai2305": {"hours": len(pilot23), "tpr": pilot_tpr_23, "min_statistic": float(np.min(pilot23)), "channels": len(sh23)}},
        {"item": "poisoning drill", "reason": "at least 14 traces are needed for a decidable detection test; windows without attack samples "
                                              "cap window-level detection", "poison_source": src_file, "poison_traces": len(src),
         "poison_segment": "attack interval with the W-second input history", "poison_windows_per_replica": poison_windows,
         "evaluation_attacks": len(evl), "evaluation_attacks_sharing_a_poisoned_primitive": n_eval_sharing},
        {"item": "degradation measurement", "reason": "with single models, training noise alone predicts acceptance with probability "
                                                      f"{FZ.p_accept_proportion(p10_pilot, len(evl), L.TAU['pdeg'], '<=', A):.2f}; "
                                                      "a three-seed majority vote per pipeline separates the effect of poisoning from training noise"},
    ],
    "parameters": {"constant_explainer": const_proc, "target_process_shares": share,
                   "sanitiser": {"model": "results/models/sanitiser_trusted.safetensors", "threshold": t_thr,
                                 "sha256": L.sha256_file(os.path.join(L.MODELS, "sanitiser_trusted.safetensors")),
                                 "clean_removal_pre_test": pilot_clean_removal},
                   "drift_monitor": {"channels": shared, "lo": env_lo.tolist(), "hi": env_hi.tolist(), "threshold": env_thr},
                   "poison_source_file": src_file, "beta_poison": 0.01,
                   "planning_design_effect": deff_plan, "planning_recall": recall_pilot},
    "feasibility_check": rows_rep,
    "pre_freeze_access": ["training files of HAI 21.03 (all columns)", "label column of the HAI 21.03 test files",
                          "column names of the 22.04 training files", "HAI 23.05 training files (drift pilot)",
                          "result files of the 22.04 study",
                          "the column names and the first data row of the 21.03 test1 file, read once on 2026-10-13 while counting tags"],
}
if DRY:
    with open(os.path.join(L.RES, "plan_dry_run.yaml"), "w", encoding="utf8") as f:
        yaml.safe_dump(L._clean(plan), f, sort_keys=False, allow_unicode=True, width=160)
    print("DRY RUN: plan written to results/plan_dry_run.yaml, not frozen; runtime_s", round(time.time() - t0, 1))
    sys.exit(0)
with open(L.PLAN, "w", encoding="utf8") as f:
    yaml.safe_dump(L._clean(plan), f, sort_keys=False, allow_unicode=True, width=160)
dig = L.sha256_file(L.PLAN)
L.save_json({"plan": os.path.basename(L.PLAN), "plan_sha256": dig, "frozen_utc": L.utc_now(),
             "note": "test-data scripts refuse to run unless the plan file matches this digest"}, "plan_freeze.json")
print("FROZEN", dig, "runtime_s", round(time.time() - t0, 1))
