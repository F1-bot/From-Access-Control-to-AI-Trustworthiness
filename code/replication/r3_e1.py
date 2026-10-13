"""R3 (E1 of the replication): validity (IV-2), operator workload (IO-2) and explanation (IE-2) on HAI 21.03.

Final procedure of the 22.04 study, applied at the operating points fixed in the frozen plan:
  * event detection with the hai_lib conventions and the on-delay of the plan (exceedances closer than 30 s form an episode;
    an alarm is raised once an episode has lasted 20 s; an attack counts as detected if an alarm overlaps [start, end + 60 s];
    other alarms are false alarms per hour of normal operation);
  * event recall: exact Clopper-Pearson bounds and the effective-sample-size bound with the dependency-graph design effect
    over attack primitives (Section 3.7.5);
  * false alarms: exact Poisson bound, hourly dispersion index X2/df and quasi-Poisson widening if it exceeds 1;
  * workload: all alarm episodes per hour of the test period;
  * explanations: the process with the largest summed contribution in the first 30 s of the alarm (hai_lib.localise);
    comparison with the constant explainer of the plan by the exact one-sided McNemar test, with the dependency-graph
    design effect of the paired differences; channel-proportional random explainer and tag-level top-5 hits reported.
Writes results/r_e1.json and results/models/<tag>_test_scores.npz.
Usage: python r3_e1.py"""
import math
import os
import time

import numpy as np
from scipy import stats as st

import r_lib as L
from r_lib import H, S

t0 = time.time()
P = L.require_frozen_plan()
prep = L.load_prep()
cols = list(prep.cols)
F = len(cols)
arrays, ys = L.test_data(prep)
attacks = L.load_attacks()
ids = [a["attack_id"] for a in attacks]
prims = {a["attack_id"]: a["primitives"] for a in attacks}
D = L.dependency_matrix(ids, prims)
by_file = {f"test{i}": [a for a in attacks if a["test_file"] == f"test{i}"] for i in range(1, 6)}
for i, y in enumerate(ys, start=1):          # schedule and labels must align
    ev = H.events_from_labels(y)
    assert [(a["label_start_idx"], a["label_end_idx"]) for a in by_file[f"test{i}"]] == ev, f"test{i} misaligned"
const = P["parameters"]["constant_explainer"]
proc_of = np.array([c.split("_")[0] for c in cols])
PROCS = sorted(set(proc_of.tolist()))
n_ch = {p: int((proc_of == p).sum()) for p in PROCS}
T_total = sum(len(y) for y in ys) / 3600.0


def mask_e1(y):
    T = len(y); m = np.ones(T, bool); m[:H.W] = False
    for s, e in H.events_from_labels(y):
        m[s:min(T, e + H.TOL + 1)] = False
    return m


def dispersion(scores, theta):
    counts, expo = [], []
    for sc, y in zip(scores, ys):
        ev = H.events_from_labels(y)
        fa = [a for a in L.alarm_episodes_k(sc, theta, L.ON_DELAY) if not any(a[0] <= e + H.TOL and a[1] >= s for s, e in ev)]
        m = mask_e1(y)
        nb = int(math.ceil(len(y) / 3600))
        c = np.zeros(nb, int)
        for a0, _ in fa:
            c[a0 // 3600] += 1
        counts += c.tolist()
        expo += [m[b * 3600:min(len(y), (b + 1) * 3600)].sum() / 3600.0 for b in range(nb)]
    c, w = np.asarray(counts, float), np.asarray(expo, float)
    keep = w > 0
    c, w = c[keep], w[keep]
    lam = c.sum() / w.sum()
    phi = float(np.sum((c - lam * w) ** 2 / (lam * w)) / (len(c) - 1)) if lam > 0 else 1.0
    return {"hourly_counts": c.tolist(), "hourly_exposure_h": w.round(6).tolist(), "dispersion_x2_over_df": phi}


out = {"plan_sha256": L.load_json("plan_freeze.json")["plan_sha256"], "constant_explainer": const,
       "test_hours_total": T_total, "variants": {}}
tags = [f"baseline_s{s}" for s in L.BASELINE_SEEDS] + ["hardened_s0"]
for tag in tags:
    th = float(P["operating_points"][tag]["theta"])
    cv = L.chanvar(tag)
    model = L.load_model(tag, F)
    per_file, scores, recs = [], {}, []
    for i, (arr, y) in enumerate(zip(arrays, ys), start=1):
        res = H.residuals(model, arr)
        sc = H.smooth_fast(H.score_from_residuals(res, cv))
        scores[f"test{i}"] = sc
        ev = L.evaluate_file_k(sc, y, th, L.ON_DELAY)
        per_file.append(ev)
        if tag not in L.VARIANTS:
            continue
        for (s, e), d, a in zip(ev["events"], ev["detected"], by_file[f"test{i}"]):
            if not d:
                continue
            hit = [al for al in ev["alarms"] if al[0] <= e + H.TOL and al[1] >= s]
            start = max(min(al[0] for al in hit), s)
            loc = H.localise(res, cv, cols, start)
            tp = set(a["target_processes"])
            recs.append({"attack_id": a["attack_id"], "test_file": f"test{i}", "primitives": a["primitives"],
                         "target_processes": sorted(tp), "top_process": loc["top_process"],
                         "process_hit": loc["top_process"] in tp, "constant_hit": const in tp,
                         "top_tags": loc["top_tags"], "tag_hit_at5": bool(set(loc["top_tags"]) & set(a["target_points"])),
                         "n_target_channels": int(sum(c in set(a["target_points"]) for c in cols)),
                         "random_hit_probability": sum(n_ch.get(p, 0) for p in tp) / F})
        del res
    agg = H.aggregate(per_file)
    det = [int(d) for e in per_file for d in e["detected"]]
    agg["detected_vector"] = det
    np.savez_compressed(os.path.join(L.MODELS, f"{tag}_test_scores.npz"), **{k: v.astype(np.float32) for k, v in scores.items()})
    rec = {"theta": th, "aggregate": agg, "ttd": [t for e in per_file for t in e["ttd"]]}
    if tag in L.VARIANTS:
        y_det = np.array(det, float)
        rec["recall_design_effect"] = L.graph_deff(y_det, D)
        rec["dispersion"] = dispersion([scores[f"test{i}"] for i in range(1, 6)], th)
        rec["alarms_total"] = int(sum(e["n_alarms"] for e in per_file))
        # explanations
        sub = [r for r in recs]
        rid = [r["attack_id"] for r in sub]
        m = np.array([r["process_hit"] for r in sub], float)
        c = np.array([r["constant_hit"] for r in sub], float)
        Dd = L.dependency_matrix(rid, prims)
        deff_pairs = L.graph_deff(m - c, Dd)
        b, cc = int(((m == 1) & (c == 0)).sum()), int(((m == 0) & (c == 1)).sum())
        p_iid = float(st.binom.sf(b - 1, b + cc, 0.5)) if b + cc else 1.0
        p_iid_rev = float(st.binom.sf(cc - 1, b + cc, 0.5)) if b + cc else 1.0
        p_dep = float(st.beta.cdf(0.5, b / deff_pairs, cc / deff_pairs + 1)) if b > 0 else 1.0
        p_dep_rev = float(st.beta.cdf(0.5, cc / deff_pairs, b / deff_pairs + 1)) if cc > 0 else 1.0
        q = [r["random_hit_probability"] for r in sub]
        tag_q = [1.0 - (math.comb(F - r["n_target_channels"], 5) / math.comb(F, 5)) for r in sub]
        rec["explanation"] = {"n_detected": len(sub), "k_model": int(m.sum()), "k_constant": int(c.sum()),
                              "b_model_only": b, "c_constant_only": cc, "deff_pairs": deff_pairs,
                              "p_one_sided_independent": p_iid, "p_reverse_independent": p_iid_rev,
                              "p_one_sided_dependence_robust": p_dep, "p_reverse_dependence_robust": p_dep_rev,
                              "model_only_attacks": [r["attack_id"] for r in sub if r["process_hit"] and not r["constant_hit"]],
                              "constant_only_attacks": [r["attack_id"] for r in sub if r["constant_hit"] and not r["process_hit"]],
                              "random_expected_accuracy": float(np.mean(q)) if q else None,
                              "tag_top5_hits": int(sum(r["tag_hit_at5"] for r in sub)),
                              "tag_top5_chance_rate": float(np.mean(tag_q)) if tag_q else None,
                              "records": sub}
    out["variants"][tag] = rec
    print(tag, "theta", round(th, 3), "recall", agg["events_detected"], "/", agg["events"], "FA", agg["false_alarms"],
          "h", round(agg["normal_hours"], 2), "rate", round(agg["far_per_h"], 3),
          ("loc " + str(rec["explanation"]["k_model"]) + "/" + str(rec["explanation"]["n_detected"]) +
           " const " + str(rec["explanation"]["k_constant"]) + " b,c " + str((rec["explanation"]["b_model_only"],
           rec["explanation"]["c_constant_only"])) + " p " + str(round(rec["explanation"]["p_one_sided_dependence_robust"], 4))
           + " deff " + str(round(rec["recall_design_effect"], 2))) if tag in L.VARIANTS else "", flush=True)
    del model
seeds = [out["variants"][f"baseline_s{s}"]["aggregate"] for s in L.BASELINE_SEEDS]
out["baseline_seeds"] = {"recall_mean": float(np.mean([a["event_recall"] for a in seeds])),
                         "recall_sd": float(np.std([a["event_recall"] for a in seeds], ddof=1)),
                         "far_mean": float(np.mean([a["far_per_h"] for a in seeds])),
                         "far_sd": float(np.std([a["far_per_h"] for a in seeds], ddof=1)),
                         "recall_range": [min(a["event_recall"] for a in seeds), max(a["event_recall"] for a in seeds)]}
out["meta"] = {"script": "r3_e1.py", "runtime_s": time.time() - t0, "finished_utc": L.utc_now(), "versions": L.versions()}
L.save_json(out, "r_e1.json")
print("seeds", out["baseline_seeds"], "runtime", round(time.time() - t0))
