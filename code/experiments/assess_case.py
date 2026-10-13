"""Level assignment for the HAI case study against the reference profile 3.CDI.ICS (assurance G-4,
alpha = 0.05), using the pre-registered thresholds and the probabilistic conformity semantics.
Legacy services are assumed to be provided by a separately evaluated hosting platform (composite
evaluation); procedural requirements are marked as evidenced by artefacts produced in this study.
Writes results/case_assessment.json."""
import json, os
import numpy as np
import aitcert_path  # noqa: F401
from aitcert import stats as S
from aitcert.model import Catalogue

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A = 0.05
pre = {"recall": 0.50, "far": 0.25, "load": 6.0, "asr": 0.20, "pdet": 0.80, "pdet_fpr": 0.05, "pdeg": 0.10, "ext": 0.90, "xfid": 0.60, "drift_tpr": 0.90, "drift_fpr": 0.05}
cat = Catalogue.load(True)
legacy_3cd3 = json.load(open(os.path.join(ROOT, "data/processed/nd_tzi_2.5-005-99_profiles.json"), encoding="utf8"))
p3cd3 = next(p for p in legacy_3cd3["profiles"] if p["id"] == "3.CD.3")["services"]
ai_req = {"ID": 3, "IR": 2, "IV": 3, "IE": 2, "IP": 2, "IO": 2, "IM": 2}
req, gq = cat.closure({**p3cd3, **ai_req}, 4)
out = {"reference_profile": {"id": "3.CDI.ICS", "legacy_base": "3.CD.3", "services": req, "G": gq,
                             "closure_added": {k: v for k, v in req.items() if k not in p3cd3 and k not in ai_req},
                             "formatted": cat.fmt(req)}, "variants": {}}
e2 = json.load(open(os.path.join(R, "e2_evasion.json"))); e2b = json.load(open(os.path.join(R, "e2b_greedy.json")))
e4 = json.load(open(os.path.join(R, "e4b_extract.json"))); e5 = json.load(open(os.path.join(R, "e5_drift.json")))
e3 = json.load(open(os.path.join(R, "e3_poison_beta0.01.json")))
e3b = json.load(open(os.path.join(R, "e3b_poison_dose.json"))) if os.path.exists(os.path.join(R, "e3b_poison_dose.json")) else None
im = json.load(open(os.path.join(R, "im_baseline_s0.json")))

def asr_union(tag, tm="TM-A", eps=0.5):
    """Event concealed by ANY attack (PGD or greedy, white-box): the strongest adaptive evaluation."""
    ids = {}
    for r in e2["models"][tag]["rows"]:
        if r["tm"] == tm and abs(r["eps"] - eps) < 1e-9:
            ids[r["attack_id"]] = ids.get(r["attack_id"], False) or r["concealed"]
    for r in e2b["models"][tag]["rows"]:
        if r["tm"] == tm and abs(r["eps"] - eps) < 1e-9 and r["mode"] == "white-box":
            ids[r["attack_id"]] = ids.get(r["attack_id"], False) or r["concealed"]
    k, n = sum(ids.values()), len(ids)
    return k, n

for tag in ("baseline_s0", "hardened_s0"):
    e1 = json.load(open(os.path.join(R, f"e1_{tag}.json"))); a = e1["aggregate"]
    ev = {}
    # IV-2
    rec = S.accept_proportion(a["events_detected"], a["events"], pre["recall"], ">=", A)
    far = S.accept_rate(a["false_alarms"], a["normal_hours"], pre["far"], A)
    ev["IV-2"] = {"recall": rec, "far": far, "met": rec["accepted"] and far["accepted"]}
    # IO-2 workload (all alarms per hour over the whole test period)
    hours = sum([24.0, 23.0, 17.333, 36.0])
    n_alarms = round(a["alarms_per_h"] * hours)
    load = S.accept_rate(n_alarms, hours, pre["load"], A)
    ev["IO-2"] = {"load": load, "met": load["accepted"]}
    # IE-2 localisation (process level)
    loc = e1["localisation"]; kx = sum(l["process_hit"] for l in loc)
    xf = S.accept_proportion(kx, len(loc), pre["xfid"], ">=", A)
    ev["IE-2"] = {"process_localisation": xf, "tag_hit_at5": float(np.mean([l["tag_hit_at5"] for l in loc])), "met": xf["accepted"]}
    # IR-2 (declared threat model TM-A, eps = 0.5 SD, strongest of PGD and greedy white-box attacks)
    k, n = asr_union(tag)
    asr = S.accept_proportion(k, n, pre["asr"], "<=", A)
    ev["IR-2"] = {"asr": asr, "met": asr["accepted"]}
    ev["IR-1"] = {"transfer_asr_TM-A_0.5": [ (r["concealed"], r["n"]) for r in e2b["models"][tag]["summary"] if r["mode"] == "transfer" and r["tm"] == "TM-A" and abs(r["eps"] - 0.5) < 1e-9][0], "met": True}
    out["variants"][tag] = ev
# model-independent evidence (baseline victim)
runs = [r for r in e4["runs"] if r["interface"] == "label" and r["Q"] == 10000]
ext_test = [r["test_dist"] for r in runs]; ext_att = [r["attacker_dist"] for r in runs]
out["IP-2"] = {"label_Q1e4_test_dist_balanced": [x["balanced"] for x in ext_test], "ucb95_test": [x["ucb95"] for x in ext_test],
               "label_Q1e4_attacker_dist_balanced": [x["balanced"] for x in ext_att], "ucb95_attacker": [x["ucb95"] for x in ext_att],
               "met_operational": all(x["ucb95"] <= pre["ext"] for x in ext_test), "met_attacker_dist": all(x["ucb95"] <= pre["ext"] for x in ext_att)}
s3 = e3["runs"]["simple"]["sanitiser"]
out["ID-3_prereg"] = {"pdet": S.accept_proportion(s3["poison_removed"], s3["poison_windows"], pre["pdet"], ">=", A),
                      "fpr": s3["fpr"], "recall_clean_test2-4": 33 / 51,
                      "recall_poisoned_test2-4": e3["runs"]["simple"]["no_sanitisation"]["event_recall"],
                      "met": False}
if e3b:
    out["ID-3_exploratory"] = [{"beta": r["beta"], "kind": r["kind"], "recall_poisoned": r["poisoned"]["event_recall"], "recall_sanitised": r["sanitised"]["event_recall"],
                                "recall_matching_poisoned": r["poisoned"]["recall_matching"], "pdet": r["pdet"], "fpr": r["fpr"]} for r in e3b["runs"]]
    out["ID-3_exploratory_clean"] = {"recall": e3b["clean"]["event_recall"], "recall_matching": e3b["clean"]["recall_matching"]}
out["IV-3"] = {"daily": {k: e5["daily"][k] for k in ("n_pos", "n_neg", "tp", "fp", "tpr_lcb95", "fpr_ucb95")},
               "hourly_psi": {k: e5["hourly"][k] for k in ("n_pos", "n_neg", "tp", "fp", "tpr_lcb95", "fpr_ucb95")},
               "hourly_residual": {k: e5["residual_monitor_hourly"][k] for k in ("n_pos", "n_neg", "tp", "fp", "tpr_lcb95", "fpr_ucb95")},
               "far_2204_shared": e5["far_2204_test_normal_per_h"], "far_2305_shared": e5["far_2305_per_h"],
               "far_2305_ucb95": e5["far_2305_ucb95"], "met": False}
out["IM-2"] = {"signature_valid": im["signature_valid"], "tamper_detected": im["tamper_detected"], "serialization": im["serialization"], "met": im["signature_valid"] and im["tamper_detected"]}
# achieved levels (cumulative semantics)
for tag, ev in out["variants"].items():
    lv = {"ID": 2,  # ID-1/ID-2 evidenced by SHA-256 manifests and versioned pipeline; ID-3 drill failed
          "IR": 2 if ev["IR-2"]["met"] else 1,
          "IV": 2 if ev["IV-2"]["met"] else 1,   # IV-3 additionally failed (drift monitors)
          "IE": 2 if ev["IE-2"]["met"] else 1,
          "IP": 2 if out["IP-2"]["met_operational"] else 1,
          "IO": 2 if ev["IO-2"]["met"] else 1,
          "IM": 2 if out["IM-2"]["met"] else 1}
    gaps = {s: (lv[s], ai_req[s]) for s in ai_req if lv[s] < ai_req[s]}
    ev["achieved"] = lv; ev["gaps"] = gaps; ev["conformant"] = not gaps
# exposure needed to certify FAR <= 0.25/h at alpha 0.05 with power 0.8, for true rates 0.10 and 0.15/h
from scipy import stats as st
def hours_needed(lam, tau=0.25, alpha=0.05, power=0.8):
    for T in np.arange(50, 5000, 10):
        # accept if poisson_upper(count, T) <= tau  <=> count <= c_max
        c = 0
        while S.poisson_upper(c, T, alpha) <= tau:
            c += 1
        cmax = c - 1
        if cmax >= 0 and st.poisson.cdf(cmax, lam * T) >= power:
            return int(T)
    return None
out["far_certification_hours"] = {str(l): hours_needed(l) for l in (0.05, 0.10, 0.15)}
json.dump(out, open(os.path.join(R, "case_assessment.json"), "w"), indent=1, default=float)
print(json.dumps({t: {"achieved": v["achieved"], "gaps": v["gaps"], "IV-2": (v["IV-2"]["recall"]["bound"], v["IV-2"]["far"]["bound"]),
                      "IR-2": (v["IR-2"]["asr"]["k"], v["IR-2"]["asr"]["n"], v["IR-2"]["asr"]["bound"]), "IE-2": v["IE-2"]["process_localisation"]["bound"],
                      "IO-2": v["IO-2"]["load"]["bound"]} for t, v in out["variants"].items()}, indent=1))
print("ref profile:", out["reference_profile"]["formatted"], "G", out["reference_profile"]["G"], "added", out["reference_profile"]["closure_added"])
print("IP-2", out["IP-2"]); print("FAR hours", out["far_certification_hours"])
