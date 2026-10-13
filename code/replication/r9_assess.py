"""R9: decisions and ratings of the confirmatory replication on HAI 21.03 (final procedure of assess_case_v2.py).

Every quantitative test is decided with the three-way rule of Section 3.4 at alpha in {0.10, 0.05, 0.01}; levels are
attained from the evidence classes of replication_evidence.yaml (demonstrated: D, NA and accepted or reported T;
conditional: also A); ratings are int() of the attained profile joined with the platform's legacy services at G-4.
Writes results/r_assessment.json. Usage: python r9_assess.py"""
import json
import os
import time

import numpy as np
import yaml
from scipy import stats as st

import r_lib as L
from r_lib import S
from aitcert.model import Catalogue

t0 = time.time()
P = L.require_frozen_plan()
TAU = L.TAU
if not os.path.exists(os.path.join(L.RES, "r_e2.json")):          # E2 was run with one process per variant
    parts = [L.load_json(f"r_e2_{t}.json") for t in L.VARIANTS]
    L.save_json({**{k: v for k, v in parts[0].items() if k not in ("variants", "meta")},
                 "variants": {t: p["variants"][t] for t, p in zip(L.VARIANTS, parts)},
                 "meta": {t: p["meta"] for t, p in zip(L.VARIANTS, parts)}}, "r_e2.json")
e1, e2, e3, e4, e5, im2 = (L.load_json(n) for n in ("r_e1.json", "r_e2.json", "r_e3.json", "r_e4.json", "r_e5.json", "r_im2.json"))
evidence = yaml.safe_load(open(os.path.join(L.HERE, "replication_evidence.yaml"), encoding="utf8"))
ref = json.load(open(os.path.join(L.EXP, "results", "case_assessment.json"), encoding="utf8"))["reference_profile"]
required = {s: ref["services"][s] for s in ("ID", "IR", "IV", "IE", "IP", "IO", "IM")}
platform = {s: l for s, l in ref["services"].items() if s not in required}
G = ref["G"]
cat = Catalogue.load(True)
T_total = e1["test_hours_total"]


def tw_prop(k, n, tau, d, a):
    lcb, ucb = S.cp_lower(k, n, a), S.cp_upper(k, n, a)
    dec = ("accept" if lcb >= tau else ("reject" if ucb < tau else "inconclusive")) if d == ">=" else \
          ("accept" if ucb <= tau else ("reject" if lcb > tau else "inconclusive"))
    return {"k": int(k), "n": int(n), "estimate": k / n if n else None, "lcb": lcb, "ucb": ucb, "tau": tau, "decision": dec,
            "naive": "accept" if (n and ((k / n >= tau) if d == ">=" else (k / n <= tau))) else "reject"}


def tw_rate(c, T, tau, a):
    lcb, ucb = S.poisson_lower(c, T, a), S.poisson_upper(c, T, a)
    return {"count": int(c), "exposure_h": T, "estimate": c / T, "lcb": lcb, "ucb": ucb, "tau": tau,
            "decision": "accept" if ucb <= tau else ("reject" if lcb > tau else "inconclusive"),
            "naive": "accept" if c / T <= tau else "reject"}


def tw_bounds(est, lcb, ucb, tau, d, method):
    dec = ("accept" if lcb >= tau else ("reject" if ucb < tau else "inconclusive")) if d == ">=" else \
          ("accept" if ucb <= tau else ("reject" if lcb > tau else "inconclusive"))
    return {"estimate": est, "lcb": lcb, "ucb": ucb, "tau": tau, "decision": dec, "method": method}


def effn(k, n, deff, a):
    d = max(1.0, deff); ne, ke = n / d, k / d
    lo = 0.0 if ke <= 0 else float(st.beta.ppf(a, ke, ne - ke + 1))
    hi = 1.0 if ke >= ne else float(st.beta.ppf(1 - a, ke + 1, ne - ke))
    return lo, hi


def conj(*ds):
    d = [x for x in ds if x != "reported"]
    if any(x == "reject" for x in d):
        return "reject"
    if all(x == "accept" for x in d):
        return "accept" if d else "reported"
    return "inconclusive"


def variant_tests(tag, a):
    v = e1["variants"][tag]; ag = v["aggregate"]; t = {}
    t["IV-1.report"] = {"decision": "reported", "recall": ag["event_recall"], "far_per_h": ag["far_per_h"]}
    k, n, deff = ag["events_detected"], ag["events"], v["recall_design_effect"]
    lo, hi = effn(k, n, deff, a)
    t["IV-2.recall"] = {**tw_bounds(k / n, lo, hi, TAU["recall"], ">=", f"effective-n CP, design effect {deff:.2f}"),
                        "exact": tw_prop(k, n, TAU["recall"], ">=", a), "deff": deff, "k": k, "n": n}
    c, T = ag["false_alarms"], ag["normal_hours"]
    ex = tw_rate(c, T, TAU["far"], a)
    phi = v["dispersion"]["dispersion_x2_over_df"]
    if phi > 1 and c > 0:
        z = st.norm.ppf(1 - a); se = np.sqrt(phi / c); lam = c / T
        lo_, hi_ = min(lam * np.exp(-z * se), ex["lcb"]), max(lam * np.exp(z * se), ex["ucb"])
    else:
        lo_, hi_ = ex["lcb"], ex["ucb"]
    t["IV-2.far"] = {**tw_bounds(c / T, lo_, hi_, TAU["far"], "<=", f"exact Poisson; dispersion X2/df = {phi:.2f}"),
                     "exact": ex, "dispersion": phi, "count": c, "hours": T}
    t["IO-2.load"] = tw_rate(v["alarms_total"], T_total, TAU["load"], a)
    x = v["explanation"]
    absolute = tw_prop(x["k_model"], x["n_detected"], TAU["xfid"], ">=", a)
    comp = "accept" if x["p_one_sided_dependence_robust"] <= a else ("reject" if x["p_reverse_dependence_robust"] <= a else "inconclusive")
    comp_iid = "accept" if x["p_one_sided_independent"] <= a else ("reject" if x["p_reverse_independent"] <= a else "inconclusive")
    t["IE-2.xfid"] = {"decision": conj(absolute["decision"], comp), "absolute": absolute,
                      "constant": tw_prop(x["k_constant"], x["n_detected"], TAU["xfid"], ">=", a),
                      "comparison": {"b": x["b_model_only"], "c": x["c_constant_only"], "deff": x["deff_pairs"],
                                     "p_dependence_robust": x["p_one_sided_dependence_robust"], "p_independent": x["p_one_sided_independent"],
                                     "decision": comp, "decision_independent_units": comp_iid}}
    s2 = e2["variants"][tag]["summary"]
    t["IR-1.transfer"] = {"decision": "reported", "k": s2["transfer_data_surrogate_by_eps"]["0.5"], "n": s2["n_attackable"]}
    if "transfer_hardaware_surrogate_by_eps" in s2:
        t["IR-1.transfer"]["hardening_aware"] = s2["transfer_hardaware_surrogate_by_eps"]["0.5"]
    ch = e4["victims"][tag]["ir1_chain"]
    t["IR-1.query"] = {"decision": "reported", "k_max": max(r["transfer_k"] for r in ch), "n": ch[0]["transfer_n"]}
    t["IR-2.asr"] = {**tw_prop(s2["strongest_ball_by_eps"]["0.5"], s2["n_attackable"], TAU["asr"], "<=", a),
                     "planned_pgd_ball": s2["planned_pgd_ball"], "planned_pgd_eps0.5": s2["planned_pgd_by_eps"]["0.5"],
                     "improved_by_eps": s2["improved_pgd_by_eps"], "greedy_by_eps": s2["greedy_by_eps"], "unbounded": s2["unbounded"]}
    runs = e4["victims"][tag]["ip2_runs"]
    u = max(max(r["moving_block_ucb95"], r["file_cluster_ucb95"]) for r in runs)
    t["IP-2.ext"] = {"decision": "accept" if u <= TAU["ext"] else "inconclusive", "max_ucb95": u,
                     "estimates": [r["balanced_operational"] for r in runs],
                     "attacker_distribution": [r["balanced_attacker_distribution"] for r in runs]}
    return t


def shared_tests(a):
    t = {}
    sc = e3["screening"]
    lcb = sc["cluster_bootstrap"]["pooled_lcb_one_sided_by_alpha"][str(a)]
    fin = tw_bounds(sc["pooled_fraction"], lcb, sc["cluster_bootstrap"]["pooled_ucb95_one_sided"], TAU["pdet"], ">=",
                    "percentile cluster bootstrap over the poisoned traces")
    clean = tw_prop(sc["clean_removed"], sc["clean_windows"], TAU["pdet_fpr"], "<=", a)
    t["ID-3.pdet"] = {**fin, "decision": conj(fin["decision"], clean["decision"]), "clean_removal": clean}
    dg = e3["degradation"]
    lo = S.cp_lower(dg["n10_clean_only"], dg["n"], a / 2) - S.cp_upper(dg["n01_poisoned_only"], dg["n"], a / 2)
    t["ID-3.pdeg"] = {**tw_bounds(dg["degradation_estimate"], lo, S.cp_upper(dg["n10_clean_only"], dg["n"], a), TAU["pdeg"], "<=",
                                  "paired exact bound"), "n10": dg["n10_clean_only"], "n01": dg["n01_poisoned_only"], "n": dg["n"]}
    r = e5["results"]["calibrated"]
    parts = {}
    for key, tau, d in (("tpr", TAU["drift_tpr"], ">="), ("fpr", TAU["drift_fpr"], "<=")):
        x = r[key]; lo_, hi_ = effn(x["k"], x["n"], x["deff"], a)
        parts[key] = {**tw_bounds(x["k"] / x["n"], lo_, hi_, tau, d, f"effective-n CP, deff {x['deff']:.2f}"), "k": x["k"], "n": x["n"]}
    t["IV-3.drift"] = {"decision": conj(parts["tpr"]["decision"], parts["fpr"]["decision"]), **parts}
    t["IM-2.sig"] = {"decision": "accept" if im2["all_tests_as_expected"] else "reject",
                     "tests": {k: v["outcome"] for k, v in im2["tests"].items()}}
    return t


def met(req, tests, reading):
    s = req["status"]
    if s in ("D", "NA"):
        return True
    if s == "A":
        return reading == "conditional"
    if s == "NE":
        return False
    return conj(*[tests[k]["decision"] if k in tests else "missing" for k in req["tests"]]) in ("accept", "reported")


def attained(tests, reading):
    lv_, det = {}, {}
    for s, lmax in required.items():
        lv = 0
        for l in range(1, lmax + 1):
            reqs = evidence.get(f"{s}-{l}", [])
            ok = [(r["req"], met(r, tests, reading)) for r in reqs]
            det[f"{s}-{l}"] = dict(ok)
            if reqs and all(v for _, v in ok):
                lv = l
            else:
                break
        lv_[s] = lv
    return lv_, det


fmt = lambda p: "{" + ", ".join(f"{s}-{l}" for s, l in p.items() if l > 0) + "}"
out = {"plan_sha256": L.load_json("plan_freeze.json")["plan_sha256"], "reference_profile": fmt(required), "by_alpha": {}}
for a in L.ALPHAS:
    sh = shared_tests(a)
    for tag in L.VARIANTS:
        tests = {**sh, **variant_tests(tag, a)}
        if tag == "hardened_s0":
            tests["ID-3.pdeg"] = {"decision": "not evaluated", "note": "the drill trained the baseline recipe only"}
        res = {}
        for reading in ("demonstrated", "conditional"):
            lv, det = attained(tests, reading)
            prof = {**platform, **{s: l for s, l in lv.items() if l > 0}}
            inner = cat.interior(prof, G)
            rt = {s: inner.get(s, 0) for s in required}
            res[reading] = {"own": lv, "rating": rt, "formatted": fmt(rt), "gaps": {s: (rt[s], required[s]) for s in required if rt[s] < required[s]},
                            "requirements": det}
        out["by_alpha"].setdefault(str(a), {})[tag] = {"tests": tests, "results": res}
for tag in L.VARIANTS:
    r = out["by_alpha"]["0.05"][tag]
    print("==", tag)
    for k, v in r["tests"].items():
        print(f"  {k:14s} {v.get('decision')}", {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                                  if kk in ("k", "n", "estimate", "lcb", "ucb", "count", "max_ucb95")})
    for reading in ("demonstrated", "conditional"):
        print("  ", reading, r["results"][reading]["formatted"], "gaps", r["results"][reading]["gaps"])
for a in L.ALPHAS:
    print("alpha", a, {t: out["by_alpha"][str(a)][t]["results"]["conditional"]["formatted"] for t in L.VARIANTS})
out["meta"] = {"script": "r9_assess.py", "runtime_s": time.time() - t0, "finished_utc": L.utc_now()}
L.save_json(out, "r_assessment.json")
