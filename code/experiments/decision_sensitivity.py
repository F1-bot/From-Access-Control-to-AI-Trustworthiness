"""Decision-rule sensitivity and three-way outcomes for the pre-registered tests of the HAI case study
(major revision, reviewer request).

For every pre-registered quantitative test and alpha in {0.10, 0.05, 0.01} the script reports
  * the bound rule of the manuscript (accept iff the one-sided exact (1 - alpha) bound clears tau),
  * the naive point-estimate rule (accept iff the estimate clears tau),
  * a three-way outcome: ACCEPT if the bound clears tau; REJECT if the opposite one-sided (1 - alpha) bound
    violates tau (LCB > tau for a '<=' requirement, UCB < tau for a '>=' requirement); INCONCLUSIVE otherwise.
    Equivalently, the (1 - 2 alpha) two-sided interval lies entirely on the conforming side (accept), entirely
    on the non-conforming side (reject), or straddles tau. A bound equal to tau is "clearing", not "violating".
Requirements with several components (IV-2: recall and FAR; IV-3: TPR and FPR; IP-2: three extraction seeds)
are ACCEPTED only if every component is accepted (intersection-union test, no alpha adjustment needed for the
consumer's risk), REJECTED if any component is rejected, and INCONCLUSIVE otherwise.

Inputs (read only): results/case_assessment.json (IV-2, IO-2, IE-2, IR-2 union counts), results/e2_evasion.json
(IR-2 literal PGD counts), results/e4b_extract.json (IP-2 per-seed agreement counts), results/e5_drift.json
(IV-3 daily PSI monitor), results/e3_poison_beta0.01.json (ID-3, added for completeness).
Bounds: Clopper-Pearson for proportions, Garwood (exact Poisson) for rates, and for the balanced extraction
fidelity the mean of the two class-wise CP bounds at alpha/2 each (the construction of e4b_extract.py).

Writes results/decision_sensitivity.json. Usage: python decision_sensitivity.py"""
import json
import math
import os
import time

import rev_lib as R
from aitcert_path import stats as S

SCRIPT = "decision_sensitivity.py"
ALPHAS = (0.10, 0.05, 0.01)
VARIANTS = ("baseline_s0", "hardened_s0")
EQ_TOL = 1e-12
t0, started = time.time(), R.utc_now()

ca = json.load(open(os.path.join(R.RES, "case_assessment.json")))
e2 = json.load(open(os.path.join(R.RES, "e2_evasion.json")))
e4 = json.load(open(os.path.join(R.RES, "e4b_extract.json")))
e5 = json.load(open(os.path.join(R.RES, "e5_drift.json")))
e3 = json.load(open(os.path.join(R.RES, "e3_poison_beta0.01.json")))


def clears(bound, tau, direction):
    return bound >= tau - EQ_TOL if direction == ">=" else bound <= tau + EQ_TOL


def violates(opp, tau, direction):
    return opp < tau - EQ_TOL if direction == ">=" else opp > tau + EQ_TOL


def three_way(bound, opp, tau, direction):
    if clears(bound, tau, direction):
        return "accept"
    if violates(opp, tau, direction):
        return "reject"
    return "inconclusive"


def boundary_flag(bound, opp, tau):
    return any(abs(b - tau) < 1e-9 for b in (bound, opp))


def prop_test(k, n, tau, direction):
    est = k / n
    out = {"type": "proportion", "k": k, "n": n, "estimate": est, "tau": tau, "direction": direction,
           "naive": "accept" if (est >= tau if direction == ">=" else est <= tau) else "reject", "by_alpha": {}}
    for a in ALPHAS:
        lcb, ucb = S.cp_lower(k, n, a), S.cp_upper(k, n, a)
        bound, opp = (lcb, ucb) if direction == ">=" else (ucb, lcb)
        out["by_alpha"][str(a)] = {"lcb": lcb, "ucb": ucb, "bound_rule": "accept" if clears(bound, tau, direction) else "not accepted",
                                   "three_way": three_way(bound, opp, tau, direction),
                                   "boundary_case": boundary_flag(bound, opp, tau)}
    return out


def rate_test(count, T, tau):
    est = count / T
    out = {"type": "poisson_rate", "count": count, "exposure_h": T, "estimate": est, "tau": tau, "direction": "<=",
           "naive": "accept" if est <= tau else "reject", "by_alpha": {}}
    for a in ALPHAS:
        lcb, ucb = S.poisson_lower(count, T, a), S.poisson_upper(count, T, a)
        out["by_alpha"][str(a)] = {"lcb": lcb, "ucb": ucb, "bound_rule": "accept" if clears(ucb, tau, "<=") else "not accepted",
                                   "three_way": three_way(ucb, lcb, tau, "<="), "boundary_case": boundary_flag(ucb, lcb, tau)}
    return out


def fidelity_test(runs, tau, dist):
    """IP-2: balanced agreement <= tau for every extraction seed (label-only, Q = 1e4)."""
    seeds = []
    for r in runs:
        d = r[dist]
        tp, tn = d["agree_pos"] * d["n_pos"], d["agree_neg"] * d["n_neg"]
        if abs(tp - round(tp)) > 1e-6 or abs(tn - round(tn)) > 1e-6:
            raise RuntimeError("could not reconstruct integer agreement counts")
        tp, tn = int(round(tp)), int(round(tn))
        rec_ucb95 = 0.5 * (S.cp_upper(tp, d["n_pos"], 0.025) + S.cp_upper(tn, d["n_neg"], 0.025))
        s = {"seed": r["seed"], "tp": tp, "n_pos": d["n_pos"], "tn": tn, "n_neg": d["n_neg"], "estimate": d["balanced"],
             "stored_ucb95": d["ucb95"], "recomputed_ucb95": rec_ucb95, "reproduces_stored": abs(rec_ucb95 - d["ucb95"]) < 1e-12,
             "by_alpha": {}}
        for a in ALPHAS:
            ucb = 0.5 * (S.cp_upper(tp, d["n_pos"], a / 2) + S.cp_upper(tn, d["n_neg"], a / 2))
            lcb = 0.5 * (S.cp_lower(tp, d["n_pos"], a / 2) + S.cp_lower(tn, d["n_neg"], a / 2))
            s["by_alpha"][str(a)] = {"lcb": lcb, "ucb": ucb, "bound_rule": "accept" if clears(ucb, tau, "<=") else "not accepted",
                                     "three_way": three_way(ucb, lcb, tau, "<=")}
        seeds.append(s)
    out = {"type": "balanced_agreement_max_over_seeds", "distribution": dist, "tau": tau, "direction": "<=",
           "estimate_max_over_seeds": max(s["estimate"] for s in seeds),
           "naive": "accept" if all(s["estimate"] <= tau for s in seeds) else "reject", "seeds": seeds, "by_alpha": {}}
    for a in ALPHAS:
        out["by_alpha"][str(a)] = combine([s["by_alpha"][str(a)] for s in seeds])
        out["by_alpha"][str(a)]["max_ucb"] = max(s["by_alpha"][str(a)]["ucb"] for s in seeds)
        out["by_alpha"][str(a)]["max_lcb"] = max(s["by_alpha"][str(a)]["lcb"] for s in seeds)
    return out


def combine(parts):
    tw = [p["three_way"] for p in parts]
    return {"bound_rule": "accept" if all(p["bound_rule"] == "accept" for p in parts) else "not accepted",
            "three_way": "accept" if all(t == "accept" for t in tw) else ("reject" if "reject" in tw else "inconclusive")}


def joint(name, comps):
    out = {"components": comps, "naive": "accept" if all(c["naive"] == "accept" for c in comps.values()) else "reject",
           "by_alpha": {}}
    for a in ALPHAS:
        out["by_alpha"][str(a)] = combine([c["by_alpha"][str(a)] for c in comps.values()])
    return out


tests = {}
for tag in VARIANTS:
    v = ca["variants"][tag]
    rec, far = v["IV-2"]["recall"], v["IV-2"]["far"]
    load = v["IO-2"]["load"]
    loc = v["IE-2"]["process_localisation"]
    asr = v["IR-2"]["asr"]
    pgd = next(s for s in e2["models"][tag]["summary"] if s["tm"] == "TM-A" and abs(s["eps"] - 0.5) < 1e-9)
    t = {}
    t["IV-2"] = joint("IV-2", {"recall>=0.50": prop_test(rec["k"], rec["n"], 0.50, ">="),
                               "FAR<=0.25/h": rate_test(far["count"], far["exposure_h"], 0.25)})
    t["IO-2"] = rate_test(load["count"], load["exposure_h"], 6.0)
    t["IE-2"] = prop_test(loc["k"], loc["n"], 0.60, ">=")
    t["IR-2_union_PGD_or_greedy"] = prop_test(asr["k"], asr["n"], 0.20, "<=")
    t["IR-2_literal_PGD_only"] = prop_test(int(pgd["concealed"]), int(pgd["n"]), 0.20, "<=")
    tests[tag] = t

runs = [r for r in e4["runs"] if r["interface"] == "label" and r["Q"] == 10000]
d5 = e5["daily"]
shared = {
    "IP-2_operational_test_windows": {**fidelity_test(runs, 0.90, "test_dist"),
                                      "scope": "victim baseline_s0 only (E4 was not run against the hardened variant)"},
    "IP-2_attacker_query_distribution": {**fidelity_test(runs, 0.90, "attacker_dist"),
                                         "scope": "victim baseline_s0 only; secondary distribution reported in the table footer"},
    "IV-3_daily_PSI_monitor": {**joint("IV-3", {"TPR>=0.90": prop_test(d5["tp"], d5["n_pos"], 0.90, ">="),
                                                "FPR<=0.05": prop_test(d5["fp"], d5["n_neg"], 0.05, "<=")}),
                               "scope": "model-independent (PSI of input channels), identical for both variants"}}
s3 = e3["runs"]["simple"]["sanitiser"]
extra = {"ID-3_poison_screening_beta1pct": {
    **joint("ID-3", {"poison_detection>=0.80": prop_test(s3["poison_removed"], s3["poison_windows"], 0.80, ">="),
                     "clean_removal<=0.05": prop_test(s3["clean_removed"], s3["clean_windows"], 0.05, "<=")}),
    "scope": "baseline pipeline only; units are overlapping training windows, which are NOT independent, so the bounds are "
             "far too narrow; included only because the decision is unambiguous (estimates 0.174 and 0.195)",
    "not_reanalysed": "recall degradation <= 0.10 (paired comparison of clean and poisoned detectors) is not re-analysed here"}}


def outcome_table():
    rows = []
    for tag in VARIANTS:
        for name, tt in tests[tag].items():
            rows.append((tag, name, tt))
        for name, tt in shared.items():
            rows.append((tag, name, tt))
    rows.append(("baseline_s0", list(extra)[0], list(extra.values())[0]))
    table = []
    for tag, name, tt in rows:
        if name.startswith("IP-2") and tag == "hardened_s0":
            table.append({"variant": tag, "test": name, "status": "not evaluated for this variant"})
            continue
        for nm, obj in [(name, tt)] + [(f"{name}/{cn}", c) for cn, c in tt.get("components", {}).items()]:
            r = {"variant": tag, "test": nm, "level": "component" if "/" in nm else "requirement", "naive": obj["naive"]}
            for a in ALPHAS:
                r[f"bound_rule_{a}"] = obj["by_alpha"][str(a)]["bound_rule"]
                r[f"three_way_{a}"] = obj["by_alpha"][str(a)]["three_way"]
            table.append(r)
    return table


def implied_levels():
    """Class-I levels implied by each rule for the statistically tested services (assess_case.py logic)."""
    out = {}
    for tag in VARIANTS:
        t = tests[tag]
        ip = shared["IP-2_operational_test_windows"]
        rules = {"naive": lambda x: x["naive"] == "accept"}
        for a in ALPHAS:
            rules[f"bound_{a}"] = (lambda a: (lambda x: x["by_alpha"][str(a)]["bound_rule"] == "accept"))(a)
        lv = {}
        for rname, ok in rules.items():
            lv[rname] = {"IV": 2 if ok(t["IV-2"]) else 1, "IO": 2 if ok(t["IO-2"]) else 1, "IE": 2 if ok(t["IE-2"]) else 1,
                         "IR (union)": 2 if ok(t["IR-2_union_PGD_or_greedy"]) else 1,
                         "IR (literal PGD)": 2 if ok(t["IR-2_literal_PGD_only"]) else 1,
                         "IP": (2 if ok(ip) else 1) if tag == "baseline_s0" else "IP-2 evidence from baseline victim: " + ("2" if ok(ip) else "1")}
        three = {}
        for a in ALPHAS:
            three[f"three_way_{a}"] = {k: t[k2]["by_alpha"][str(a)]["three_way"] for k, k2 in
                                       (("IV-2", "IV-2"), ("IO-2", "IO-2"), ("IE-2", "IE-2"),
                                        ("IR-2 (union)", "IR-2_union_PGD_or_greedy"), ("IR-2 (literal PGD)", "IR-2_literal_PGD_only"))}
            three[f"three_way_{a}"]["IP-2"] = ip["by_alpha"][str(a)]["three_way"]
            three[f"three_way_{a}"]["IV-3"] = shared["IV-3_daily_PSI_monitor"]["by_alpha"][str(a)]["three_way"]
        out[tag] = {"levels": lv, "three_way_outcomes": three}
    return out


table = outcome_table()
changes = []
for r in table:
    if "naive" not in r:
        continue
    vals = {k: v for k, v in r.items() if k.startswith(("bound_rule", "three_way"))}
    b = {r[f"bound_rule_{a}"] for a in ALPHAS}
    w = {r[f"three_way_{a}"] for a in ALPHAS}
    naive_vs_bound = [a for a in ALPHAS if (r["naive"] == "accept") != (r[f"bound_rule_{a}"] == "accept")]
    if len(b) > 1 or len(w) > 1 or naive_vs_bound:
        changes.append({"variant": r["variant"], "test": r["test"], "naive": r["naive"],
                        "bound_rule_changes_with_alpha": len(b) > 1, "three_way_changes_with_alpha": len(w) > 1,
                        "naive_differs_from_bound_rule_at_alpha": naive_vs_bound,
                        **vals})

out = {"script": SCRIPT, "started_utc": started, "finished_utc": R.utc_now(), "runtime_s": time.time() - t0,
       "versions": R.versions(), "settings": {"alphas": ALPHAS, "equality_tolerance": EQ_TOL,
                                              "three_way_rule": "accept if the (1-alpha) bound clears tau; reject if the opposite "
                                                                "(1-alpha) bound violates tau (strictly); inconclusive otherwise",
                                              "joint_rule": "accept iff all components accept; reject iff any component rejects"},
       "inputs": {"case_assessment": "results/case_assessment.json", "pgd": "results/e2_evasion.json (TM-A, eps 0.5)",
                  "extraction": "results/e4b_extract.json (label-only, Q = 10000, seeds 0-2)",
                  "drift": "results/e5_drift.json (daily PSI rule)", "poisoning": "results/e3_poison_beta0.01.json"},
       "outcome_table": table, "decision_changes": changes, "implied_levels": implied_levels(),
       "per_variant_tests": tests, "shared_tests": shared, "additional_tests_for_completeness": extra}


def walk_boundary(scope, name, tt):
    found = []
    for a, b in tt.get("by_alpha", {}).items():
        if b.get("boundary_case"):
            found.append({"scope": scope, "test": name, "alpha": a, "lcb": b.get("lcb"), "ucb": b.get("ucb"),
                          "tau": tt.get("tau"), "three_way": b["three_way"]})
    for cn, c in tt.get("components", {}).items():
        found += walk_boundary(scope, f"{name}/{cn}", c)
    return found


out["boundary_cases"] = ([x for tag in VARIANTS for name, tt in tests[tag].items() for x in walk_boundary(tag, name, tt)]
                         + [x for name, tt in shared.items() for x in walk_boundary("shared", name, tt)]
                         + [x for name, tt in extra.items() for x in walk_boundary("baseline_s0", name, tt)])
R.jdump(out, "decision_sensitivity.json", SCRIPT)
for r in table:
    print(r)
print("boundary cases:", out["boundary_cases"])
print("runtime_s", round(time.time() - t0, 2))
