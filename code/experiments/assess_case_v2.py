"""Evidence-based re-assessment of the HAI case study against 3.CDI.ICS/G-4 (revision 2).

Every requirement of the class-I levels up to the reference profile carries an evidence class from
case_evidence.yaml: D (documented artefact), T (quantitative test), A (assumed: platform or operator
procedure), NA (not applicable, declared) or NE (not evaluated). Quantitative tests are decided with the
three-way rule of Section 3.4 (accept: the bound clears tau; reject: the opposite one-sided bound at the same
level violates tau; otherwise inconclusive), once under the procedure fixed in the evaluation plan
('registered') and once under the corrected procedure ('final', see the deviations below).

Ratings. A level is attained if all requirements of it and of the lower levels are met:
  demonstrated: D, NA and accepted (or, for tests without a threshold, reported) T count;
  conditional:  in addition, A counts (composite evaluation relying on platform and operator evidence).
The legacy services of the base profile 3.CD.3 (plus KA-1, added by the closure) and the assurance level G-4
are assumed from the hosting platform in both readings. The rating is int() of the attained profile
(Catalogue.interior), which also removes levels whose own requirements are met but whose necessary
conditions are not.

Inputs: results of E1-E5 and of the revision analyses (results/*.json). Writes results/case_assessment_v2.json.
"""
import hashlib, json, os, time
import numpy as np
import yaml
from scipy import stats as st
import aitcert_path  # noqa: F401
from aitcert import stats as S
from aitcert.model import Catalogue

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
A0 = 0.05                       # alpha_G for G-4
ALPHAS = (0.10, 0.05, 0.01)
VARIANTS = ("baseline_s0", "hardened_s0")
TAU = {"recall": 0.50, "far": 0.25, "load": 6.0, "asr": 0.20, "pdet": 0.80, "pdet_fpr": 0.05, "pdeg": 0.10,
       "ext": 0.90, "xfid": 0.60, "drift_tpr": 0.90, "drift_fpr": 0.05}
DELTA = 0.2                     # margin declared post hoc for proportion requirements (deviation D6)
TEST_HOURS = 24.0 + 23.0 + 17.333 + 36.0   # duration of test1-4 (IO-2 exposure, as in assess_case.py)
t0 = time.time()


def load(name):
    with open(os.path.join(R, name), encoding="utf8") as f:
        return json.load(f)


# ------------------------------------------------------------------ three-way decisions
def tw_prop(k, n, tau, direction, alpha):
    lcb, ucb = S.cp_lower(k, n, alpha), S.cp_upper(k, n, alpha)
    if direction == ">=":
        dec = "accept" if lcb >= tau else ("reject" if ucb < tau else "inconclusive")
        naive = k / n >= tau
    else:
        dec = "accept" if ucb <= tau else ("reject" if lcb > tau else "inconclusive")
        naive = k / n <= tau
    return {"k": int(k), "n": int(n), "estimate": k / n, "lcb": lcb, "ucb": ucb, "tau": tau, "direction": direction,
            "alpha": alpha, "decision": dec, "naive": "accept" if naive else "reject"}


def tw_rate(c, T, tau, alpha):
    lcb, ucb = S.poisson_lower(c, T, alpha), S.poisson_upper(c, T, alpha)
    dec = "accept" if ucb <= tau else ("reject" if lcb > tau else "inconclusive")
    return {"count": int(c), "exposure_h": T, "estimate": c / T, "lcb": lcb, "ucb": ucb, "tau": tau, "direction": "<=",
            "alpha": alpha, "decision": dec, "naive": "accept" if c / T <= tau else "reject"}


def tw_bounds(est, lcb, ucb, tau, direction, alpha, method):
    """Three-way decision from externally computed one-sided bounds (cluster-robust or bootstrap)."""
    if direction == ">=":
        dec = "accept" if lcb >= tau else ("reject" if ucb < tau else "inconclusive")
    else:
        dec = "accept" if ucb <= tau else ("reject" if lcb > tau else "inconclusive")
    return {"estimate": est, "lcb": lcb, "ucb": ucb, "tau": tau, "direction": direction, "alpha": alpha,
            "decision": dec, "method": method}


def conj(*decisions):
    """Decision of a conjunction of requirements (intersection-union test)."""
    d = [x for x in decisions if x != "reported"]
    if any(x == "reject" for x in d):
        return "reject"
    if all(x == "accept" for x in d):
        return "accept" if d else "reported"
    return "inconclusive"


def cp_effn_lower(k, n, deff, alpha):
    """Effective-sample-size Clopper-Pearson lower bound (Korn and Graubard), deff truncated at 1."""
    d = max(1.0, deff)
    ne, ke = n / d, k / d
    return 0.0 if ke <= 0 else float(st.beta.ppf(alpha, ke, ne - ke + 1))


def cp_effn_upper(k, n, deff, alpha):
    d = max(1.0, deff)
    ne, ke = n / d, k / d
    return 1.0 if ke >= ne else float(st.beta.ppf(1 - alpha, ke + 1, ne - ke))


def mcnemar_one_sided(b, c):
    """Exact one-sided McNemar p-value for H1: the first method has more hits (b > c)."""
    m = b + c
    return float(st.binom.sf(b - 1, m, 0.5)) if m else 1.0


def mcnemar_dependent(tag):
    """Exact McNemar test of the model's process attribution against the constant explainer P1, with the effective
    sample size of Section 3.7: the dependency-graph variance of the paired differences over attacks that share a
    primitive gives a design effect deff (truncated at 1), and the exact one-sided test is evaluated at b/deff and
    c/deff, i.e. p = I_{1/2}(b/deff, c/deff + 1)."""
    from rev_lib import load_attacks
    prims = {a["attack_id"]: set(a["primitives"]) for a in load_attacks(True)}
    recs = e1b["variants"][tag]["records"]
    m = np.array([r["top_sum"] in r["target_processes"] for r in recs], float)
    c = np.array(["P1" in r["target_processes"] for r in recs], float)
    ids = [r["attack_id"] for r in recs]
    dep = np.array([[1.0 if prims[a] & prims[b_] else 0.0 for b_ in ids] for a in ids])
    d = m - c
    r = d - d.mean()
    deff = max(1.0, float(r @ dep @ r) / float(r @ r)) if float(r @ r) > 0 else 1.0
    b, cc = int(((m == 1) & (c == 0)).sum()), int(((m == 0) & (c == 1)).sum())
    p = float(st.beta.cdf(0.5, b / deff, cc / deff + 1)) if b > 0 else 1.0
    p_rev = float(st.beta.cdf(0.5, cc / deff, b / deff + 1)) if cc > 0 else 1.0
    pairs = sorted({tuple(sorted((a, b_))) for i, a in enumerate(ids) for j, b_ in enumerate(ids)
                    if i != j and d[i] != 0 and d[j] != 0 and prims[a] & prims[b_]})
    return {"b": b, "c": cc, "deff": deff, "p": p, "p_reverse": p_rev, "discordant_pairs_sharing_a_primitive": pairs}


def n_min_prop(tau, direction, delta, alpha, beta=0.2):
    """Stable exact sample size of the bound rule (Section 3.4); '<=' requirements are mirrored."""
    t = tau if direction == ">=" else 1.0 - tau
    if t + delta >= 1.0:   # alternative at the boundary: zero-failure rule
        return S.n_zero_failure(alpha, 1.0 - t)
    return S.n_exact_power_stable(t, delta, alpha, beta)


def max_lcb(n, alpha):
    """Largest attainable one-sided lower bound with n units (all successes)."""
    return S.cp_lower(n, n, alpha)


# ------------------------------------------------------------------ inputs
cat = Catalogue.load(True)
ref = load("case_assessment.json")["reference_profile"]
required = {s: ref["services"][s] for s in ("ID", "IR", "IV", "IE", "IP", "IO", "IM")}
platform = {s: l for s, l in ref["services"].items() if s not in required}
G = ref["G"]
evidence = yaml.safe_load(open(os.path.join(HERE, "case_evidence.yaml"), encoding="utf8"))
e1 = {v: load(f"e1_{v}.json") for v in VARIANTS}
e2, e2b, pgd = load("e2_evasion.json"), load("e2b_greedy.json"), load("pgd_sanity.json")
ir1, thw = load("ir1_query_chain.json"), load("transfer_hardaware.json")
e3, id3 = load("e3_poison_beta0.01.json"), load("id3_units.json")
e4b, ip2, e4c = load("e4b_extract.json"), load("ip2_reexamination.json"), load("e4c_extract_deployed.json")
e5, e5b = load("e5_drift.json"), load("e5b_drift_retrained.json")
e1b, cb, e1c = load("e1b_localisation.json"), load("cluster_bootstrap.json"), load("e1c_false_alarm_dispersion.json")
im1, im2 = load("im_baseline_s0.json"), load("im2_v2.json")


# ------------------------------------------------------------------ IR-2 per-attack aggregation
def concealed_sets(tag, eps_max=0.5, tm="TM-A"):
    """Per-attack concealment by each white-box attack over the nested ball eps <= eps_max."""
    out = {}
    def add(name, rows, cond):
        d = {}
        for r in rows:
            if cond(r) and r["eps"] <= eps_max + 1e-9:
                d[r["attack_id"]] = d.get(r["attack_id"], False) or bool(r["concealed"])
        out[name] = d
    add("pgd_registered", e2["models"][tag]["rows"], lambda r: r["tm"] == tm)
    add("greedy", e2b["models"][tag]["rows"], lambda r: r["tm"] == tm and r["mode"] == "white-box")
    add("pgd_improved", pgd["variants"][tag]["rows"]["improved"], lambda r: True)
    single = {}
    for r in e2["models"][tag]["rows"]:
        if r["tm"] == tm and abs(r["eps"] - eps_max) < 1e-9:
            single[r["attack_id"]] = bool(r["concealed"])
    out["pgd_registered_single_eps"] = single
    ids = set(out["pgd_registered"])
    for name, d in out.items():
        assert set(d) == ids, (tag, name, "attack sets differ")
    return out, sorted(ids)


def ir2_tests(tag, alpha):
    cs, ids = concealed_sets(tag)
    n = len(ids)
    union = {a: any(cs[k][a] for k in ("pgd_registered", "greedy", "pgd_improved")) for a in ids}
    res = {
        "pgd_single_eps0.5": tw_prop(sum(cs["pgd_registered_single_eps"].values()), n, TAU["asr"], "<=", alpha),
        "pgd_registered_ball": tw_prop(sum(cs["pgd_registered"].values()), n, TAU["asr"], "<=", alpha),
        "greedy_ball": tw_prop(sum(cs["greedy"].values()), n, TAU["asr"], "<=", alpha),
        "pgd_improved_ball": tw_prop(sum(cs["pgd_improved"].values()), n, TAU["asr"], "<=", alpha),
        "strongest_union_ball": tw_prop(sum(union.values()), n, TAU["asr"], "<=", alpha),
    }
    res["concealed_by_union"] = sorted(a for a in ids if union[a])
    res["pgd_registered_by_eps"] = {str(e): sum(1 for r in e2["models"][tag]["rows"] if r["tm"] == "TM-A" and abs(r["eps"] - e) < 1e-9 and r["concealed"])
                                     for e in (0.1, 0.25, 0.5, 1.0)}
    res["pgd_improved_by_eps"] = {str(r["eps"]): r["k"] for r in pgd["variants"][tag]["improved_pgd_by_eps"]}
    res["unbounded_eps5_TM-A"] = {k: pgd["variants"][tag]["unbounded"][k] for k in ("k", "n", "lcb95", "ucb95")}
    return res


# ------------------------------------------------------------------ per-variant tests
def variant_tests(tag, alpha):
    a = e1[tag]["aggregate"]
    t = {}
    # IV-1: reported with two-sided 95% intervals
    lo, hi = S.cp_interval(a["events_detected"], a["events"], 0.05)
    t["IV-1.report"] = {"decision": "reported", "recall": a["events_detected"] / a["events"], "recall_ci95": [lo, hi],
                        "far_per_h": a["false_alarms"] / a["normal_hours"],
                        "far_ci95": [S.poisson_lower(a["false_alarms"], a["normal_hours"], 0.025), S.poisson_upper(a["false_alarms"], a["normal_hours"], 0.025)]}
    # IV-2 recall: registered exact CP; final cluster-robust (dependency-graph effective-n CP, primitive dependence)
    reg = tw_prop(a["events_detected"], a["events"], TAU["recall"], ">=", alpha)
    cbr = cb["summary"][tag]["event_recall"]
    deff = cbr["dependency_graph_deff"]
    lcb_c = cp_effn_lower(a["events_detected"], a["events"], deff, alpha)
    ucb_c = cp_effn_upper(a["events_detected"], a["events"], deff, alpha)
    if abs(alpha - 0.05) < 1e-12:
        assert abs(lcb_c - cbr["dependency_graph_effn_cp"][0]) < 1e-6, (tag, lcb_c, cbr["dependency_graph_effn_cp"][0])
    fin = tw_bounds(reg["estimate"], lcb_c, ucb_c, TAU["recall"], ">=", alpha, f"effective-n CP, design effect {deff:.2f} (attack primitives)")
    t["IV-2.recall"] = {"registered": reg, "final": fin}
    # IV-2 false-alarm rate: registered exact Poisson; final quasi-Poisson where hourly counts are overdispersed
    c, T = a["false_alarms"], a["normal_hours"]
    reg = tw_rate(c, T, TAU["far"], alpha)
    d = e1c["a_dispersion"][tag]
    y, w = np.asarray(d["hourly_counts"], float), np.asarray(d["hourly_exposure_h"], float)
    lam = y.sum() / w.sum()
    phi = float(np.sum((y - lam * w) ** 2 / (lam * w)) / (len(y) - 1))
    f = max(1.0, phi)
    z = st.norm.ppf(1 - alpha)
    se = np.sqrt(f / c)                      # log-scale standard error of a quasi-Poisson rate
    q_lcb, q_ucb = lam * np.exp(-z * se), lam * np.exp(z * se)
    fin = tw_bounds(c / T, min(q_lcb, reg["lcb"]) if f > 1 else reg["lcb"], max(q_ucb, reg["ucb"]) if f > 1 else reg["ucb"],
                    TAU["far"], "<=", alpha, f"exact Poisson; dispersion X2/df = {phi:.2f}" + (" (quasi-Poisson widening)" if f > 1 else ""))
    fin["dispersion"] = phi
    t["IV-2.far"] = {"registered": reg, "final": fin}
    # IO-2 workload (all alarms per hour over the test period)
    n_alarms = round(a["alarms_per_h"] * TEST_HOURS)
    r = tw_rate(n_alarms, TEST_HOURS, TAU["load"], alpha)
    t["IO-2.load"] = {"registered": r, "final": r}
    # IE-2 process localisation: registered absolute threshold; final adds the trivial-explainer comparison
    loc = e1[tag]["localisation"]
    k = sum(l["process_hit"] for l in loc)
    reg = tw_prop(k, len(loc), TAU["xfid"], ">=", alpha)
    sm = e1b["summary"][tag]
    kc, nc = sm["constant_P1"][0], sm["constant_P1"][1]
    mc = sm["mcnemar_sum_vs_P1"]
    p = mcnemar_one_sided(mc["model_only_hits_b"], mc["comparator_only_hits_c"])
    p_rev = mcnemar_one_sided(mc["comparator_only_hits_c"], mc["model_only_hits_b"])
    assert abs(p - mc["p_one_sided_exact"]) < 1e-12
    trivial = tw_prop(kc, nc, TAU["xfid"], ">=", alpha)
    comp_iid = "accept" if p <= alpha else ("reject" if p_rev <= alpha else "inconclusive")
    md = mcnemar_dependent(tag)
    assert (md["b"], md["c"]) == (mc["model_only_hits_b"], mc["comparator_only_hits_c"])
    comp = "accept" if md["p"] <= alpha else ("reject" if md["p_reverse"] <= alpha else "inconclusive")
    t["IE-2.xfid"] = {"registered": reg,
                      "final": {"decision": conj(reg["decision"], comp), "absolute": reg, "trivial_constant_P1": trivial,
                                "tau_admissible": kc / nc < TAU["xfid"],
                                "mcnemar_vs_constant_P1": {"b_model_only": mc["model_only_hits_b"], "c_constant_only": mc["comparator_only_hits_c"],
                                                           "p_one_sided": p, "p_reverse": p_rev, "decision_independent_units": comp_iid,
                                                           "dependence": md, "p_dependence_robust": md["p"], "decision": comp},
                                "random_channel_proportional_expected_accuracy": sm["channel_proportional_random"]["expected_accuracy"],
                                "tag_top5": sm["tag_top5"]["observed"][:2], "tag_top5_chance_rate": sm["tag_top5"]["chance_rate"]}}
    # IR-1 (no threshold): transfer and query-based black-box attacks, reported
    tr = [r for r in e2b["models"][tag]["rows"] if r["mode"] == "transfer" and r["tm"] == "TM-A" and abs(r["eps"] - 0.5) < 1e-9]
    ktr, ntr = sum(r["concealed"] for r in tr), len(tr)
    t["IR-1.transfer"] = {"decision": "reported", "k": ktr, "n": ntr, "ci95": list(S.cp_interval(ktr, ntr, 0.05))}
    if tag == "hardened_s0":
        hw = [r for r in thw["greedy_transfer_by_eps"] if abs(r["eps"] - 0.5) < 1e-9][0]
        t["IR-1.transfer"]["hardening_aware_surrogate"] = {"k": hw["k"], "n": hw["n"], "ci95": list(S.cp_interval(hw["k"], hw["n"], 0.05))}
    qr = ir1["victims"][tag]["runs"]
    kq, nq = max(r["transfer_k"] for r in qr), qr[0]["transfer_n"]
    t["IR-1.query"] = {"decision": "reported", "k_max_over_interfaces_and_budgets": kq, "n": nq, "ci95": list(S.cp_interval(kq, nq, 0.05)),
                       "runs": [{k2: r[k2] for k2 in ("interface", "Q", "balanced_agreement", "transfer_k", "transfer_n")} for r in qr],
                       "registered_available": tag == "baseline_s0"}
    # IR-2
    t["IR-2.asr"] = ir2_tests(tag, alpha)
    t["IR-2.asr"]["registered"] = t["IR-2.asr"]["pgd_registered_ball"]
    t["IR-2.asr"]["final"] = t["IR-2.asr"]["strongest_union_ball"]
    return t


def shared_tests(alpha):
    """Tests of the data pipeline, the drift monitor, extraction and the signed release (baseline pipeline)."""
    t = {}
    # ID-3 poison detection (registered: window-level CP; final: 7 source traces as units, cluster bootstrap)
    s3 = e3["runs"]["simple"]["sanitiser"]
    reg = tw_prop(s3["poison_removed"], s3["poison_windows"], TAU["pdet"], ">=", alpha)
    tr = id3["traces"]["simple"]
    lo, hi = tr["cluster_bootstrap"]["pooled_ci95"]
    fin = tw_bounds(tr["pooled_fraction"], tr["cluster_bootstrap"]["pooled_lcb95_one_sided"], hi, TAU["pdet"], ">=", 0.05,
                    "percentile cluster bootstrap over the 7 source traces (B = 10000)")
    fin["max_attainable_lcb_7_traces"] = max_lcb(7, alpha)
    fin["oracle_max_window_level"] = tr["oracle_max_pdet_window_level"]
    clean = tw_prop(s3["clean_removed"], s3["clean_windows"], TAU["pdet_fpr"], "<=", alpha)
    t["ID-3.pdet"] = {"registered": {**reg, "decision": conj(reg["decision"], clean["decision"]), "clean_removal": clean},
                      "final": {**fin, "decision": conj(fin["decision"], clean["decision"]), "clean_removal": clean}}
    # ID-3 recall degradation (paired exact bound d <= p10)
    dg = id3["degradation"]["tests"]
    def deg(key):
        x = dg[key]
        ucb = S.cp_upper(x["n10_clean_only"], x["n"], alpha)
        lcb = S.cp_lower(x["n10_clean_only"], x["n"], alpha / 2) - S.cp_upper(x["n01_other_only"], x["n"], alpha / 2)
        return tw_bounds(x["degradation_estimate"], lcb, ucb, TAU["pdeg"], "<=", alpha, f"paired exact bound, {key}") | {"n10": x["n10_clean_only"], "n01": x["n01_other_only"], "n": x["n"]}
    t["ID-3.pdeg"] = {"registered": deg("simple/no_sanitisation/vs_baseline_s0_e1"), "final": deg("simple/no_sanitisation/vs_clean_e3_recipe")}
    # IV-3 daily PSI drift test (pre-specified); hourly monitors exploratory
    dd = e5["daily"]
    tpr = tw_prop(dd["tp"], dd["n_pos"], TAU["drift_tpr"], ">=", alpha)
    fpr = tw_prop(dd["fp"], dd["n_neg"], TAU["drift_fpr"], "<=", alpha)
    reg = {"decision": conj(tpr["decision"], fpr["decision"]), "tpr": tpr, "fpr": fpr,
           "feasible": max_lcb(dd["n_pos"], alpha) >= TAU["drift_tpr"] and S.cp_upper(0, dd["n_neg"], alpha) <= TAU["drift_fpr"]}
    rm = e5b["residual_monitor_hourly"]
    ex = {"tpr": tw_prop(rm["tp"], rm["n_pos"], TAU["drift_tpr"], ">=", alpha), "fpr": tw_prop(rm["fp"], rm["n_neg"], TAU["drift_fpr"], "<=", alpha),
          "auc": rm["auc"], "detector": "retrained on the 60 shared channels (E5b)"}
    ex["decision"] = conj(ex["tpr"]["decision"], ex["fpr"]["decision"])
    rm0 = e5["residual_monitor_hourly"]
    ex0 = {"tpr": tw_prop(rm0["tp"], rm0["n_pos"], TAU["drift_tpr"], ">=", alpha), "fpr": tw_prop(rm0["fp"], rm0["n_neg"], TAU["drift_fpr"], "<=", alpha)}
    ex0["decision"] = conj(ex0["tpr"]["decision"], ex0["fpr"]["decision"])
    t["IV-3.daily"] = {"registered": reg, "final": reg, "exploratory_hourly_residual_E5b": ex, "exploratory_hourly_residual_E5": ex0}
    # IP-2 extraction fidelity at Q = 1e4, label-only (baseline victim)
    runs = [r for r in e4b["runs"] if r["interface"] == "label" and r["Q"] == 10000]
    u_test = max(r["test_dist"]["ucb95"] for r in runs)
    u_att = max(r["attacker_dist"]["ucb95"] for r in runs)
    att_lcb = min(S.cp_lower(round(r["attacker_dist"]["balanced"] * (r["attacker_dist"]["n_pos"] + r["attacker_dist"]["n_neg"])),
                             r["attacker_dist"]["n_pos"] + r["attacker_dist"]["n_neg"], alpha) for r in runs)
    reg = {"decision": "accept" if u_test <= TAU["ext"] else "inconclusive", "max_ucb95_operational_cp": u_test,
           "estimates_operational": [r["test_dist"]["balanced"] for r in runs], "note": "evaluation distribution not named in the plan"}
    attacker = {"decision": "reject" if att_lcb > TAU["ext"] else ("accept" if u_att <= TAU["ext"] else "inconclusive"),
                "max_ucb95": u_att, "min_lcb_approx": att_lcb, "estimates": [r["attacker_dist"]["balanced"] for r in runs]}
    rr = [r for r in ip2["recompute"]["runs"] if r["interface"] == "label" and r["Q"] == 10000]
    u_dep = max(max(r["moving_block_ucb95"], r["file_cluster_ucb95"]) for r in rr)
    fin = {"decision": "accept" if u_dep <= TAU["ext"] else "inconclusive", "max_ucb95_dependent_data": u_dep,
           "estimates": [r["balanced"] for r in rr], "method": "moving-block (600 s) and test-file cluster bootstrap on all 360,956 test windows"}
    e4c_q = {str(r.get("Q")): r for r in e4c["runs"]} if isinstance(e4c.get("runs"), list) else {}
    t["IP-2.ext"] = {"registered": reg, "final": fin, "attacker_query_distribution": attacker,
                     "exploratory_deployed_oracle_E4c": {"post_hoc": True, "runs": len(e4c.get("runs", []))}}
    # IM-2 signed release
    reg = {"decision": "accept" if (im1["signature_valid"] and im1["tamper_detected"]) else "reject",
           "flaw": "v1 verifier trusts the public key shipped in the envelope; a foreign-key re-signature is accepted",
           "v1_on_foreign_key_forgery": im2["tests"]["iii_foreign_key_resigned"]["v1_logic_on_same_forgery"]}
    fin = {"decision": "accept" if im2["all_tests_as_expected"] else "reject",
           "tests": {k: v["outcome"] for k, v in im2["tests"].items()}, "anchor_sha256": im2["trust_anchor"]["sha256"]}
    t["IM-2.sig"] = {"registered": reg, "final": fin}
    return t


def hardened_ip2(alpha):
    """IP-2 for the hardened victim: only the extraction step of the IR-1 chain is available (one run per budget;
    E4b was not repeated). Conservative bound of balanced agreement: mean of Bonferroni one-sided CP bounds."""
    r = [x for x in ir1["victims"]["hardened_s0"]["runs"] if x["interface"] == "label" and x["Q"] == 10000][0]
    npos = r["victim_pos"]; nneg = r["n_eval"] - npos
    kp, kn = round(r["agree_pos"] * npos), round(r["agree_neg"] * nneg)
    ucb = 0.5 * (S.cp_upper(kp, npos, alpha / 2) + S.cp_upper(kn, nneg, alpha / 2))
    return {"decision": "accept" if ucb <= TAU["ext"] else "inconclusive", "balanced": r["balanced_agreement"], "ucb": ucb,
            "note": "single extraction run of the IR-1 chain (E4 procedure); windows treated as independent; exploratory"}


# ------------------------------------------------------------------ requirement status and levels
def requirement_met(req, tests, proc, reading):
    st_ = req["status"]
    if st_ in ("D", "NA"):
        return True
    if st_ == "A":
        return reading == "conditional"
    if st_ == "NE":
        return False
    decs = []
    for key in req["tests"]:
        x = tests.get(key)
        if x is None:
            return False
        if "registered" in x and proc in x:
            d = x[proc]["decision"] if isinstance(x[proc], dict) else x[proc]
        else:
            d = x["decision"]
            if proc == "registered" and x.get("registered_available") is False:
                return False
        decs.append(d)
    return conj(*decs) in ("accept", "reported")


def attained(tests, proc, reading):
    levels, detail = {}, {}
    for s, lmax in required.items():
        lv = 0
        for l in range(1, lmax + 1):
            reqs = evidence.get(f"{s}-{l}", [])
            ok = [(r["req"], requirement_met(r, tests, proc, reading)) for r in reqs]
            detail[f"{s}-{l}"] = {k: v for k, v in ok}
            if reqs and all(v for _, v in ok):
                lv = l
            else:
                break
        levels[s] = lv
    return levels, detail


def rating(levels):
    prof = {**platform, **{s: l for s, l in levels.items() if l > 0}}
    inner = cat.interior(prof, G)
    return {s: inner.get(s, 0) for s in required}, all(inner.get(s, 0) >= l for s, l in platform.items())


def fmt(p):
    return "{" + ", ".join(f"{s}-{l}" for s, l in p.items() if l > 0) + "}"


out = {"reference_profile": {"id": "3.CDI.ICS/G-4", "required_class_I": required, "platform_assumed": platform, "G": G,
                             "formatted_required": fmt(required)},
       "tau": TAU, "delta_post_hoc": DELTA, "alpha_primary": A0, "variants": {}, "shared": {}, "by_alpha": {}}
for alpha in ALPHAS:
    sh = shared_tests(alpha)
    for tag in VARIANTS:
        tv = variant_tests(tag, alpha)
        tests = {**sh, **tv}
        if tag == "hardened_s0":
            tests["IP-2.ext"] = {"registered": {"decision": "not evaluated"}, "final": hardened_ip2(alpha)}
            for k in ("ID-3.pdet", "IM-2.sig"):
                tests[k] = {**sh[k], "note": "pipeline and release evidence of the baseline (same data, same signing procedure)"}
            tests["ID-3.pdeg"] = {"registered": {"decision": "not evaluated"}, "final": {"decision": "not evaluated"},
                                  "note": "the poisoning drill trained the baseline recipe only"}
        res = {}
        for proc in ("registered", "final"):
            for reading in ("demonstrated", "conditional"):
                lv, det = attained(tests, proc, reading)
                rt, plat_ok = rating(lv)
                res[f"{proc}/{reading}"] = {"own_requirements": lv, "rating": rt, "formatted": fmt(rt), "platform_valid": plat_ok,
                                            "blocked_by_conditions": {s: (lv[s], rt[s]) for s in lv if rt[s] < lv[s]},
                                            "gaps": {s: (rt[s], required[s]) for s in required if rt[s] < required[s]},
                                            "requirements": det}
        if abs(alpha - A0) < 1e-12:
            out["variants"][tag] = {"tests": tv, "results": res}
            if tag == "hardened_s0":
                out["variants"][tag]["tests"]["IP-2.ext"] = tests["IP-2.ext"]
        out["by_alpha"].setdefault(str(alpha), {})[tag] = {k: v["formatted"] for k, v in res.items()} | {
            "decisions_final": {k: (x["final"]["decision"] if isinstance(x.get("final"), dict) else x.get("decision")) for k, x in tests.items()},
            "decisions_registered": {k: (x["registered"]["decision"] if isinstance(x.get("registered"), dict) else x.get("decision")) for k, x in tests.items()}}
    if abs(alpha - A0) < 1e-12:
        out["shared"] = sh

# ------------------------------------------------------------------ feasibility and sample sizes (TE-AI-1)
feas = {}
for name, tau, dirn, n in (("IV-2 recall", TAU["recall"], ">=", 58), ("IE-2 localisation", TAU["xfid"], ">=", 38),
                           ("IR-2 attack success", TAU["asr"], "<=", 38), ("IV-3 drift TPR (days)", TAU["drift_tpr"], ">=", 9),
                           ("ID-3 poison detection (traces)", TAU["pdet"], ">=", 7)):
    row = {"tau": tau, "direction": dirn, "n": n}
    for d in (0.1, 0.2):
        row[f"n_min_delta{d}"] = n_min_prop(tau, dirn, d, A0)
    t = tau if dirn == ">=" else 1 - tau
    row["passable_with_all_successes"] = max_lcb(n, A0) >= t
    feas[name] = row
feas["IV-3 drift FPR (days)"] = {"tau": TAU["drift_fpr"], "n": 1, "smallest_ucb_with_zero_failures": S.cp_upper(0, 1, A0),
                                 "passable": S.cp_upper(0, 1, A0) <= TAU["drift_fpr"], "n_zero_failure": S.n_zero_failure(A0, TAU["drift_fpr"])}
feas["IV-2 FAR (hours)"] = {"tau": TAU["far"], "zero_alarm_hours_needed": float(st.chi2.ppf(1 - A0, 2) / 2 / TAU["far"])}
out["feasibility"] = feas

# ------------------------------------------------------------------ retrospective data integrity (ID-1.3; ID-2.2 stays NE)
man = json.load(open(os.path.join(ROOT, "data/raw/hai/MANIFEST.json"), encoding="utf8"))
chk = []
for m in man:
    if not (m["file"].startswith("hai-22.04/") or m["file"].startswith("hai-23.05/")):
        continue
    p = os.path.join(ROOT, "data/raw/hai", m["file"])
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    chk.append(h.hexdigest() == m["sha256"])
out["retrospective_integrity_check"] = {"files": len(chk), "all_match_manifest": all(chk), "date": time.strftime("%Y-%m-%d")}

# ------------------------------------------------------------------ deviations from the evaluation plan
out["deviations"] = [
    {"id": "D1", "item": "IR-2 attack", "registered": "white-box adaptive PGD within L_inf <= 0.5 SD (TM-A)",
     "change": "attack success over the whole ball (eps in {0.1, 0.25, 0.5}) with the strongest attack: union of the planned PGD, an improved PGD (restarts, warm start, step schedule) and greedy prediction replay",
     "reason": "the planned PGD success fell with the budget (5, 4, 0, 0 of 38), the signature of a failed optimiser; adaptive-attack guidance requires the strongest available attack",
     "decided": "greedy attack after E2; improved PGD and nested aggregation in the revision"},
    {"id": "D2", "item": "IP-2 evaluation distribution", "registered": "fidelity <= 0.90 at Q = 10^4, label-only; distribution not named",
     "change": "surrogates retrained after the first ones proved under-trained (E4b); operational test windows named as the evaluation distribution; attacker's query distribution reported; dependent-data bounds; extraction against the deployed alarm (E4c) exploratory",
     "reason": "the plan left the distribution open; fidelity on the task distribution is the conventional definition", "decided": "after E4 (redesign E4b) and in the revision"},
    {"id": "D3", "item": "ID-3 evaluation units", "registered": "poison detection rate >= 0.80 (window level), degradation <= 0.10",
     "change": "poison detection bounded with the 7 source traces as units; exact paired bound for degradation; cross-fitted sanitisers (E3b, E3c) exploratory",
     "reason": "10,308 windows are replicas of 7 traces and are not independent", "decided": "revision"},
    {"id": "D4", "item": "IV-3", "registered": "daily drift alarm TPR >= 0.90 with daily FPR <= 0.05",
     "change": "none to the decision; hourly monitors exploratory; drift re-estimated with a detector retrained on the 60 channels shared by both campaigns (E5b)",
     "reason": "only one in-distribution day exists, so the daily test cannot be passed; E5 had imputed eight renamed tags", "decided": "revision"},
    {"id": "D5", "item": "IE-2 trivial baseline", "registered": "process-localisation accuracy >= 0.60",
     "change": "the explainer must also beat the best constant explainer (exact one-sided McNemar test at alpha_G)",
     "reason": "a constant 'P1' explainer reaches 0.79, above the planned threshold", "decided": "revision"},
    {"id": "D6", "item": "margin delta", "registered": "not declared", "change": "delta = 0.2 declared for proportion requirements",
     "reason": "TE-AI-1 requires n >= n_min(alpha_G, delta)", "decided": "revision"},
    {"id": "D7", "item": "dependence of evaluation units", "registered": "exact binomial and Poisson bounds",
     "change": "cluster-robust bound for event recall and for the paired comparison with the constant explainer (shared attack primitives); dispersion check and quasi-Poisson widening of hourly false-alarm counts",
     "reason": "attacks reuse primitives; Section 3.4 requires cluster or block bounds for dependent units", "decided": "revision"},
    {"id": "D8", "item": "IM-2 verification", "registered": "signature and tamper test (v1)",
     "change": "verification against a trust anchor pinned outside the envelope (v2) with negative tests",
     "reason": "the v1 verifier accepted a model re-signed with a foreign key", "decided": "revision"},
]
out["meta"] = {"script": "assess_case_v2.py", "evidence_file": "case_evidence.yaml", "runtime_s": round(time.time() - t0, 2),
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "numpy": np.__version__}
json.dump(out, open(os.path.join(R, "case_assessment_v2.json"), "w", encoding="utf8"), indent=1, default=float)

# ------------------------------------------------------------------ console summary
for tag in VARIANTS:
    res = out["variants"][tag]["results"]
    print(f"== {tag}")
    for k, v in res.items():
        print(f"  {k:26s} own {fmt(v['own_requirements']):48s} rating {v['formatted']:48s} gaps {v['gaps']}")
    tv = out["variants"][tag]["tests"]
    for k in ("IV-2.recall", "IV-2.far", "IE-2.xfid", "IO-2.load", "IR-2.asr"):
        r_, f_ = tv[k]["registered"], tv[k]["final"]
        print(f"  {k:12s} registered {r_['decision']:12s} final {f_['decision']}")
    ir = tv["IR-2.asr"]
    print("   IR-2:", {k: (ir[k]["k"], ir[k]["n"], round(ir[k]["ucb"], 3), ir[k]["decision"]) for k in ("pgd_single_eps0.5", "pgd_registered_ball", "greedy_ball", "pgd_improved_ball", "strongest_union_ball")})
for k in ("ID-3.pdet", "ID-3.pdeg", "IV-3.daily", "IP-2.ext", "IM-2.sig"):
    print(f"  {k:12s} registered {out['shared'][k]['registered']['decision']:12s} final {out['shared'][k]['final']['decision']}")
print("hardened IP-2:", out["variants"]["hardened_s0"]["tests"]["IP-2.ext"]["final"])
print("by alpha:", json.dumps({a: {t: v[t]["final/conditional"] for t in v} for a, v in out["by_alpha"].items()}, indent=1))
print("feasibility:", json.dumps(feas, indent=1, default=float))
print("integrity:", out["retrospective_integrity_check"])
