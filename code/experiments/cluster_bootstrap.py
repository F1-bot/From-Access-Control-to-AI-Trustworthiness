"""Cluster-robust sensitivity of the attack-level acceptance tests (major revision, reviewer request).

The 58 HAI 22.04 attacks are a designed set that reuses 37 attack primitives (33 single-primitive attacks and
25 two-primitive combinations), so attack-level outcomes need not be independent. For three attack-level
proportions of the pre-registered case study,

  * event recall (58 attacks; requirement >= 0.50; 38/58 baseline, 36/58 hardened),
  * IR-2 concealment, union of PGD and greedy white-box attacks (TM-A, eps = 0.5; detected attacks;
    requirement <= 0.20; 9/38 baseline and 6/36 hardened),
  * IE-2 process localisation (detected attacks; requirement >= 0.60; 37/38 and 36/36),

this script computes one-sided 95% bounds that respect the dependence and compares them with the exact
Clopper-Pearson (CP) bounds used in the manuscript.

Scheme selection (documented in the output):
  1. Connected components of the attack-primitive graph are the natural clusters, but they give one giant
     component (42 of 58 attacks; Kish effective number of clusters 1.9). A component bootstrap is therefore
     not informative; it is reported only as a diagnostic.
  2. Primitive-level schemes:
     (P) product-weight ("pigeonhole") bootstrap (Owen 2007; Owen and Eckles 2012): independent Poisson(1)
         weight per primitive, attack weight = product of the weights of its primitives;
     (F) classical cluster bootstrap with clusters = first listed primitive (34 clusters);
     (D) dependency-graph (multi-membership cluster-robust) variance: V = n^-2 sum_ij D_ij r_i r_j with
         D_ij = 1 if i = j or attacks i and j share a primitive, r_i = y_i - p_hat; the design effect
         deff = V / (p_hat (1 - p_hat) / n) enters an effective-sample-size Clopper-Pearson bound with
         n_eff = n / max(1, deff) (Korn and Graubard 1998).
     For (P) and (F) both the one-sided percentile bound and the effective-n CP bound with the bootstrap
     design effect are reported.
  3. Reference / completeness: iid attack bootstrap, exact test-file bootstrap (4 clusters, all 4^4
     resamples), worst-case bound for degenerate proportions (each connected component = one unit), and the
     literal reading of primitive "AP3" (A415) as a separate primitive.
  4. Calibration: a simulation under a logistic primitive random-effects model on the actual attack-primitive
     design, y_i ~ Bernoulli(expit(b0 + sum_{p in P_i} a_p)), a_p ~ N(0, s^2), with b0 set so that the
     design-averaged success probability equals the observed proportion, estimates the one-sided coverage of
     every method for s in {0, 0.5, 1, 1.5, 2}, and the mean dependency-graph design effect at each s
     (to locate the observed data).

Writes results/cluster_bootstrap.json. Usage: python cluster_bootstrap.py [--B 100000] [--seed 20261012]
[--nsim 4000] [--bsim 2000]"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
from scipy import optimize, special
from scipy import stats as st

import rev_lib as R
from aitcert_path import stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=100000, help="bootstrap replicates for the observed data")
ap.add_argument("--seed", type=int, default=20261012)
ap.add_argument("--nsim", type=int, default=4000, help="simulated data sets per setting and sigma")
ap.add_argument("--bsim", type=int, default=2000, help="bootstrap replicates per simulated data set")
args = ap.parse_args()
t0, started = time.time(), R.utc_now()
SCRIPT = "cluster_bootstrap.py"
ALPHA = 0.05
VARIANTS = ("baseline_s0", "hardened_s0")
TM, EPS = "TM-A", 0.5
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

attacks = R.load_attacks(fix_primitives=True)
attacks_literal = R.load_attacks(fix_primitives=False)
ids = [a["attack_id"] for a in attacks]
prim = {a["attack_id"]: a["primitives"] for a in attacks}
prim_literal = {a["attack_id"]: a["primitives"] for a in attacks_literal}
tfile = {a["attack_id"]: a["test_file"] for a in attacks}
e2 = json.load(open(os.path.join(R.RES, "e2_evasion.json")))
e2b = json.load(open(os.path.join(R.RES, "e2b_greedy.json")))
seeds = iter(np.random.SeedSequence(args.seed).spawn(4096))


def new_rng():
    return np.random.default_rng(next(seeds))


# ---------------------------------------------------------------- outcomes per unit
def outcomes(tag):
    e1 = json.load(open(os.path.join(R.RES, f"e1_{tag}.json")))
    det = dict(zip(ids, e1["detected_vector"]))
    out = {"event_recall": {i: int(det[i]) for i in ids}}
    conc, pgd = {}, {}
    for r in e2["models"][tag]["rows"]:
        if r["tm"] == TM and abs(r["eps"] - EPS) < 1e-9:
            conc[r["attack_id"]] = conc.get(r["attack_id"], False) or bool(r["concealed"])
            pgd[r["attack_id"]] = bool(r["concealed"])
    for r in e2b["models"][tag]["rows"]:
        if r["tm"] == TM and abs(r["eps"] - EPS) < 1e-9 and r["mode"] == "white-box":
            conc[r["attack_id"]] = conc.get(r["attack_id"], False) or bool(r["concealed"])
    detected = [i for i in ids if det[i]]
    if sorted(conc) != sorted(detected):
        raise RuntimeError(f"{tag}: concealment rows do not cover exactly the detected attacks")
    out["ir2_union"] = {i: int(conc[i]) for i in detected}
    out["ie2_process"] = {l["attack_id"]: int(l["process_hit"]) for l in e1["localisation"]}
    if sorted(out["ie2_process"]) != sorted(detected):
        raise RuntimeError(f"{tag}: localisation records do not cover exactly the detected attacks")
    return out, {"pgd_literal_concealed": int(sum(pgd.values())), "n": len(pgd)}


METRICS = {"event_recall": (">=", 0.50), "ir2_union": ("<=", 0.20), "ie2_process": (">=", 0.60)}


# ---------------------------------------------------------------- vectorised bounds (Y: [n, N] binary)
def cp_vec(k, n, direction):
    k = np.asarray(k, dtype=float)
    n = np.broadcast_to(np.asarray(n, dtype=float), k.shape)
    if direction == ">=":
        with np.errstate(all="ignore"):
            b = st.beta.ppf(ALPHA, np.maximum(k, 1e-300), n - k + 1)
        return np.where(k <= 0, 0.0, b)
    with np.errstate(all="ignore"):
        b = st.beta.ppf(1 - ALPHA, k + 1, np.maximum(n - k, 1e-300))
    return np.where(k >= n, 1.0, b)


def effn_cp(p, n, deff, direction):
    """Effective-sample-size CP bound; deff < 1 is truncated to 1, an undefined deff (p in {0,1}) to 1."""
    d = np.where(np.isfinite(deff), np.maximum(deff, 1.0), 1.0)
    neff = n / d
    return cp_vec(p * neff, neff, direction), neff


def membership(units, prims):
    plist = sorted(set(p for u in units for p in prims[u]))
    col = {p: j for j, p in enumerate(plist)}
    M = np.zeros((len(units), len(plist)))
    for i, u in enumerate(units):
        for p in prims[u]:
            M[i, col[p]] = 1.0
    return M, plist


def dependency_matrix(units, prims):
    M, _ = membership(units, prims)
    return ((M @ M.T) > 0).astype(float)          # includes the diagonal


def sandwich_deff(Y, D):
    n = Y.shape[0]
    p = Y.mean(0)
    Rr = Y - p
    V = (Rr * (D @ Rr)).sum(0) / n ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        deff = V / (p * (1 - p) / n)
    deff[(p <= 0) | (p >= 1)] = np.nan
    return deff


def product_weights(units, prims, rng, B):
    M, plist = membership(units, prims)
    c = rng.poisson(1.0, size=(B, M.shape[1])).astype(float)
    # attack weight = product over its primitives; computed as exp(M log c) with log 0 handled by a zero mask
    zero = (c == 0).astype(float) @ M.T > 0
    with np.errstate(divide="ignore"):
        lc = np.where(c > 0, np.log(np.where(c > 0, c, 1.0)), 0.0)
    W = np.exp(lc @ M.T)
    W[zero] = 0.0
    return W


def cluster_weights(units, cluster_of, rng, B):
    cl = sorted(set(cluster_of[u] for u in units))
    col = {g: j for j, g in enumerate(cl)}
    m = rng.multinomial(len(cl), np.full(len(cl), 1.0 / len(cl)), size=B).astype(float)
    return m[:, [col[cluster_of[u]] for u in units]]


def boot_theta(W, Y):
    den = W.sum(1)
    ok = den > 0
    return (W[ok] @ Y) / den[ok][:, None], int((~ok).sum())


def boot_bounds(theta, p, n, direction):
    q = np.quantile(theta, ALPHA if direction == ">=" else 1 - ALPHA, axis=0)
    var = theta.var(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        deff = var / (p * (1 - p) / n)
    deff = np.where((p <= 0) | (p >= 1), np.nan, deff)
    eb, neff = effn_cp(p, n, deff, direction)
    return q, deff, eb, neff, np.sqrt(var)


def all_methods(Y, units, direction, rng, B, prims=prim):
    """Every bound for each column of Y. Returns dict name -> array [N] (plus diagnostics)."""
    n = Y.shape[0]
    p = Y.mean(0)
    k = Y.sum(0)
    out = {"cp_exact": cp_vec(k, n, direction)}
    D = dependency_matrix(units, prims)
    dd = sandwich_deff(Y, D)
    out["dependency_graph_effn_cp"], out["_neff_dependency_graph"] = effn_cp(p, n, dd, direction)
    out["_deff_dependency_graph"] = dd
    schemes = {"primitive_product": product_weights(units, prims, rng, B),
               "first_primitive_cluster": cluster_weights(units, {u: prims[u][0] for u in units}, rng, B),
               "iid_attack": cluster_weights(units, {u: u for u in units}, rng, B)}
    for name, W in schemes.items():
        th, dropped = boot_theta(W, Y)
        q, deff, eb, neff, se = boot_bounds(th, p, n, direction)
        out[f"{name}_percentile"] = q
        out[f"{name}_effn_cp"] = eb
        out[f"_deff_{name}"] = deff
        out[f"_neff_{name}"] = neff
        out[f"_se_{name}"] = se
        out[f"_dropped_{name}"] = dropped
    return out


def accepted(b, direction, tau):
    return bool(b >= tau) if direction == ">=" else bool(b <= tau)


def exact_cluster_distribution(units, cluster_of, y):
    """All G^G equally likely ordered cluster resamples (feasible for the 4 test files)."""
    cl = sorted(set(cluster_of[u] for u in units))
    G = len(cl)
    k = np.array([sum(y[i] for i, u in enumerate(units) if cluster_of[u] == g) for g in cl], dtype=float)
    n = np.array([sum(1 for u in units if cluster_of[u] == g) for g in cl], dtype=float)
    vals = [k[list(d)].sum() / n[list(d)].sum() for d in itertools.product(range(G), repeat=G)]
    return np.array(vals), G


def kish_clusters(groups):
    sizes = np.array([len(g) for g in groups], dtype=float)
    return float(sizes.sum() ** 2 / (sizes ** 2).sum())


def f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


# ---------------------------------------------------------------- scheme diagnostics (all 58 attacks)
comp_all = R.components(ids, prim)
diag = {"n_attacks": len(ids), "n_primitives": len(set(p for u in ids for p in prim[u])),
        "n_primitives_literal_AP3": len(set(p for u in ids for p in prim_literal[u])),
        "n_two_primitive_attacks": int(sum(len(prim[u]) > 1 for u in ids)),
        "connected_components": {"n": len(comp_all), "sizes": [len(g) for g in comp_all],
                                 "largest_share": len(comp_all[0]) / len(ids),
                                 "kish_effective_clusters": kish_clusters(comp_all), "members": comp_all},
        "first_primitive_clusters": len(set(prim[u][0] for u in ids)),
        "test_files": len(set(tfile.values())),
        "primitive_reuse": {p: [u for u in ids if p in prim[u]] for p in sorted(set(p for u in ids for p in prim[u]))
                            if sum(p in prim[u] for u in ids) > 1}}

# ---------------------------------------------------------------- observed data
results = {}
for tag in VARIANTS:
    oc, pgd_info = outcomes(tag)
    results[tag] = {"pgd_literal_TM-A_eps0.5": pgd_info}
    for metric, (direction, tau) in METRICS.items():
        units = sorted(oc[metric])
        y = np.array([oc[metric][u] for u in units], dtype=float)
        k, n = int(y.sum()), len(units)
        p_hat = k / n
        comps = R.components(units, prim)
        Y = y[:, None]
        m = all_methods(Y, units, direction, new_rng(), args.B)
        m_lit = all_methods(Y, units, direction, new_rng(), args.B, prims=prim_literal)
        blk = {"k": k, "n": n, "estimate": p_hat, "direction": direction, "tau": tau,
               "units_with_outcome_1": [u for u, v in zip(units, y) if v],
               "clusters": {"primitives": len(set(p for u in units for p in prim[u])),
                            "first_primitive": len(set(prim[u][0] for u in units)),
                            "connected_components": len(comps), "component_sizes": [len(g) for g in comps],
                            "kish_effective_components": kish_clusters(comps)},
               "bounds": {}, "design_effects": {}, "effective_n": {}}
        for name in ("cp_exact", "dependency_graph_effn_cp", "primitive_product_percentile", "primitive_product_effn_cp",
                     "first_primitive_cluster_percentile", "first_primitive_cluster_effn_cp",
                     "iid_attack_percentile", "iid_attack_effn_cp"):
            b = float(m[name][0])
            blk["bounds"][name] = {"bound": b, "accepted": accepted(b, direction, tau)}
        for name in ("dependency_graph", "primitive_product", "first_primitive_cluster", "iid_attack"):
            blk["design_effects"][name] = f(float(m[f"_deff_{name}"][0]))
            blk["effective_n"][name] = f(float(m[f"_neff_{name}"][0]))
        blk["bootstrap_se"] = {name: f(float(m[f"_se_{name}"][0])) for name in ("primitive_product", "first_primitive_cluster", "iid_attack")}
        blk["bootstrap_replicates"] = args.B
        blk["bootstrap_replicates_dropped_zero_weight"] = {"primitive_product": m["_dropped_primitive_product"]}
        blk["literal_AP3_sensitivity"] = {name: float(m_lit[name][0]) for name in
                                          ("dependency_graph_effn_cp", "primitive_product_percentile", "primitive_product_effn_cp",
                                           "first_primitive_cluster_percentile", "first_primitive_cluster_effn_cp")}
        # diagnostic: component bootstrap; completeness: exact test-file bootstrap
        cmap = {u: gi for gi, g in enumerate(comps) for u in g}
        th, _ = boot_theta(cluster_weights(units, cmap, new_rng(), args.B), Y)
        qc = float(np.quantile(th, ALPHA if direction == ">=" else 1 - ALPHA))
        blk["bounds"]["component_bootstrap_percentile_diagnostic"] = {"bound": qc, "accepted": accepted(qc, direction, tau),
                                                                      "clusters": len(comps)}
        th, G = exact_cluster_distribution(units, tfile, y)
        qf = float(np.quantile(th, ALPHA if direction == ">=" else 1 - ALPHA))
        blk["bounds"]["test_file_bootstrap_exact_percentile"] = {"bound": qf, "accepted": accepted(qf, direction, tau),
                                                                 "clusters": G, "note": "4 clusters: not a credible basis"}
        if p_hat in (0.0, 1.0):
            g = len(comps)
            wc = float(cp_vec(g if p_hat == 1.0 else 0, g, direction))
            blk["bounds"]["worst_case_component_cp"] = {"units": g, "bound": wc, "accepted": accepted(wc, direction, tau),
                                                        "note": "perfect dependence within components; components independent"}
            blk["note_degenerate"] = ("all outcomes equal: every bootstrap distribution is a point mass at the estimate and "
                                      "the design effect is undefined, so the percentile bounds equal the estimate and the "
                                      "effective-n bounds fall back to n_eff = n (the exact CP bound); only the worst-case "
                                      "component bound reflects the dependence")
        results[tag][metric] = blk
        print(tag, metric, f"{k}/{n}", {kk: round(v["bound"], 4) for kk, v in blk["bounds"].items()},
              {kk: (None if v is None else round(v, 2)) for kk, v in blk["design_effects"].items()}, flush=True)


# ---------------------------------------------------------------- calibration by simulation
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(80)
gh_w = gh_w / gh_w.sum()


def design_mean(b0, sigma, m_counts):
    """Design-averaged P(y_i = 1) under the logistic primitive random-effects model (Gauss-Hermite)."""
    return float(np.mean([np.sum(gh_w * special.expit(b0 + math.sqrt(mc) * sigma * gh_x)) for mc in m_counts]))


def simulate_setting(name, units, mu, direction, nsim, bsim, observed_deff=(), group=200):
    M, plist = membership(units, prim)
    m_counts = M.sum(1)
    rows = []
    for sigma in SIGMAS:
        b0 = optimize.brentq(lambda b: design_mean(b, sigma, m_counts) - mu, -20, 20)
        rng = new_rng()
        cover = {}
        means = {}
        deffs = []
        degenerate = 0
        for g0 in range(0, nsim, group):
            N = min(group, nsim - g0)
            A = rng.normal(0.0, sigma, size=(N, M.shape[1])) if sigma > 0 else np.zeros((N, M.shape[1]))
            P = special.expit(b0 + A @ M.T)                       # [N, n]
            Y = (rng.random(P.shape) < P).astype(float).T        # [n, N]
            degenerate += int(((Y.mean(0) <= 0) | (Y.mean(0) >= 1)).sum())
            res = all_methods(Y, units, direction, rng, bsim)
            deffs.append(res["_deff_dependency_graph"])
            for kname, v in res.items():
                if kname.startswith("_"):
                    continue
                ok = (v <= mu) if direction == ">=" else (v >= mu)
                cover.setdefault(kname, []).append(ok)
                means.setdefault(kname, []).append(v)
        dd = np.concatenate(deffs)
        rows.append({"sigma": sigma, "b0": b0, "true_mean": mu,
                     "coverage": {kname: float(np.concatenate(v).mean()) for kname, v in cover.items()},
                     "mean_bound": {kname: float(np.concatenate(v).mean()) for kname, v in means.items()},
                     "mean_dependency_graph_deff": float(np.nanmean(dd)), "median_dependency_graph_deff": float(np.nanmedian(dd)),
                     "dependency_graph_deff_quantiles": {q: float(np.nanquantile(dd, float(q))) for q in ("0.05", "0.5", "0.95")},
                     "share_deff_ge_observed": {f"{od:.3f}": float(np.nanmean(dd >= od)) for od in observed_deff},
                     "degenerate_datasets": degenerate})
        print("sim", name, "sigma", sigma, "deff mean", round(float(np.nanmean(dd)), 2), "q95",
              round(rows[-1]["dependency_graph_deff_quantiles"]["0.95"], 2), "P(deff>=obs)", rows[-1]["share_deff_ge_observed"],
              {kk: round(v, 3) for kk, v in rows[-1]["coverage"].items()}, flush=True)
    return {"units": len(units), "true_mean": mu, "direction": direction, "nsim": nsim, "bsim": bsim,
            "mc_se_at_0.95": math.sqrt(0.95 * 0.05 / nsim), "by_sigma": rows}


oc_b, _ = outcomes("baseline_s0")
det_b = sorted(oc_b["ir2_union"])
calib = {
    "model": "y_i ~ Bernoulli(expit(b0 + sum_{p in P_i} a_p)), a_p ~ N(0, sigma^2) iid over primitives; "
             "b0 such that the design-averaged success probability equals true_mean; coverage = P(bound on the "
             "correct side of true_mean)",
    "settings": {
        "recall_like_58_attacks": simulate_setting(
            "recall", ids, 38 / 58, ">=", args.nsim, args.bsim,
            observed_deff=[results[t]["event_recall"]["design_effects"]["dependency_graph"] for t in VARIANTS]),
        "concealment_like_38_detected": simulate_setting(
            "conceal", det_b, 9 / 38, "<=", args.nsim, args.bsim,
            observed_deff=[results["baseline_s0"]["ir2_union"]["design_effects"]["dependency_graph"]]),
        "localisation_like_38_detected": simulate_setting("localise", det_b, 37 / 38, ">=", args.nsim, args.bsim)}}

KEY = ("cp_exact", "dependency_graph_effn_cp", "primitive_product_percentile", "primitive_product_effn_cp",
       "first_primitive_cluster_percentile", "first_primitive_cluster_effn_cp", "iid_attack_percentile")
cov_ranges = {}
for sname in ("recall_like_58_attacks", "concealment_like_38_detected", "localisation_like_38_detected"):
    rows = calib["settings"][sname]["by_sigma"]
    cov_ranges[sname] = {kname: [min(r["coverage"][kname] for r in rows), max(r["coverage"][kname] for r in rows)] for kname in KEY}
summary = {}
for tag in VARIANTS:
    summary[tag] = {}
    for metric in METRICS:
        b = results[tag][metric]
        s = {"k_n": [b["k"], b["n"]], "estimate": b["estimate"]}
        for kname in KEY + ("worst_case_component_cp",):
            if kname in b["bounds"]:
                s[kname] = [b["bounds"][kname]["bound"], "accept" if b["bounds"][kname]["accepted"] else "not accepted"]
        s["decision_differs_from_cp"] = [kname for kname in KEY + ("worst_case_component_cp",) if kname in b["bounds"]
                                         and b["bounds"][kname]["accepted"] != b["bounds"]["cp_exact"]["accepted"]]
        s["dependency_graph_deff"] = b["design_effects"]["dependency_graph"]
        summary[tag][metric] = s
rc, cc_ = cov_ranges["recall_like_58_attacks"], cov_ranges["concealment_like_38_detected"]
recommendation = (
    "Across sigma in [0, 6] the one-sided coverage (nominal 0.95) in the recall-like / concealment-like settings was: "
    f"exact CP {rc['cp_exact'][0]:.3f}-{rc['cp_exact'][1]:.3f} / {cc_['cp_exact'][0]:.3f}-{cc_['cp_exact'][1]:.3f}; "
    f"dependency-graph effective-n CP {rc['dependency_graph_effn_cp'][0]:.3f}-{rc['dependency_graph_effn_cp'][1]:.3f} / "
    f"{cc_['dependency_graph_effn_cp'][0]:.3f}-{cc_['dependency_graph_effn_cp'][1]:.3f}; "
    f"primitive product-weight percentile {rc['primitive_product_percentile'][0]:.3f}-{rc['primitive_product_percentile'][1]:.3f} / "
    f"{cc_['primitive_product_percentile'][0]:.3f}-{cc_['primitive_product_percentile'][1]:.3f}; "
    f"first-primitive cluster percentile {rc['first_primitive_cluster_percentile'][0]:.3f}-{rc['first_primitive_cluster_percentile'][1]:.3f} / "
    f"{cc_['first_primitive_cluster_percentile'][0]:.3f}-{cc_['first_primitive_cluster_percentile'][1]:.3f}; "
    f"iid bootstrap {rc['iid_attack_percentile'][0]:.3f}-{rc['iid_attack_percentile'][1]:.3f} / "
    f"{cc_['iid_attack_percentile'][0]:.3f}-{cc_['iid_attack_percentile'][1]:.3f}. "
    "Recommended cluster-robust bound: the dependency-graph effective-n CP bound (close to nominal throughout); "
    "bootstrap scheme of record: the primitive product-weight bootstrap (percentile), which is valid but conservative "
    "under weak dependence. The first-primitive cluster bootstrap under-covers and is not recommended. Percentile "
    "bounds must not be used for proportions near 1 (localisation-like setting: coverage "
    f"{cov_ranges['localisation_like_38_detected']['primitive_product_percentile'][0]:.2f}-"
    f"{cov_ranges['localisation_like_38_detected']['primitive_product_percentile'][1]:.2f}).")

cc = diag["connected_components"]
decision = (f"Connected components of the attack-primitive graph give {cc['n']} clusters, one of which holds "
            f"{cc['sizes'][0]} of the {len(ids)} attacks (Kish effective number of clusters "
            f"{cc['kish_effective_clusters']:.2f}); a component bootstrap would essentially resample a single unit "
            "and is reported only as a diagnostic. Test files give only 4 clusters (256 possible resamples) and are "
            "reported for completeness. The dependence is therefore handled at the primitive level, with three "
            "schemes that differ in how they treat the 25 two-primitive attacks: the product-weight bootstrap "
            "(conservative by construction: a two-primitive attack's weight has variance 3 instead of 1), the "
            "first-primitive cluster bootstrap (ignores the dependence through the second primitive) and the "
            "dependency-graph cluster-robust variance with an effective-sample-size CP bound. The calibration "
            "simulation reports their one-sided coverage under a primitive random-effects model.")

out = {"script": SCRIPT, "started_utc": started, "finished_utc": R.utc_now(), "runtime_s": time.time() - t0,
       "versions": R.versions(),
       "settings": {"alpha_one_sided": ALPHA, "B": args.B, "seed": args.seed, "nsim": args.nsim, "bsim": args.bsim,
                    "sigmas": SIGMAS, "threat_model": TM, "eps": EPS,
                    "percentile_bound": "one-sided percentile bound of the bootstrap distribution (numpy linear quantile)",
                    "primitive_reading": "AP3 (A415) read as AP33; literal reading as sensitivity"},
       "summary": summary, "recommendation": recommendation, "calibration_coverage_ranges": cov_ranges,
       "scheme_diagnostics": diag, "scheme_decision": decision, "inconsistent_rows_flagged": R.INCONSISTENT_ROWS,
       "results": results, "calibration": calib}
R.jdump(out, "cluster_bootstrap.json", SCRIPT)
print("runtime_s", round(time.time() - t0, 1))
