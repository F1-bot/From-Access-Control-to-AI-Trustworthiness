"""Probabilistic conformity semantics: the numbers of Section 4.7 and Figure 7 of the article.

(a) Operating characteristic (OC) of the requirement 'event recall >= tau' (tau = 0.5, n = 58 attack events) just below
    p = tau, under the naive point-estimate rule and the LCB rule for alpha = 0.05.
(b) Number of evaluation units needed so that a system exceeding tau by delta passes with probability >= 0.8: the first
    exact binomial n, the stable exact n (power >= 0.8 for every larger sample) and the distribution-free Hoeffding
    requirement, which controls both the consumer's risk alpha and the producer's risk beta = 0.2.
(c) Empirical acceptance frequencies of 15 detector variants (5 seeds x 3 pre-computed operating points of E1 on
    HAI 22.04) under half-sample replication (29 of the 58 events, without replacement), and the exact hypergeometric
    acceptance probabilities of half-samples of a fixed set of 58 events.

Reads code/experiments/results/e1_baseline_s*.json and writes data/results/stats_semantics.json.
Usage: python run_stats_semantics.py
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aitcert import stats as ST  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "data", "results", "stats_semantics.json")
TAU, N_FULL, BETA = 0.5, 58, 0.2
ALPHAS = (0.10, 0.05, 0.01)
DELTAS = np.array([0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3])
NH, R, SEED = 29, 4000, 2026          # half-sample size, replications, seed of the half-sample generator


def compute():
    """All quantities of the semantics; 'out' is the content of stats_semantics.json."""
    tau = TAU
    n_first = {a: [ST.n_exact_power(tau, d, a, BETA) for d in DELTAS] for a in ALPHAS}         # first crossing
    n_exact = {a: [ST.n_exact_power_stable(tau, d, a, BETA) for d in DELTAS] for a in ALPHAS}  # power >= 0.8 for all larger n
    n_hoef = [ST.n_hoeffding(0.05, d, beta=BETA) for d in DELTAS]   # distribution-free, alpha = 0.05 and beta = 0.2
    # (c) empirical half-sample acceptance of the real detector variants
    vecs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "code", "experiments", "results", "e1_baseline_s*.json"))):
        d = json.load(open(f))
        for q, v in d["by_quantile"].items():
            vecs.append((d["tag"], q, np.array(v["detected_vector"])))
    rng = np.random.default_rng(SEED)
    emp = []
    for tag, q, v in vecs:
        acc_n = acc_l = 0
        for _ in range(R):
            idx = rng.choice(len(v), NH, replace=False); k = int(v[idx].sum())
            acc_n += (k / NH >= tau); acc_l += (ST.cp_lower(k, NH, 0.05) >= tau)
        emp.append({"variant": f"{tag}@q{q}", "recall_full": float(v.mean()), "acc_naive": acc_n / R, "acc_lcb": acc_l / R})
    # exact acceptance probability of a half-sample (29 of a fixed set of 58 events, without replacement) as a function
    # of the number K of detected events: hypergeometric, not binomial
    k_naive, k_lcb = int(np.ceil(tau * NH - 1e-12)), ST.min_successes(NH, tau, 0.05)
    K = np.arange(0, N_FULL + 1)
    hyp_naive = np.array([hypergeom(N_FULL, k, NH).sf(k_naive - 1) for k in K])
    hyp_lcb = np.array([hypergeom(N_FULL, k, NH).sf(k_lcb - 1) for k in K])
    oc_tau_naive = float(ST.oc_curve([tau - 1e-9], N_FULL, tau, None)[0])
    oc_tau_lcb = float(ST.oc_curve([tau - 1e-9], N_FULL, tau, 0.05)[0])
    out = {"tau": tau, "n_full": N_FULL, "beta": BETA, "oc_at_tau_minus": {"naive": oc_tau_naive, "lcb_0.05": oc_tau_lcb},
           "n_exact_stable": {str(a): {str(d): int(n) for d, n in zip(DELTAS, n_exact[a])} for a in ALPHAS},
           "n_exact_first_crossing": {str(a): {str(d): int(n) for d, n in zip(DELTAS, n_first[a])} for a in ALPHAS},
           "n_hoeffding_power_0.05": {str(d): int(n) for d, n in zip(DELTAS, n_hoef)},
           "n_hoeffding_consumer_only_0.05": {str(d): ST.n_hoeffding(0.05, d) for d in DELTAS},
           "half_sample": {"n": NH, "k_naive": k_naive, "k_lcb": int(k_lcb),
                           "hypergeometric": {str(int(k)): {"naive": float(a_), "lcb": float(b_)}
                                              for k, a_, b_ in zip(K, hyp_naive, hyp_lcb)}},
           "empirical": emp}
    return {"out": out, "n_first": n_first, "n_exact": n_exact, "n_hoef": n_hoef, "emp": emp, "K": K,
            "hyp_naive": hyp_naive, "hyp_lcb": hyp_lcb, "oc_tau_naive": oc_tau_naive, "oc_tau_lcb": oc_tau_lcb}


def write(c, path=OUT):
    json.dump(c["out"], open(path, "w"), indent=1)


if __name__ == "__main__":
    c = compute()
    write(c)
    dev = max(max(abs(e["acc_naive"] - c["hyp_naive"][round(e["recall_full"] * N_FULL)]),
                  abs(e["acc_lcb"] - c["hyp_lcb"][round(e["recall_full"] * N_FULL)])) for e in c["emp"])
    print("wrote", os.path.relpath(OUT, ROOT), "| OC at tau-:", c["out"]["oc_at_tau_minus"], "| stable exact n, alpha 0.05:",
          c["out"]["n_exact_stable"]["0.05"], "| max |empirical - hypergeometric|:", round(dev, 4))
