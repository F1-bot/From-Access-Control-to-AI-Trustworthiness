"""IR-2 aggregation table (reviewer round).

Re-reads the EXISTING results of E2 (PGD, results/e2_evasion.json) and E2b (greedy prediction-replay,
white-box and transfer; results/e2b_greedy.json) and, for both variants and each threat model,
reports the attack success rate (events concealed / detected events attacked) under several
aggregation rules, with exact one-sided 95% Clopper-Pearson bounds and a three-way decision against
tau = 0.20 (IR-2). No model is run; nothing existing is modified.

Rules (per variant, per threat model TM-ALL/TM-A/TM-B):
  (a) literal pre-registered procedure: PGD at eps = 0.5;
  (b) PGD aggregated over the nested budgets in the declared ball eps in {0.1, 0.25, 0.5}
      (concealed if ANY admissible run concealed the event);
  (c) greedy (white-box) alone, at eps = 0.5 and nested over {0.1, 0.25, 0.5};
  (d) strongest attack: union of PGD and greedy (white-box), nested over {0.1, 0.25, 0.5}.
Also: PGD success by eps for each (variant, TM) to document the reported non-monotonicity, and the
transfer-greedy success rates for reference (these belong to IR-1, not the IR-2 white-box decision).

Usage: PYTHONIOENCODING=utf-8 python ir2_aggregation.py
"""
import json
import os
import platform
import sys
import time

import numpy as np
import scipy

import adv_lib as A
import hai_lib as H

t0 = time.time()
R = H.RES
NESTED = [0.1, 0.25, 0.5]        # budgets inside the declared L_inf ball (eps <= 0.5 SD)
DECLARED = 0.5
TAU = 0.20
ALPHA = 0.05
TMS = ["TM-ALL", "TM-A", "TM-B"]
VARIANTS = ["baseline_s0", "hardened_s0"]

e2 = json.load(open(os.path.join(R, "e2_evasion.json")))
e2b = json.load(open(os.path.join(R, "e2b_greedy.json")))
ALL_EPS = sorted(set(e2["eps"]) | set(e2b["eps"]))

near = lambda a, b: abs(float(a) - float(b)) < 1e-9
in_nested = lambda ep: any(near(ep, v) for v in NESTED)


def decide(d):
    k, n = A.kn(d)
    return A.three_way(k, n, TAU, "<=", ALPHA)


out = {"description": "IR-2 adaptive-evasion aggregation; tau=0.20, one-sided 95% Clopper-Pearson, G-4.",
       "tau": TAU, "alpha": ALPHA, "declared_ball_eps": DECLARED, "nested_eps": NESTED,
       "sources": {"pgd": "e2_evasion.json", "greedy": "e2b_greedy.json"}, "variants": {}}

for tag in VARIANTS:
    pgd_rows = e2["models"][tag]["rows"]
    grd_rows = e2b["models"][tag]["rows"]
    theta = e2["models"][tag]["theta"]
    vout = {"theta": theta, "threat_models": {}, "pgd_success_by_eps": {}, "transfer_greedy": {}}

    for tm in TMS:
        # (a) literal pre-registered: PGD at eps = 0.5
        a_pgd05 = A.aggregate_concealed(pgd_rows, lambda r, tm=tm: r["tm"] == tm and near(r["eps"], DECLARED))
        # (b) PGD nested over the declared ball
        b_pgd_nested = A.aggregate_concealed(pgd_rows, lambda r, tm=tm: r["tm"] == tm and in_nested(r["eps"]))
        # (c) greedy white-box at eps=0.5 and nested
        c_grd05 = A.aggregate_concealed(
            grd_rows, lambda r, tm=tm: r["tm"] == tm and r["mode"] == "white-box" and near(r["eps"], DECLARED))
        c_grd_nested = A.aggregate_concealed(
            grd_rows, lambda r, tm=tm: r["tm"] == tm and r["mode"] == "white-box" and in_nested(r["eps"]))
        # (d) union of PGD and greedy (white-box), nested
        union = dict(b_pgd_nested)
        for aid, v in c_grd_nested.items():
            union[aid] = union.get(aid, False) or v

        if A.kn(a_pgd05)[1] == 0:
            continue  # threat model empty for this variant (no channels), skip
        vout["threat_models"][tm] = {
            "a_pgd_eps0.5": decide(a_pgd05),
            "b_pgd_nested": decide(b_pgd_nested),
            "c_greedy_eps0.5": decide(c_grd05),
            "c_greedy_nested": decide(c_grd_nested),
            "d_union_nested": decide(union),
        }

        # PGD success by eps (document non-monotonicity)
        by_eps = {}
        for ep in ALL_EPS:
            d = A.aggregate_concealed(pgd_rows, lambda r, tm=tm, ep=ep: r["tm"] == tm and near(r["eps"], ep))
            k, n = A.kn(d)
            if n:
                by_eps[str(ep)] = {"k": k, "n": n}
        vout["pgd_success_by_eps"][tm] = by_eps

        # transfer greedy (IR-1 reference), eps=0.5 and nested
        tr05 = A.aggregate_concealed(
            grd_rows, lambda r, tm=tm: r["tm"] == tm and r["mode"] == "transfer" and near(r["eps"], DECLARED))
        trn = A.aggregate_concealed(
            grd_rows, lambda r, tm=tm: r["tm"] == tm and r["mode"] == "transfer" and in_nested(r["eps"]))
        if A.kn(tr05)[1]:
            k05, n05 = A.kn(tr05)
            kn_, nn_ = A.kn(trn)
            vout["transfer_greedy"][tm] = {"eps0.5": {"k": k05, "n": n05}, "nested": {"k": kn_, "n": nn_}}

    out["variants"][tag] = vout

out["runtime_s"] = time.time() - t0
out["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                   "platform": platform.platform()}
with open(os.path.join(R, "ir2_aggregation.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)

# console summary
for tag in VARIANTS:
    print("\n==", tag, "==")
    v = out["variants"][tag]
    for tm, cell in v["threat_models"].items():
        print(f" {tm}")
        for rule in ("a_pgd_eps0.5", "b_pgd_nested", "c_greedy_eps0.5", "c_greedy_nested", "d_union_nested"):
            c = cell[rule]
            print(f"   {rule:17s} {c['k']:2d}/{c['n']:<2d} est={c['estimate']:.3f} "
                  f"LCB={c['lcb95']:.3f} UCB={c['ucb95']:.3f} -> {c['decision']}")
        print("   PGD by eps:", {k: f"{d['k']}/{d['n']}" for k, d in v["pgd_success_by_eps"][tm].items()})
        if tm in v["transfer_greedy"]:
            tg = v["transfer_greedy"][tm]
            print(f"   transfer greedy eps0.5 {tg['eps0.5']['k']}/{tg['eps0.5']['n']}  nested {tg['nested']['k']}/{tg['nested']['n']}")
print("\nruntime_s", round(out["runtime_s"], 2))
