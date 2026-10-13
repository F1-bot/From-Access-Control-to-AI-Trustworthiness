"""Sensitivity analyses of the ATLAS threat-coverage analysis and of the framework crosswalk
(major revision, reviewer requests). Outputs:
    data/results/atlas_sensitivity.json
    data/results/crosswalk_sensitivity.json
All specification files are read at run time (spec/atlas_mitigation_map.yaml, spec/crosswalk_data.py,
spec/*_services.yaml, spec/*_guarantees.yaml), as are the ATLAS release named in the mitigation map,
the framework inventories in data/processed and data/results/reference_profiles.json (whose closures
are recomputed with the current catalogue). Re-run after any revision of the specification:
    cd code/aitcert && python run_sensitivity.py

Element sets (targets):
    X_leg  = legacy services + legacy assurance rows + X:PROFILE
    X_ai   = class-I services + AI assurance rows + X:AIPROFILE           ('class I plus AI rows alone')
    X_ext  = X_leg + X_ai
The ATLAS map references no profile mechanism, so adding X:PROFILE / X:AIPROFILE leaves every ATLAS
number unchanged (checked by the reproduction test against atlas_coverage.json).

Analyses:
 T1  counts of the ATLAS mitigations of each category that are fully / at least partially required
     under X_leg, X_ai and X_ext;
 T2  less lenient technique aggregators: mean_{m in M(t)} max_x w(m,x); every mitigation at least
     partially required; every mitigation fully required (149 mitigated techniques, 73 case studies);
 T3a every F mapping to a class-I service or an AI row downgraded to P (variant: also X:AIPROFILE);
 T3b 1000 replicates (seed 0); in each, 10% of the grades of the ATLAS map and 10% of the grades of the
     crosswalk are reassigned (F -> P or none, P -> F or none, uniformly); percentiles 2.5/50/97.5.
     Secondary: the same plus an equal number of random additions (omission errors);
 T4  leave-one-element-out marginal contributions for every element of X_ext, group removals,
     contributions over X_leg, and exact Shapley values (airport-game formula) as an additive attribution;
 T5  profile-level coverage for the five reference profiles: services of the closed profile, assurance
     rows whose first '+' mark is at a guarantee level <= the profile's G, and the profile mechanisms.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import OrderedDict

import numpy as np

import sens_lib as L

HEADLINE_FW = L.FW_ORDER                       # the five frameworks
EXTRA_FW = [L.CSF_HIGH]


# ============================================================================ helpers
def r6(x):
    return None if x is None else round(float(x), 6)


def masks_for(E, targets):
    return OrderedDict([("legacy", L.xmask(targets, E.X_leg, cat=E.cat)),
                        ("ai_only", L.xmask(targets, E.X_ai, cat=E.cat)),
                        ("extended", L.xmask(targets, E.X_ext, cat=E.cat))])


def columns_of(E, targets, x):
    """Columns that belong to element x: x itself plus level-specific targets 'x-n' of a service x."""
    return [j for j, t in enumerate(targets) if t == x or (L.SVC_LEVEL_RE.match(t) and t[:2] == x)]


def element_player_matrix(E, targets, V):
    """Fold the target columns of V into one column per element of X_ext (players of the coverage game)."""
    return np.stack([V[:, columns_of(E, targets, x)].max(axis=1) for x in E.X_ext], axis=1)


def downgrade(codes, targets, which):
    cols = [j for j, t in enumerate(targets) if t in which or (L.SVC_LEVEL_RE.match(t) and t[:2] in which)]
    out = codes.copy()
    sub = out[:, cols]
    sub[sub == 2] = 1
    out[:, cols] = sub
    return out, int((codes[:, cols] == 2).sum())


# ============================================================================ ATLAS pieces
def category_counts(AM, masks, codes=None):
    out = OrderedDict()
    cats = list(AM.categories.items()) + [("All mitigations", AM.mits)]
    for c, ms in cats:
        j = np.array([AM.mix[m] for m in ms])
        r = OrderedDict(n=len(ms), out_of_scope_unmapped=[m for m in ms if not AM.in_scope[AM.mix[m]]])
        best = {}
        for lab, mask in masks.items():
            mb = AM.mitigation_best(mask, codes)[j]
            best[lab] = mb
            r[lab] = OrderedDict(fully_required=int((mb == 2).sum()), at_least_partially_required=int((mb >= 1).sum()))
        ids = np.array(ms)
        r["extended_minus_ai_only"] = OrderedDict(
            fully_required_only_via_legacy=ids[(best["extended"] == 2) & (best["ai_only"] < 2)].tolist(),
            touched_only_via_legacy=ids[(best["extended"] >= 1) & (best["ai_only"] == 0)].tolist())
        r["not_fully_required_extended"] = ids[best["extended"] < 2].tolist()
        r["untouched_extended"] = ids[best["extended"] == 0].tolist()
        out[c] = r
    return out


def aggregator_block(AM, masks, codes=None, scope=None):
    """T2 aggregators for X_leg and X_ext."""
    res = OrderedDict()
    pts = {}
    for lab in ("legacy", "extended"):
        pt = AM.per_technique(masks[lab], codes, 0.5, scope)
        pts[lab] = pt
        mt = AM.mitigated
        r = OrderedDict()
        for subset, sel in (("mitigated_techniques", mt), ("all_techniques", np.ones_like(mt))):
            r[subset] = OrderedDict(
                n=int(sel.sum()),
                max_weighted=r6(pt["max_w"][sel].mean()), max_addressed=r6(pt["addressed"][sel].mean()),
                max_full=r6(pt["full"][sel].mean()),
                mean_over_mitigations=r6(pt["mean_w"][sel].mean()),
                all_mitigations_at_least_partial=r6(pt["all_partial"][sel].mean()),
                all_mitigations_full=r6(pt["all_full"][sel].mean()),
                count_all_mitigations_at_least_partial=int(pt["all_partial"][sel].sum()),
                count_all_mitigations_full=int(pt["all_full"][sel].sum()))
        cv = {k: AM.case_values(pt[k]) for k in ("max_w", "addressed", "full", "mean_w", "all_partial", "all_full")}
        r["case_studies"] = OrderedDict(
            n=len(AM.cases),
            max_weighted=r6(cv["max_w"].mean()), max_addressed=r6(cv["addressed"].mean()), max_full=r6(cv["full"].mean()),
            mean_over_mitigations=r6(cv["mean_w"].mean()),
            all_mitigations_at_least_partial=r6(cv["all_partial"].mean()),
            all_mitigations_full=r6(cv["all_full"].mean()))
        r["_case_values"] = cv
        res[lab] = r
    # case studies that gain (aggregators are monotone in X, so no case can lose)
    gains = OrderedDict()
    for k in ("max_w", "full", "mean_w", "all_partial", "all_full"):
        d = res["extended"]["_case_values"][k] - res["legacy"]["_case_values"][k]
        gains[k] = {"gain": int((d > 1e-12).sum()), "unchanged": int((np.abs(d) <= 1e-12).sum()), "loss": int((d < -1e-12).sum())}
    for lab in res:
        res[lab].pop("_case_values")
    res["case_studies_gaining"] = gains
    mt = AM.mitigated
    techs = np.array(AM.techs)
    res["techniques_all_mitigations_full"] = {lab: techs[(pts[lab]["all_full"] > 0) & mt].tolist() for lab in pts}
    res["techniques_all_mitigations_at_least_partial_count"] = {lab: int(((pts[lab]["all_partial"] > 0) & mt).sum()) for lab in pts}
    return res


def atlas_headline(AM, masks, codes=None):
    out = OrderedDict()
    for lab in ("legacy", "ai_only", "extended"):
        out[lab] = AM.summary(masks[lab], codes)
    cc = category_counts(AM, masks, codes)
    out["technical_ai_counts"] = {lab: cc["Technical - AI"][lab] for lab in ("legacy", "ai_only", "extended")}
    agg = aggregator_block(AM, masks, codes)
    out["aggregators"] = {lab: {"mitigated_techniques": agg[lab]["mitigated_techniques"], "case_studies": agg[lab]["case_studies"]}
                          for lab in ("legacy", "extended")}
    return out


def rationale_min_levels(grades, rationale, E):
    """Minimum level of each mapped *service* named in the rationale of a mitigation (e.g. 'IR-2 and IR-3' -> 2).
    Heuristic extraction used only for the level-aware sensitivity of the profile-level ATLAS coverage."""
    out = OrderedDict()
    for m, mp in grades.items():
        for x in mp:
            if x in E.cat.services:
                lv = [int(z) for z in re.findall(r"(?<![A-Z])" + x + r"-(\d)", rationale.get(m, ""))]
                if lv:
                    out.setdefault(m, OrderedDict())[x] = min(lv)
    return out


def level_aware_codes(AM, targets, minlv, levels):
    """Remove the credit of service x for mitigation m when the profile's level of x is below the level
    named in the rationale."""
    c = AM.codes.copy()
    removed = []
    for m, d in minlv.items():
        for x, n in d.items():
            if 0 < levels.get(x, 0) < n:
                j = targets.index(x)
                if c[AM.mix[m], j] > 0:
                    removed.append(f"{m}:{x}(needs {x}-{n}, profile has {x}-{levels.get(x)})")
                c[AM.mix[m], j] = 0
    return c, removed


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--out-dir", default=os.path.join(L.ROOT, "data", "results"))
    args = ap.parse_args()
    t0 = time.time()

    # ------------------------------------------------------------------ load everything at run time
    E = L.Elements()
    atlas_path, release = L.atlas_yaml_path()
    A = L.Atlas(atlas_path)
    grades, rationale = L.load_atlas_grades()
    fw, csf_w, csf_high, raw_cw = L.load_crosswalk()
    lvl_targets = L.level_targets([grades] + list(fw.values()))
    targets = E.X_ext + lvl_targets
    AM = L.AtlasModel(A, grades, targets)
    CM = L.CrosswalkModel(fw, csf_w, csf_high, targets)
    masks = masks_for(E, targets)
    ext_idx = {x: j for j, x in enumerate(E.X_ext)}
    atlas_cols = np.array([not t.startswith("X:") for t in targets])

    inputs = [L.PATHS[k] for k in ("atlas_map", "crosswalk", "legacy_guarantees", "ai_guarantees", "legacy_services",
                                   "ai_services", "rmf", "csf", "aiact", "ir8596", "ndtzi_profiles", "reference_profiles",
                                   "atlas_coverage", "crosswalk_coverage")] + [atlas_path]
    common = OrderedDict(
        element_sets=E.counts(), level_specific_targets_in_mappings=lvl_targets,
        atlas_release=release, atlas_file=L.rel(atlas_path),
        atlas_grades=OrderedDict(n_mitigations=len(AM.mits), n_mapped=int(AM.in_scope.sum()), n_grades=AM.table.n_grades(),
                                 n_full=int((AM.codes == 2).sum()), n_partial=int((AM.codes == 1).sum())),
        crosswalk_grades=OrderedDict(n_units_applicable=int(CM.codes.shape[0]), n_units_na=sum(len(t.na_units) for t in CM.tables.values()),
                                     n_grades=int((CM.codes > 0).sum()), n_full=int((CM.codes == 2).sum()),
                                     n_partial=int((CM.codes == 1).sum()),
                                     per_framework={n: {"applicable": len(t.units), "na": len(t.na_units), "grades": t.n_grades()}
                                                    for n, t in CM.tables.items()}))

    # ------------------------------------------------------------------ reproduction checks
    repro_a, repro_c = OrderedDict(), OrderedDict()
    base_a = {lab: AM.summary(masks[lab]) for lab in ("legacy", "extended")}
    base_c = {lab: CM.summary(masks[lab]) for lab in ("legacy", "ai_only", "extended")}
    try:
        old = json.load(open(L.PATHS["atlas_coverage"], encoding="utf8"))
        dif = []
        for sub in ("mitigated", "all"):
            for lab in ("legacy", "extended"):
                for k in ("n", "addressed", "full", "weighted"):
                    dif.append(abs(old[sub][lab][k] - base_a[lab][sub][k]))
        for k in ("weighted", "addressed", "full"):
            for lab in ("legacy", "extended"):
                dif.append(abs(old["case_means"][k][lab] - base_a[lab]["case_means"][k]))
        repro_a["max_abs_diff_summary"] = float(max(dif))
        cc0 = category_counts(AM, masks)
        ok = True
        for c, r in old["mitigation_categories"].items():
            ok &= (r["n"] == cc0[c]["n"] and r["legacy_F"] == cc0[c]["legacy"]["fully_required"]
                   and r["legacy_any"] == cc0[c]["legacy"]["at_least_partially_required"]
                   and r["ai_F"] == cc0[c]["ai_only"]["fully_required"] and r["ai_any"] == cc0[c]["ai_only"]["at_least_partially_required"]
                   and r["unmapped"] == len(cc0[c]["out_of_scope_unmapped"]))
        repro_a["mitigation_categories_identical"] = bool(ok)
        repro_a["_old_ai_marginal"] = old.get("ai_marginal")
    except Exception as e:  # noqa: BLE001
        repro_a["error"] = repr(e)
    try:
        old = json.load(open(L.PATHS["crosswalk_coverage"], encoding="utf8"))
        dif = []
        for n in L.FW_ORDER + [L.CSF_HIGH]:
            for lab in ("legacy", "ai_only", "extended"):
                o, m = old[n][lab], base_c[lab][n]
                dif += [abs(o["w0.25"] - m["w0_25"]), abs(o["w0.5"] - m["w0_5"]), abs(o["w0.75"] - m["w0_75"]),
                        abs(o["strict"] - m["strict"]), abs(o["addressed"] - m["addressed"]), abs(o["n"] - m["n"])]
        for lab in ("legacy", "extended"):
            dif.append(abs(old["NIST CSF 2.0"]["priority_weighted"][lab] - base_c[lab]["NIST CSF 2.0"]["priority_weighted_w0_5"]))
            for k, v in old["NIST AI RMF by function"].items():
                dif.append(abs(v[lab] - base_c[lab]["NIST AI RMF by function"][k]["w0_5"]))
            for k, v in old["EU AI Act by article"].items():
                dif.append(abs(v[lab] - base_c[lab]["EU AI Act by article"][k]["w0_5"]))
        repro_c["max_abs_diff_vs_crosswalk_coverage_json"] = float(max(dif))
    except Exception as e:  # noqa: BLE001
        repro_c["error"] = repr(e)

    # ------------------------------------------------------------------ T1
    t1 = category_counts(AM, masks)
    tai = t1["Technical - AI"]
    t1_summary = OrderedDict(
        n=tai["n"],
        legacy=tai["legacy"], class_I_plus_AI_rows_alone=tai["ai_only"], extended=tai["extended"],
        explanation=("Extended counts exceed the class-I-only counts by the mitigations that are fully required (or touched) "
                     "only through legacy targets: " + json.dumps(tai["extended_minus_ai_only"])))

    # ------------------------------------------------------------------ T2
    t2 = OrderedDict()
    t2["all_mitigations_in_M(t)"] = aggregator_block(AM, masks)
    t2["in_scope_mitigations_only"] = aggregator_block(AM, masks, scope=AM.in_scope)
    t2["in_scope_mitigations_only"]["note"] = (
        "M(t) restricted to mitigations with a non-empty mapping; mitigations deliberately left unmapped as outside the "
        f"scope of product evaluation ({[m for m, s in zip(AM.mits, AM.in_scope) if not s]}) are dropped; techniques "
        "whose only mitigations are out of scope score 0.")

    # ------------------------------------------------------------------ T3a
    ai_targets = set(E.ai_svc) | set(E.ai_rows)
    t3a_a, t3a_c = OrderedDict(), OrderedDict()
    for lab, which in (("downgrade_ai_services_and_ai_rows", ai_targets),
                       ("downgrade_ai_services_ai_rows_and_X:AIPROFILE", ai_targets | {"X:AIPROFILE"})):
        ca, na = downgrade(AM.codes, targets, which)
        cc, nc = downgrade(CM.codes, targets, which)
        t3a_a[lab] = OrderedDict(n_full_grades_downgraded=na, metrics=atlas_headline(AM, masks, ca))
        t3a_c[lab] = OrderedDict(n_full_grades_downgraded=nc,
                                 metrics={k: CM.summary(masks[k], cc) for k in ("legacy", "ai_only", "extended")})
    t3a_a["baseline"] = atlas_headline(AM, masks)
    t3a_c["baseline"] = {k: base_c[k] for k in ("legacy", "ai_only", "extended")}

    # ------------------------------------------------------------------ T3b
    def atlas_rep_metrics(codes):
        out = {}
        for lab in ("legacy", "extended"):
            pt = AM.per_technique(masks[lab], codes)
            mt = AM.mitigated
            out[f"{lab}|full"] = pt["full"][mt].mean()
            out[f"{lab}|weighted"] = pt["max_w"][mt].mean()
            out[f"{lab}|addressed"] = pt["addressed"][mt].mean()
            out[f"{lab}|mean_over_mitigations"] = pt["mean_w"][mt].mean()
            out[f"{lab}|all_mitigations_at_least_partial"] = pt["all_partial"][mt].mean()
            out[f"{lab}|case_full"] = AM.case_values(pt["full"]).mean()
            out[f"{lab}|case_weighted"] = AM.case_values(pt["max_w"]).mean()
        tj = np.array([AM.mix[m] for m in AM.categories["Technical - AI"]])
        mb = AM.mitigation_best(masks["extended"], codes)[tj]
        out["extended|technical_ai_fully_required"] = float((mb == 2).sum())
        out["extended|technical_ai_at_least_partially_required"] = float((mb >= 1).sum())
        return out

    def cw_rep_metrics(codes):
        out = {}
        for lab in ("legacy", "extended"):
            s = CM.summary(masks[lab], codes, full=False)
            for n in HEADLINE_FW + EXTRA_FW:
                out[f"{lab}|{n}|w0_5"] = s[n]["w0_5"]
            out[f"{lab}|NIST CSF 2.0|priority_weighted_w0_5"] = s["NIST CSF 2.0"]["priority_weighted_w0_5"]
            best = L.best_codes(codes, masks[lab])
            for n in HEADLINE_FW:
                out[f"{lab}|{n}|strict"] = float((best[CM.rows[n]] == 2).mean())
        return out

    def summarise(recs, base):
        keys = list(recs[0])
        arr = {k: np.array([r[k] for r in recs]) for k in keys}
        out = OrderedDict()
        for k in keys:
            out[k] = OrderedDict(baseline=r6(base[k]), **{q: r6(v) for q, v in L.pct(arr[k]).items()})
        # gains extended - legacy
        for k in keys:
            if k.startswith("extended|"):
                kl = "legacy|" + k[len("extended|"):]
                if kl in arr:
                    g = arr[k] - arr[kl]
                    out["gain|" + k[len("extended|"):]] = OrderedDict(
                        baseline=r6(base[k] - base[kl]), **{q: r6(v) for q, v in L.pct(g).items()},
                        share_of_replicates_extended_gt_legacy=r6((g > 1e-12).mean()))
        return out

    def run_reps(symmetric: bool):
        rng = np.random.default_rng(args.seed)
        ra, rc = [], []
        change_types = np.zeros(4)                                  # F->P, F->none, P->F, P->none
        ka = kc = 0
        for _ in range(args.reps):
            ca, ka = L.reassign(AM.codes, rng, args.frac)
            cc, kc = L.reassign(CM.codes, rng, args.frac)
            for old, new in ((AM.codes, ca), (CM.codes, cc)):
                ch = old != new                                   # every reassigned grade changes
                o, n_ = old[ch], new[ch]
                change_types += [((o == 2) & (n_ == 1)).sum(), ((o == 2) & (n_ == 0)).sum(),
                                 ((o == 1) & (n_ == 2)).sum(), ((o == 1) & (n_ == 0)).sum()]
            if symmetric:
                ca = L.add_random(ca, rng, ka, AM.in_scope, atlas_cols & np.array([not L.SVC_LEVEL_RE.match(t) for t in targets]))
                cc = L.add_random(cc, rng, kc, np.ones(CM.codes.shape[0], bool), np.array([not L.SVC_LEVEL_RE.match(t) for t in targets]))
            ra.append(atlas_rep_metrics(ca))
            rc.append(cw_rep_metrics(cc))
        return ra, rc, ka, kc, change_types / args.reps

    base_ra, base_rc = atlas_rep_metrics(AM.codes), cw_rep_metrics(CM.codes)
    t3b_a, t3b_c = OrderedDict(), OrderedDict()
    for lab, sym in (("reassign_10pct", False), ("reassign_10pct_plus_equal_random_additions", True)):
        ra, rc, ka, kc, ct = run_reps(sym)
        design = OrderedDict(replicates=args.reps, seed=args.seed, fraction=args.frac,
                             atlas_grades=AM.table.n_grades(), atlas_grades_reassigned_per_replicate=ka,
                             crosswalk_grades=int((CM.codes > 0).sum()), crosswalk_grades_reassigned_per_replicate=kc,
                             mean_changes_per_replicate_atlas_plus_crosswalk={"F->P": r6(ct[0]), "F->none": r6(ct[1]),
                                                                              "P->F": r6(ct[2]), "P->none": r6(ct[3])},
                             rule="reassigned grades chosen uniformly without replacement among existing grades; F -> P or none, "
                                  "P -> F or none with probability 1/2 each; the crosswalk is perturbed as one pool of grades "
                                  "across the five frameworks (the CSF high-priority subset inherits the CSF perturbation)")
        if sym:
            design["additions"] = ("in addition, as many new grades as were reassigned (F or P with probability 1/2) are placed on "
                                   "uniformly chosen ungraded cells: ATLAS, mapped mitigations x services and assurance rows; "
                                   "crosswalk, applicable units x all elements of X_ext. Secondary analysis (omission errors), not "
                                   "requested by the reviewer.")
        t3b_a[lab] = OrderedDict(design=design, metrics=summarise(ra, base_ra))
        t3b_c[lab] = OrderedDict(design=design, metrics=summarise(rc, base_rc))

    # ------------------------------------------------------------------ T4
    def atlas_metrics(mask):
        s = AM.summary(mask)
        return OrderedDict(mitigated_weighted=s["mitigated"]["weighted"], mitigated_full=s["mitigated"]["full"],
                           mitigated_addressed=s["mitigated"]["addressed"], case_weighted=s["case_means"]["weighted"],
                           case_full=s["case_means"]["full"])

    def cw_metrics(mask):
        s = CM.summary(mask)
        out = OrderedDict()
        for n in HEADLINE_FW + EXTRA_FW:
            out[f"{n}|w0_5"] = s[n]["w0_5"]
        for n in HEADLINE_FW:
            out[f"{n}|strict"] = s[n]["strict"]
        return out

    full_a, full_c = atlas_metrics(masks["extended"]), cw_metrics(masks["extended"])
    leg_a, leg_c = atlas_metrics(masks["legacy"]), cw_metrics(masks["legacy"])
    loo_a, loo_c, add_a, add_c = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    loo_a_raw = OrderedDict()
    for x in E.X_ext:
        mk = L.xmask(targets, [y for y in E.X_ext if y != x], cat=E.cat)
        ma, mc = atlas_metrics(mk), cw_metrics(mk)
        loo_a_raw[x] = full_a["mitigated_weighted"] - ma["mitigated_weighted"]
        loo_a[x] = OrderedDict((k, r6(full_a[k] - ma[k])) for k in full_a)
        loo_c[x] = OrderedDict((k, r6(full_c[k] - mc[k])) for k in full_c)
        if x in E.X_ai:
            mk2 = L.xmask(targets, E.X_leg + [x], cat=E.cat)
            ma2, mc2 = atlas_metrics(mk2), cw_metrics(mk2)
            add_a[x] = OrderedDict((k, r6(ma2[k] - leg_a[k])) for k in leg_a)
            add_c[x] = OrderedDict((k, r6(mc2[k] - leg_c[k])) for k in leg_c)
    groups = OrderedDict([("legacy_services", E.leg_svc), ("legacy_rows", E.leg_rows), ("X:PROFILE", E.prof_leg),
                          ("class_I_services", E.ai_svc), ("AI_rows", E.ai_rows), ("X:AIPROFILE", E.prof_ai),
                          ("profile_mechanisms", E.prof_leg + E.prof_ai),
                          ("class_I_plus_AI_rows_plus_X:AIPROFILE", E.X_ai)])
    grp_a, grp_c = OrderedDict(), OrderedDict()
    grp_a_raw = OrderedDict()
    for g, mem in groups.items():
        mk = L.xmask(targets, [y for y in E.X_ext if y not in set(mem)], cat=E.cat)
        ma, mc = atlas_metrics(mk), cw_metrics(mk)
        grp_a_raw[g] = full_a["mitigated_weighted"] - ma["mitigated_weighted"]
        grp_a[g] = OrderedDict((k, r6(full_a[k] - ma[k])) for k in full_a)
        grp_c[g] = OrderedDict((k, r6(full_c[k] - mc[k])) for k in full_c)

    # Shapley (exact)
    Va = element_player_matrix(E, targets, AM.element_values(0.5))[AM.mitigated]
    Vc = element_player_matrix(E, targets, CM.unit_values(0.5))
    shap_a = OrderedDict([("mitigated_weighted", L.shapley_max_rows(Va).mean(axis=0)),
                          ("mitigated_full", L.shapley_max_rows((Va >= 1).astype(float)).mean(axis=0))])
    shap_c = OrderedDict()
    for n in HEADLINE_FW + EXTRA_FW:
        V = Vc[CM.rows[n]]
        shap_c[f"{n}|w0_5"] = L.shapley_max_rows(V).mean(axis=0)
        if n in HEADLINE_FW:
            shap_c[f"{n}|strict"] = L.shapley_max_rows((V >= 1).astype(float)).mean(axis=0)
    eff = []
    for k, v in shap_a.items():
        eff.append(abs(v.sum() - full_a[k]))
    for k, v in shap_c.items():
        eff.append(abs(v.sum() - full_c[k]))
    # brute-force verification of the closed form on random 10-player sub-games of real techniques/units
    rng = np.random.default_rng(args.seed)
    bf_err = 0.0
    for V in (Va, Vc):
        for _ in range(20):
            row = V[rng.integers(V.shape[0])]
            cols = rng.choice(V.shape[1], size=10, replace=False)
            sub = row[cols]
            bf_err = max(bf_err, float(np.abs(L.shapley_max_rows(sub[None, :])[0] - L.shapley_bruteforce(sub)).max()))

    def shap_table(sh):
        out = OrderedDict()
        for k, v in sh.items():
            order = np.argsort(-v, kind="stable")
            out[k] = OrderedDict(
                total=r6(v.sum()),
                by_group=OrderedDict((g, r6(sum(v[ext_idx[x]] for x in mem))) for g, mem in groups.items()
                                     if g not in ("profile_mechanisms", "class_I_plus_AI_rows_plus_X:AIPROFILE")),
                top10=[[E.X_ext[i], r6(v[i])] for i in order[:10]],
                all={x: r6(v[ext_idx[x]]) for x in E.X_ext})
        return out

    def loo_summary(loo, metric_keys):
        out = OrderedDict()
        for k in metric_keys:
            vals = [(x, d[k]) for x, d in loo.items()]
            nz = sorted([(x, v) for x, v in vals if v > 1e-12], key=lambda z: -z[1])
            out[k] = OrderedDict(n_elements_with_unique_contribution=len(nz), elements=[[x, v] for x, v in nz],
                                 n_elements_without_unique_contribution=len(vals) - len(nz))
        return out

    old_m = repro_a.pop("_old_ai_marginal", None)
    if old_m:
        dd = [abs(old_m[s] - loo_a_raw[s]) for s in old_m if s in loo_a_raw]
        if "AI-rows" in old_m:
            dd.append(abs(old_m["AI-rows"] - grp_a_raw["AI_rows"]))
        repro_a["ai_marginal_max_abs_diff"] = float(max(dd))

    def element_table(loo, add, shap, pairs):
        """Per element of X_ai: LOO, contribution over X_leg and Shapley value for the given (loo/add key, shapley key)."""
        tab = OrderedDict()
        for x in E.X_ai:
            row = OrderedDict()
            for k_lo, k_sh in pairs:
                row[f"loo|{k_lo}"] = loo[x][k_lo]
                row[f"over_legacy|{k_lo}"] = add[x][k_lo]
                row[f"shapley|{k_sh}"] = r6(shap[k_sh][ext_idx[x]])
            tab[x] = row
        return tab

    def zero_shapley(sh):
        return OrderedDict((k, [x for x in E.X_ext if abs(v[ext_idx[x]]) < 1e-12]) for k, v in sh.items())

    t4_a = OrderedDict(
        definition="LOO(x) = metric(X_ext) - metric(X_ext without x); group removal analogous; contribution over legacy = "
                   "metric(X_leg + {x}) - metric(X_leg) for x in X_ai; Shapley values are exact (airport-game formula for "
                   "max-coverage games, verified by enumeration) and add up to metric(X_ext).",
        metrics_at_X_ext={k: r6(v) for k, v in full_a.items()}, metrics_at_X_leg={k: r6(v) for k, v in leg_a.items()},
        existing_ai_marginal_in_atlas_coverage_json=old_m,
        class_I_and_AI_rows_table=element_table(loo_a, add_a, shap_a, [("mitigated_weighted", "mitigated_weighted"),
                                                                       ("mitigated_full", "mitigated_full")]),
        loo=loo_a, loo_summary=loo_summary(loo_a, list(full_a)), group_removal=grp_a,
        contribution_over_legacy=add_a, shapley=shap_table(shap_a), elements_with_zero_shapley=zero_shapley(shap_a),
        checks=OrderedDict(shapley_efficiency_max_abs_err=float(max(eff)), shapley_closed_form_vs_enumeration_max_abs_err=bf_err))
    t4_c = OrderedDict(
        definition=t4_a["definition"], metrics_at_X_ext={k: r6(v) for k, v in full_c.items()},
        metrics_at_X_leg={k: r6(v) for k, v in leg_c.items()},
        class_I_and_AI_rows_table=element_table(loo_c, add_c, shap_c, [(f"{n}|w0_5", f"{n}|w0_5") for n in HEADLINE_FW]),
        loo=loo_c, loo_summary=loo_summary(loo_c, list(full_c)), group_removal=grp_c,
        contribution_over_legacy=add_c, shapley=shap_table(shap_c), elements_with_zero_shapley=zero_shapley(shap_c),
        checks=t4_a["checks"])

    # ------------------------------------------------------------------ T5
    base_prof = {p["id"]: p["services"] for p in json.load(open(L.PATHS["ndtzi_profiles"], encoding="utf8"))["profiles"]}
    refp = json.load(open(L.PATHS["reference_profiles"], encoding="utf8"))
    minlv = rationale_min_levels(grades, rationale, E)
    t5_a, t5_c = OrderedDict(), OrderedDict()
    cat_c = {k: CM.summary(masks[k]) for k in ("legacy", "extended")}
    cat_a = {k: AM.summary(masks[k]) for k in ("legacy", "extended")}
    for p in refp:
        cl, g = E.cat.closure({**base_prof[p["base"]], **p["ai"]}, p["G"])
        consistent = (cl == p.get("closure")) and (g == p.get("G_closure"))
        rows_leg = [r for r in E.leg_rows if E.rows_info[r]["first_plus"] and E.rows_info[r]["first_plus"] <= g]
        rows_ai = [r for r in E.ai_rows if E.rows_info[r]["first_plus"] and E.rows_info[r]["first_plus"] <= g]
        legacy_base = base_prof[p["base"]]
        sets = OrderedDict([
            ("profile_extended", (list(cl) + rows_leg + rows_ai + E.prof_leg + E.prof_ai, cl)),
            ("profile_extended_without_profile_mechanisms", (list(cl) + rows_leg + rows_ai, cl)),
            ("profile_legacy_base", (list(legacy_base) + rows_leg + E.prof_leg, legacy_base)),
        ])
        pm = {k: L.xmask(targets, mem, levels=lv) for k, (mem, lv) in sets.items()}
        info = OrderedDict(
            id=p["id"], domain=p["domain"], legacy_base=p["base"], G=g,
            closure_recomputed_equals_reference_profiles_json=bool(consistent),
            services=OrderedDict((s, cl[s]) for s in E.cat.order() if s in cl),
            n_services=len(cl), class_I_services_absent=[s for s in E.ai_svc if s not in cl],
            legacy_rows_included=rows_leg, legacy_rows_excluded=[r for r in E.leg_rows if r not in rows_leg],
            ai_rows_included=rows_ai, ai_rows_excluded=[r for r in E.ai_rows if r not in rows_ai],
            n_elements=OrderedDict((k, int(m.sum())) for k, m in pm.items()))
        cw_block = OrderedDict()
        for n in HEADLINE_FW + EXTRA_FW + ["NIST CSF 2.0 priority-weighted"]:
            key = "NIST CSF 2.0" if n.endswith("priority-weighted") else n
            row = OrderedDict()
            for lab, src in (("catalogue_legacy", cat_c["legacy"]), ("catalogue_extended", cat_c["extended"])):
                row[lab] = ({"w0_5": r6(src[key]["priority_weighted_w0_5"])} if n.endswith("priority-weighted")
                            else {k: r6(v) for k, v in src[key].items()})
            for lab, m in pm.items():
                s = CM.summary(m)
                row[lab] = ({"w0_5": r6(s[key]["priority_weighted_w0_5"])} if n.endswith("priority-weighted")
                            else {k: r6(v) for k, v in s[key].items()})
            cw_block[n] = row
        t5_c[p["id"]] = OrderedDict(profile=info, coverage=cw_block)
        at_block = OrderedDict(catalogue_legacy=cat_a["legacy"], catalogue_extended=cat_a["extended"])
        for lab, m in pm.items():
            at_block[lab] = AM.summary(m)
        lac, removed = level_aware_codes(AM, targets, minlv, cl)
        at_block["profile_extended_level_aware_from_rationale"] = AM.summary(pm["profile_extended"], lac)
        lac_b, removed_b = level_aware_codes(AM, targets, minlv, legacy_base)
        at_block["profile_legacy_base_level_aware_from_rationale"] = AM.summary(pm["profile_legacy_base"], lac_b)
        t5_a[p["id"]] = OrderedDict(profile=info, coverage=at_block,
                                    level_aware_credits_removed_extended=removed, level_aware_credits_removed_legacy_base=removed_b)

    # ------------------------------------------------------------------ write
    t1_end = time.time()
    seeds = {"perturbation": args.seed, "shapley_verification_subgames": args.seed}
    meta = L.meta_block(__file__, t0, t1_end, seeds, inputs,
                        notes="Reads the specification at run time; no existing file is modified.")
    atlas_out = OrderedDict(
        meta=meta, common=common, reproduction_check=repro_a,
        T1_category_counts=OrderedDict(technical_ai=t1_summary, all_categories=t1),
        T2_aggregators=t2,
        T3a_downgrade=t3a_a, T3b_random_perturbation=t3b_a,
        T4_marginal_contributions=t4_a,
        T5_profile_level=OrderedDict(
            definition=("X_p = services of the closed reference profile (any level; the mappings are not level-specific) + "
                        "assurance rows whose first '+' mark is at G-k <= G of the profile + X:PROFILE + X:AIPROFILE. "
                        "'profile_legacy_base' = services of the legacy base profile of ND TZI 2.5-005-99 + legacy rows at <= G + "
                        "X:PROFILE. 'level_aware_from_rationale' additionally withdraws the credit of a service when the "
                        "mitigation's rationale names a higher level of that service than the profile requires "
                        "(heuristic regex extraction)."),
            rationale_min_levels=minlv, profiles=t5_a))
    cw_out = OrderedDict(
        meta=meta, common=common, reproduction_check=repro_c,
        baseline=OrderedDict((k, base_c[k]) for k in ("legacy", "ai_only", "extended")),
        T3a_downgrade=t3a_c, T3b_random_perturbation=t3b_c,
        T4_marginal_contributions=t4_c,
        T5_profile_level=OrderedDict(definition=atlas_out["T5_profile_level"]["definition"].split(" 'level_aware")[0],
                                     profiles=t5_c))
    os.makedirs(args.out_dir, exist_ok=True)
    pa = os.path.join(args.out_dir, "atlas_sensitivity.json")
    pc = os.path.join(args.out_dir, "crosswalk_sensitivity.json")
    json.dump(L.jsonable(atlas_out), open(pa, "w", encoding="utf8"), indent=1)
    json.dump(L.jsonable(cw_out), open(pc, "w", encoding="utf8"), indent=1)
    print(f"wrote {L.rel(pa)} and {L.rel(pc)} in {t1_end - t0:.1f} s")
    print("reproduction:", json.dumps(repro_a), json.dumps(repro_c))


if __name__ == "__main__":
    main()
