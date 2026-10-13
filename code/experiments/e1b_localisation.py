"""E1b: the IE-2 explanation test against trivial baselines (major revision, reviewer request).

Reviewers noted that most detected attacks target P1, that P1 supplies 34 of the 68 model channels and that
hai_lib.localise attributes an alarm to the process with the largest SUM of per-channel contributions, so a
constant "P1" explainer could pass the pre-registered IE-2 test (process-localisation accuracy >= 0.60) too.
For both pre-registered variants (baseline_s0, hardened_s0) this script

  0. recomputes the residuals of the four test files with the stored models, reproduces the smoothed score,
     the alarm episodes, the detected-attack vector, the alarm start and the original (sum) localisation of
     e1_baseline.py, and keeps the per-channel contributions over the same horizon (30 s from the alarm
     start, hai_lib.localise defaults);
  1. (a) evaluates the constant explainers (P1 and the best constant process) with exact one-sided bounds;
     (b) the channel-proportional random explainer (process drawn with probability n_P / 68): exact
         Poisson-binomial distribution of its hits; for reference also a uniform random process and an
         aggregator-specific channel-label permutation null (exploratory, Monte Carlo);
     (c) an exact one-sided McNemar test of the model against the constant P1 explainer (paired outcomes);
     (d) localisation with per-process MEAN contributions (sum divided by the process's channel count) and,
         as an extra check, the process of the top-ranked channel (per-process MAX);
     (e) the tag-level top-5 hit rate against its chance level (5 of the 68 channels drawn at random) and
         against the best constant 5-tag set (selected in-sample, hence optimistic).
An attack on two processes counts as a hit if either of them ranks first; every chance rate uses that rule.

Writes results/e1b_localisation.json. Usage: python e1b_localisation.py [--perm 20000] [--seed 20261012]"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
from scipy import stats as st

import hai_lib as H
import rev_lib as R
from aitcert_path import stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--perm", type=int, default=20000, help="label permutations per attack (exploratory null)")
ap.add_argument("--seed", type=int, default=20261012)
args = ap.parse_args()
t0, started = time.time(), R.utc_now()
SCRIPT = "e1b_localisation.py"
TAU, HORIZON, TOPK = 0.60, 30, 5
ALPHAS = (0.10, 0.05, 0.01)
A0 = 0.05                                   # pre-registered alpha_G (G-4)
VARIANTS = ("baseline_s0", "hardened_s0")
torch.manual_seed(0)

prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
cols = list(prep.cols)
F = len(cols)
proc_of = np.array([c.split("_")[0] for c in cols])
PROCS = sorted(set(proc_of.tolist()))
n_ch = {p: int((proc_of == p).sum()) for p in PROCS}
n_vec = np.array([n_ch[p] for p in PROCS], dtype=np.float64)
onehot = np.stack([(proc_of == p) for p in PROCS], 1)            # [F, P] bool
col_set = set(cols)
norm = lambda s: "".join(ch for ch in s.upper() if ch.isalnum())
colnorm = {norm(c): c for c in cols}

attacks = R.load_attacks()
by_file = {f"test{i}": [a for a in attacks if a["test_file"] == f"test{i}"] for i in range(1, 5)}
test = H.read_files("hai-22.04", [f"test{i}.csv" for i in range(1, 5)])
arrays = [prep.to_array(df) for df in test]
ys = [H.labels(df) for df in test]
del test

checks = {"label_alignment": []}
for fi in range(1, 5):
    ev = H.events_from_labels(ys[fi - 1])
    sub = by_file[f"test{fi}"]
    if len(ev) != len(sub):
        raise RuntimeError(f"test{fi}: {len(ev)} labelled events but {len(sub)} attacks in the table")
    for (s, e), a in zip(ev, sub):
        if (s, e) != (a["label_start_idx"], a["label_end_idx"]):
            checks["label_alignment"].append({"attack_id": a["attack_id"], "events": [s, e],
                                              "table": [a["label_start_idx"], a["label_end_idx"]]})
checks["label_alignment_ok"] = not checks["label_alignment"]


def bound_block(k, n):
    """Estimate, exact one-sided Clopper-Pearson lower bounds and IE-2 decisions."""
    lcb = {str(a): S.cp_lower(k, n, a) for a in ALPHAS}
    lo2, hi2 = S.cp_interval(k, n, 0.05)
    return {"k": int(k), "n": int(n), "estimate": k / n if n else None, "lcb": lcb,
            "cp_two_sided_95": [lo2, hi2],
            "ie2_pass_lcb": {a: bool(v >= TAU) for a, v in lcb.items()}, "ie2_pass_naive": bool(k / n >= TAU)}


def pass_probabilities(pmf, n):
    xs = np.arange(len(pmf))
    out = {"p_pass_naive": float(pmf[xs / n >= TAU].sum())}
    for a in ALPHAS:
        ok = np.array([S.cp_lower(int(x), n, a) >= TAU for x in xs])
        out[f"p_pass_lcb_{a}"] = float(pmf[ok].sum())
    return out


def mcnemar_one_sided(model_hit, other_hit):
    """Exact one-sided McNemar (sign) test of H1: the model hits more often than the comparator."""
    m, o = np.asarray(model_hit, bool), np.asarray(other_hit, bool)
    b, c = int((m & ~o).sum()), int((~m & o).sum())
    p = float(st.binom.sf(b - 1, b + c, 0.5)) if b + c > 0 else 1.0
    return {"model_only_hits_b": b, "comparator_only_hits_c": c, "discordant": b + c,
            "concordant_hits": int((m & o).sum()), "concordant_misses": int((~m & ~o).sum()),
            "p_one_sided_exact": p}


def tag_chance(m):
    """P(a random set of TOPK of the F channels contains at least one of m target channels)."""
    return 1.0 - (math.comb(F - m, TOPK) / math.comb(F, TOPK) if F - m >= TOPK else 0.0)


def _combo_hits(target_sets):
    """Hit matrix [combinations of TOPK candidate tags, attacks] for constant tag sets."""
    cand = sorted(set().union(*target_sets)) if target_sets else []
    combos = list(itertools.combinations(cand, min(TOPK, len(cand))))
    member = np.array([[t in ts for ts in target_sets] for t in cand], dtype=bool)      # [tags, attacks]
    idx = {t: i for i, t in enumerate(cand)}
    hm = np.zeros((len(combos), len(target_sets)), dtype=bool)
    for j, combo in enumerate(combos):
        hm[j] = member[[idx[t] for t in combo]].any(0)
    return combos, hm


def best_constant_tagset(target_sets):
    """In-sample best constant TOPK-tag set (first optimum in lexicographic order) and leave-one-out estimate."""
    combos, hm = _combo_hits(target_sets)
    tot = hm.sum(1)
    j = int(np.argmax(tot))
    n = hm.shape[1]
    loo_tie_avg, loo_first = [], []
    for i in range(n):
        s = tot - hm[:, i]
        sel = s == s.max()
        loo_tie_avg.append(float(hm[sel, i].mean()))          # average over all optimal sets (random tie-break)
        loo_first.append(bool(hm[int(np.argmax(s)), i]))      # lexicographically first optimal set
    return {"in_sample": {"tags": list(combos[j]), "k": int(tot[j]), "n_optimal_sets": int((tot == tot[j]).sum())},
            "leave_one_out": {"expected_hits_tie_averaged": float(sum(loo_tie_avg)),
                              "expected_accuracy_tie_averaged": float(sum(loo_tie_avg) / n),
                              "hits_first_optimum": int(sum(loo_first))}}


rng = np.random.default_rng(args.seed)
perms = np.argsort(rng.random((args.perm, F)), axis=1)              # label permutations (common to all attacks)
lab = onehot[perms]                                                 # [R, F, P] permuted process membership
labf = lab.astype(np.float64)
_perm_cache = {}


def permutation_null(key, contrib, targets):
    """Probability that each aggregator ranks a target process first when channel labels are exchangeable."""
    if key in _perm_cache:
        return _perm_cache[key]
    c = np.asarray(contrib, dtype=np.float64)
    sums = np.einsum("f,rfp->rp", c, labf)
    means = sums / n_vec[None, :]
    maxs = np.where(lab, c[None, :, None], -np.inf).max(1)
    tmask = np.array([p in targets for p in PROCS])
    res = {agg: float(tmask[np.argmax(v, 1)].mean()) for agg, v in (("sum", sums), ("mean", means), ("max", maxs))}
    _perm_cache[key] = res
    return res


variants = {}
for tag in VARIANTS:
    meta = json.load(open(os.path.join(H.RES, f"e1_{tag}.json")))
    theta = meta["theta"]
    cv = np.load(os.path.join(H.RES, "models", f"{tag}_chanvar.npy"))
    saved = np.load(os.path.join(H.RES, "models", f"{tag}_scores.npz"))
    model = H.load_model(tag, F)
    e1loc = {l["attack_id"]: l for l in meta["localisation"]}
    recs, det_vec, score_diff = [], [], {}
    for fi in range(1, 5):
        arr, y = arrays[fi - 1], ys[fi - 1]
        res = H.residuals(model, arr)
        sc = H.smooth_fast(H.score_from_residuals(res, cv))
        ok = ~np.isnan(sc)
        score_diff[f"test{fi}"] = float(np.max(np.abs(sc[ok].astype(np.float32) - saved[f"test{fi}"][ok])))
        ev = H.evaluate_file(sc, y, theta)
        for (s, e), d, a in zip(ev["events"], ev["detected"], by_file[f"test{fi}"]):
            det_vec.append(int(d))
            if not d:
                continue
            hit = [al for al in ev["alarms"] if al[0] <= e + H.TOL and al[1] >= s]
            start = max(min(al[0] for al in hit), s)
            L = H.localise(res, cv, cols, start, horizon=HORIZON, topk=TOPK)     # original rule (sum)
            contrib = np.nanmean(res[start:start + HORIZON] / cv[None, :], axis=0).astype(np.float64)
            psum = {p: float(L["process_scores"][p]) for p in PROCS}
            pmean = {p: psum[p] / n_ch[p] for p in PROCS}
            pmax = {p: float(contrib[proc_of == p].max()) for p in PROCS}
            top_mean = max(PROCS, key=lambda p: pmean[p])
            top_max = L["top_tags"][0].split("_")[0]
            old = e1loc.get(a["attack_id"], {})
            tp_raw = set(a["target_points"])
            recs.append({
                "attack_id": a["attack_id"], "test_file": f"test{fi}", "primitives": a["primitives"],
                "target_processes": a["target_processes"], "target_points": a["target_points"],
                "event": [int(s), int(e)], "alarm_start": int(start), "alarm_start_offset_s": int(start - s),
                "process_sum": psum, "process_mean": pmean, "process_max": pmax,
                "top_sum": L["top_process"], "top_mean": top_mean, "top_max": top_max,
                "top_tags": L["top_tags"], "tag_hit_at5": bool(set(L["top_tags"]) & tp_raw),
                "contrib": [round(float(v), 6) for v in contrib],
                "reproduces_e1": {"top_process": old.get("top_process") == L["top_process"],
                                  "top_tags": old.get("top_tags") == L["top_tags"],
                                  "tag_hit_at5": old.get("tag_hit_at5") == bool(set(L["top_tags"]) & tp_raw)}})
        del res
    del model
    torch.cuda.empty_cache()
    rep = {"detected_vector_matches_e1": det_vec == meta["detected_vector"],
           "n_detected": len(recs), "n_detected_e1": len(meta["localisation"]),
           "max_abs_score_diff_vs_saved_float32": score_diff,
           "all_localisations_reproduced": all(all(r["reproduces_e1"].values()) for r in recs)}
    variants[tag] = {"theta": theta, "reproduction": rep, "records": recs}
    print(tag, json.dumps(rep), flush=True)


def analyse(recs, exclude=()):
    recs = [r for r in recs if r["attack_id"] not in exclude]
    n = len(recs)
    T = [set(r["target_processes"]) for r in recs]
    out = {"n": n, "n_targeting": {p: int(sum(p in t for t in T)) for p in PROCS},
           "n_two_process_attacks": int(sum(len(t) > 1 for t in T))}
    hits = {agg: np.array([r[f"top_{agg}"] in t for r, t in zip(recs, T)]) for agg in ("sum", "mean", "max")}
    # (a) constant explainers
    const = {p: bound_block(int(sum(p in t for t in T)), n) for p in PROCS}
    best = max(PROCS, key=lambda p: const[p]["k"])
    out["a_constant"] = {"per_process": const, "best_process": best,
                         "note": "the best constant process is chosen in-sample (optimistic)"}
    # model explainers (sum = pre-registered rule of e1; mean = channel-count normalised; max = top channel)
    out["model"] = {agg: bound_block(int(h.sum()), n) for agg, h in hits.items()}
    out["model"]["sum"]["misses"] = [r["attack_id"] for r, h in zip(recs, hits["sum"]) if not h]
    out["model"]["mean"]["misses"] = [r["attack_id"] for r, h in zip(recs, hits["mean"]) if not h]
    out["model"]["max"]["misses"] = [r["attack_id"] for r, h in zip(recs, hits["max"]) if not h]
    out["model"]["top_process_counts"] = {agg: {p: int(sum(r[f"top_{agg}"] == p for r in recs)) for p in PROCS}
                                          for agg in ("sum", "mean", "max")}
    # (b) random explainers
    rand = {}
    for name, q in (("channel_proportional", [sum(n_ch[p] for p in t) / F for t in T]),
                    ("uniform_process", [len(t & set(PROCS)) / len(PROCS) for t in T])):
        pmf = R.poisson_binomial_pmf(q)
        blk = R.pmf_summary(pmf, n)
        blk.update(pass_probabilities(pmf, n))
        blk["per_attack_hit_probability"] = {r["attack_id"]: qi for r, qi in zip(recs, q)}
        blk["p_ge_model_hits"] = {agg: R.tail_ge(pmf, int(h.sum())) for agg, h in hits.items()}
        rand[name] = blk
    perm_pi = [permutation_null((r["attack_id"], tuple(r["contrib"])), r["contrib"], set(r["target_processes"]))
               for r in recs]
    pnull = {}
    for agg in ("sum", "mean", "max"):
        q = [p[agg] for p in perm_pi]
        pmf = R.poisson_binomial_pmf(q)
        blk = R.pmf_summary(pmf, n)
        blk.update(pass_probabilities(pmf, n))
        blk["p_ge_model_hits"] = R.tail_ge(pmf, int(hits[agg].sum()))
        blk["per_attack_hit_probability"] = {r["attack_id"]: qi for r, qi in zip(recs, q)}
        pnull[agg] = blk
    rand["label_permutation_null_exploratory"] = {"permutations": args.perm, "seed": args.seed, "by_aggregator": pnull}
    out["b_random"] = rand
    # (c) McNemar against the constant P1 explainer
    p1 = np.array(["P1" in t for t in T])
    out["c_mcnemar_vs_constant_P1"] = {agg: mcnemar_one_sided(h, p1) for agg, h in hits.items()}
    out["c_mcnemar_vs_constant_P1"]["discordant_attacks"] = {
        agg: {"model_only": [r["attack_id"] for r, h, c in zip(recs, hits[agg], p1) if h and not c],
              "P1_only": [r["attack_id"] for r, h, c in zip(recs, hits[agg], p1) if c and not h]}
        for agg in ("sum", "mean", "max")}
    # (d) mean-normalised localisation: disagreements with the sum rule
    out["d_mean_vs_sum_disagreements"] = [{"attack_id": r["attack_id"], "targets": r["target_processes"],
                                           "top_sum": r["top_sum"], "top_mean": r["top_mean"]}
                                          for r in recs if r["top_sum"] != r["top_mean"]]
    # (e) tag level
    tag_blocks = {}
    for name, mapper in (("as_recorded", lambda t: t if t in col_set else None),
                         ("names_normalised", lambda t: colnorm.get(norm(t)))):
        tsets = [set(x for x in (mapper(t) for t in r["target_points"]) if x) for r in recs]
        obs = np.array([bool(set(r["top_tags"]) & ts) for r, ts in zip(recs, tsets)])
        q = [tag_chance(len(ts)) for ts in tsets]
        pmf = R.poisson_binomial_pmf(q)
        blk = {"observed": bound_block(int(obs.sum()), n), "chance": R.pmf_summary(pmf, n),
               "p_ge_observed": R.tail_ge(pmf, int(obs.sum())),
               "attacks_without_target_channel": [r["attack_id"] for r, ts in zip(recs, tsets) if not ts],
               "per_attack": [{"attack_id": r["attack_id"], "target_channels": sorted(ts), "hit": bool(h), "chance": qi}
                              for r, ts, h, qi in zip(recs, tsets, obs, q)]}
        blk["observed"]["lcb_minus_chance_rate"] = blk["observed"]["lcb"][str(A0)] - blk["chance"]["expected_accuracy"]
        bc = best_constant_tagset(tsets)
        blk["best_constant_5_tag_set"] = {
            "in_sample": {**bc["in_sample"], **bound_block(bc["in_sample"]["k"], n),
                          "note": "selected on the same attacks (optimistic)"},
            "leave_one_out": {**bc["leave_one_out"], "n": n,
                              "note": ("for each attack the set is re-optimised on the other attacks; the tie-averaged value "
                                       "averages over all optimal sets (random tie-break) and is the reported estimate; "
                                       "hits_first_optimum uses the lexicographically first optimum and is shown only to "
                                       "document its sensitivity to tie-breaking")}}
        const_hit = np.array([bool(set(bc["in_sample"]["tags"]) & ts) for ts in tsets])
        blk["mcnemar_vs_in_sample_constant_set"] = {"H1_model_better": mcnemar_one_sided(obs, const_hit),
                                                    "H1_constant_better": mcnemar_one_sided(const_hit, obs)}
        tag_blocks[name] = blk
    out["e_tag_level_top5"] = tag_blocks
    return out


analysis, summary = {}, {}
for tag in VARIANTS:
    recs = variants[tag]["records"]
    analysis[tag] = {"primary": analyse(recs),
                     "sensitivity_excluding_inconsistent_rows": analyse(
                         recs, exclude=[k for k in R.INCONSISTENT_ROWS if k not in ("A415", "A417")])}
    a = analysis[tag]["primary"]
    cp_ = a["b_random"]["channel_proportional"]
    tg = a["e_tag_level_top5"]["as_recorded"]
    summary[tag] = {
        "n_detected": a["n"], "n_targeting_P1": a["n_targeting"]["P1"],
        "model_sum_preregistered": [a["model"]["sum"]["k"], a["n"], a["model"]["sum"]["lcb"][str(A0)]],
        "model_mean_normalised": [a["model"]["mean"]["k"], a["n"], a["model"]["mean"]["lcb"][str(A0)]],
        "constant_P1": [a["a_constant"]["per_process"]["P1"]["k"], a["n"], a["a_constant"]["per_process"]["P1"]["lcb"][str(A0)],
                        "IE-2 pass" if a["a_constant"]["per_process"]["P1"]["ie2_pass_lcb"][str(A0)] else "IE-2 fail"],
        "channel_proportional_random": {"expected_accuracy": cp_["expected_accuracy"], "sd_hits": cp_["sd_hits"],
                                        "p_pass_lcb_0.05": cp_["p_pass_lcb_0.05"]},
        "mcnemar_sum_vs_P1": a["c_mcnemar_vs_constant_P1"]["sum"],
        "mcnemar_mean_vs_P1": a["c_mcnemar_vs_constant_P1"]["mean"],
        "tag_top5": {"observed": [tg["observed"]["k"], a["n"], tg["observed"]["lcb"][str(A0)]],
                     "chance_expected_hits": tg["chance"]["expected_hits"], "chance_rate": tg["chance"]["expected_accuracy"],
                     "p_chance_ge_observed": tg["p_ge_observed"],
                     "constant_set_in_sample": [tg["best_constant_5_tag_set"]["in_sample"]["k"], a["n"]],
                     "constant_set_leave_one_out_expected": tg["best_constant_5_tag_set"]["leave_one_out"]["expected_hits_tie_averaged"]}}

out = {"script": SCRIPT, "started_utc": started, "finished_utc": R.utc_now(), "runtime_s": time.time() - t0,
       "versions": R.versions(["torch", "safetensors"]),
       "settings": {"tau": TAU, "alpha_pre_registered": A0, "alphas": ALPHAS, "horizon_s": HORIZON, "topk": TOPK,
                    "variants": VARIANTS, "permutations": args.perm, "seed": args.seed,
                    "hit_rule": "hit if the top-ranked process is any of the attack's target processes",
                    "aggregators": {"sum": "hai_lib.localise (pre-registered E1 rule): sum of channel contributions per process",
                                    "mean": "sum divided by the process's channel count",
                                    "max": "process of the top-ranked channel (exploratory)"}},
       "channels_per_process": n_ch, "n_channels": F,
       "summary": summary,
       "input_checks": checks, "inconsistent_rows_flagged": R.INCONSISTENT_ROWS,
       "variants": {t: {"theta": variants[t]["theta"], "reproduction": variants[t]["reproduction"],
                        "analysis": analysis[t], "records": variants[t]["records"]} for t in VARIANTS}}
R.jdump(out, "e1b_localisation.json", SCRIPT)
for t in VARIANTS:
    a = analysis[t]["primary"]
    print(t, "n", a["n"], "targets", a["n_targeting"], "| model sum/mean/max",
          {g: (a["model"][g]["k"], round(a["model"][g]["lcb"][str(A0)], 3)) for g in ("sum", "mean", "max")},
          "| const P1", (a["a_constant"]["per_process"]["P1"]["k"], round(a["a_constant"]["per_process"]["P1"]["lcb"][str(A0)], 3)),
          "| McNemar p", {g: round(a["c_mcnemar_vs_constant_P1"][g]["p_one_sided_exact"], 4) for g in ("sum", "mean", "max")},
          "| tag5", a["e_tag_level_top5"]["as_recorded"]["observed"]["k"],
          round(a["e_tag_level_top5"]["as_recorded"]["chance"]["expected_hits"], 2))
print("runtime_s", round(time.time() - t0, 1))
