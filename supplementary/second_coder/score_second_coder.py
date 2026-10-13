"""Agreement between an independent second coding and the authors' mapping (crosswalk units and ATLAS
mitigations sampled by code/aitcert/make_second_coder.py).

Usage
    python score_second_coder.py --sheet to_coder/coding_sheet_XY.csv            # score a completed sheet
    python score_second_coder.py --sheet ... --out agreement_XY.json --disagreements disagreements_XY.csv
    python score_second_coder.py --selftest                                       # verify the statistics

The authors' mapping is read at run time from code/aitcert/spec/crosswalk_data.py and
code/aitcert/spec/atlas_mitigation_map.yaml (project root = two levels above this file, or --root). If it
no longer matches the digest recorded in sample_manifest.json when the sheet was prepared, agreement is also
computed against the frozen copy data/results/second_coder_reference_key.json.

Statistics (grades coded 0 = none, 1 = partial P, 2 = full F)
 * pair level: every (unit, candidate target) pair is an item rated by both coders. Cohen's kappa
   (unweighted, linear and quadratic weights), Krippendorff's alpha (nominal, ordinal, interval), observed
   agreement, the 3x3 confusion matrix, and positive specific agreement on which targets are credited at
   all (PSA = 2a / (2a + b + c), a = pairs credited by both, b and c = pairs credited by one coder only).
   Most pairs are 'none' for both coders, so kappa and alpha at this level must be read together with PSA.
 * unit level: the coverage score of each unit, max over the targets in X of the grade, for X = X_ext (all
   targets) and X = X_leg (legacy services, legacy assurance rows, X:PROFILE). This is the quantity that enters
   the coverage metrics of the article.
 * coverage: framework coverage on the sampled units from each coding (w = 0.5 and strict) and the difference.
 * 95% confidence intervals: percentile bootstrap over units, stratified by framework (B = 2000, seed 0).
Only numpy (and PyYAML for the ATLAS map) are required.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import sys
import time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS_FW = "MITRE ATLAS mitigations"
CW_ATTR = OrderedDict([("NIST AI RMF 1.0", "NIST_AI_RMF"), ("EU AI Act Arts. 9-15", "EU_AI_ACT"),
                       ("ISO/IEC 42001 Annex A", "ISO_42001_ANNEX_A"), ("OWASP LLM Top 10 (2025)", "OWASP_LLM_2025"),
                       ("NIST CSF 2.0", "CSF2")])
CODE = {"P": 1, "F": 2}
TOKEN_RE = re.compile(r"^(.+):([FP])$", re.I)


# ============================================================================ statistics
def confusion(a, b, k=3):
    C = np.zeros((k, k))
    np.add.at(C, (np.asarray(a, int), np.asarray(b, int)), 1.0)
    return C


def kappa_from_confusion(C, weights="linear"):
    """Cohen's kappa from a k x k confusion matrix; weights None, 'linear' or 'quadratic'."""
    k = C.shape[0]
    n = C.sum()
    if n == 0:
        return float("nan")
    O = C / n
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    i, j = np.indices((k, k))
    Wd = (i != j).astype(float) if weights is None else (np.abs(i - j) / (k - 1) if weights == "linear" else ((i - j) / (k - 1)) ** 2)
    de = (Wd * E).sum()
    return float("nan") if de == 0 else float(1.0 - (Wd * O).sum() / de)


def _delta2(level, vals, nc):
    K = len(vals)
    i, j = np.indices((K, K))
    if level == "nominal":
        return (i != j).astype(float)
    if level == "interval":
        return (vals[i] - vals[j]) ** 2
    if level == "ratio":
        s = vals[i] + vals[j]
        return np.where(s == 0, 0.0, ((vals[i] - vals[j]) / np.where(s == 0, 1, s)) ** 2)
    if level == "ordinal":
        cs = np.concatenate([[0.0], np.cumsum(nc)])
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        return (cs[hi + 1] - cs[lo] - (nc[i] + nc[j]) / 2.0) ** 2
    raise ValueError(level)


def alpha_from_coincidence(o, vals, level):
    nc = o.sum(axis=1)
    n = nc.sum()
    keep = nc > 0
    if keep.sum() < 2:
        return float("nan")
    o, vals, nc = o[np.ix_(keep, keep)], vals[keep], nc[keep]
    d2 = _delta2(level, vals, nc)
    De = (np.outer(nc, nc) * d2).sum()
    return float("nan") if De == 0 else float(1.0 - (n - 1) * (o * d2).sum() / De)


def alpha_two_coders(C, level="ordinal", vals=(0.0, 1.0, 2.0)):
    """Krippendorff's alpha for two coders without missing values: the coincidence matrix is C + C^T."""
    return alpha_from_coincidence(C + C.T, np.asarray(vals, float), level)


def krippendorff_alpha(data, level="ordinal"):
    """General Krippendorff's alpha for a coders x units matrix with np.nan for missing values."""
    data = np.asarray(data, dtype=float)
    vals = np.unique(data[~np.isnan(data)])
    vix = {v: i for i, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)))
    for u in range(data.shape[1]):
        col = data[:, u][~np.isnan(data[:, u])]
        m = len(col)
        if m < 2:
            continue
        for x in range(m):
            for y in range(m):
                if x != y:
                    o[vix[col[x]], vix[col[y]]] += 1.0 / (m - 1)
    return alpha_from_coincidence(o, vals, level)


def stats_from_confusion(C):
    n = C.sum()
    a = C[1:, 1:].sum()
    b = C[1:, 0].sum()
    c = C[0, 1:].sum()
    both = C[1:, 1:]
    return OrderedDict(
        n_items=int(n), observed_agreement=float(np.trace(C) / n) if n else float("nan"),
        kappa_unweighted=kappa_from_confusion(C, None), kappa_linear=kappa_from_confusion(C, "linear"),
        kappa_quadratic=kappa_from_confusion(C, "quadratic"),
        alpha_nominal=alpha_two_coders(C, "nominal"), alpha_ordinal=alpha_two_coders(C, "ordinal"),
        alpha_interval=alpha_two_coders(C, "interval"),
        positive_specific_agreement=float(2 * a / (2 * a + b + c)) if (2 * a + b + c) else float("nan"),
        grade_agreement_when_both_credit=float(np.trace(both) / both.sum()) if both.sum() else float("nan"),
        confusion_authors_rows_coder_cols=C.astype(int).tolist())


BOOT_KEYS = ("kappa_linear", "kappa_quadratic", "alpha_ordinal", "alpha_interval", "positive_specific_agreement")


def boot_ci(per_unit_C, strata, B, seed):
    """Percentile 95% CI by resampling units with replacement within strata (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    idx_by = OrderedDict()
    for i, s in enumerate(strata):
        idx_by.setdefault(s, []).append(i)
    reps = {k: [] for k in BOOT_KEYS}
    for _ in range(B):
        C = np.zeros((3, 3))
        for s, ids in idx_by.items():
            pick = rng.choice(ids, size=len(ids), replace=True)
            C += per_unit_C[pick].sum(axis=0)
        st = stats_from_confusion(C)
        for k in BOOT_KEYS:
            reps[k].append(st[k])
    return OrderedDict((k, [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]) for k, v in reps.items())


# ============================================================================ inputs
def read_sheet(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1251", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    need = {"id", "framework", "coding", "coded"}
    # The delimiter is chosen from the header line (Excel writes ';' in some locales); csv.Sniffer is not
    # reliable here because the cells themselves contain many separators.
    for delim in (",", ";", "\t"):
        rd = csv.DictReader(io.StringIO(text), delimiter=delim)
        if rd.fieldnames and need <= {f.strip() for f in rd.fieldnames}:
            rows = [{(k or "").strip(): v for k, v in r.items()} for r in rd]
            bad = [i + 2 for i, r in enumerate(rows) if None in r or any(v is None for v in r.values())]
            if bad:
                raise SystemExit(f"{path}: rows {bad[:10]} have a different number of fields than the header")
            return rows
    raise SystemExit(f"{path}: the sheet must contain the columns {sorted(need)}")


def parse_coding(s, allowed, lenient, where):
    out, bad = OrderedDict(), []
    for tok in re.split(r"[\s,;]+", (s or "").strip()):
        if not tok:
            continue
        m = TOKEN_RE.match(tok)
        if not m:
            bad.append(tok)
            continue
        tgt, g = m.group(1).upper(), m.group(2).upper()
        if tgt not in allowed:
            bad.append(tok)
            continue
        out[tgt] = max(out.get(tgt, 0), CODE[g])
    if bad and not lenient:
        raise SystemExit(f"{where}: unknown or malformed tokens {bad}; fix the sheet or use --lenient")
    return out, bad


def load_reference(root, manifest):
    spec = os.path.join(root, "code", "aitcert", "spec")
    m = importlib.util.spec_from_file_location("cw_spec_score", os.path.join(spec, "crosswalk_data.py"))
    cw = importlib.util.module_from_spec(m)
    m.loader.exec_module(cw)
    import yaml
    amap = yaml.safe_load(open(os.path.join(spec, "atlas_mitigation_map.yaml"), encoding="utf8"))["mitigations"]
    ref = OrderedDict()
    for stratum, s in manifest["strata"].items():
        for u in s["sampled"]:
            if stratum == ATLAS_FW:
                mp = {str(x): str(g) for x, g in ((amap.get(u) or {}).get("map") or {}).items()}
            else:
                txt = getattr(cw, CW_ATTR[stratum]).get(u, "")
                if txt.strip() == "NA":
                    mp = None
                else:
                    mp = OrderedDict()
                    for tok in txt.split():
                        t, g = tok.rsplit(":", 1)
                        mp[t] = g
            ref[f"{stratum}|{u}"] = mp
    return ref


def digest(ref):
    s = json.dumps({k: sorted(f"{t}:{g}" for t, g in (v or {}).items()) for k, v in sorted(ref.items())}, sort_keys=True)
    return hashlib.sha256(s.encode("utf8")).hexdigest()


# ============================================================================ scoring
def score(rows_by_key, ref, manifest, B, seed):
    groups = manifest["target_groups"]
    legacy = {x for x, g in groups.items() if g in manifest["legacy_groups"]}
    lut = {"w0_5": np.array([0.0, 0.5, 1.0]), "strict": np.array([0.0, 0.0, 1.0])}
    per_unit_pairs, unit_rows, strata = [], [], []
    for key, coder in rows_by_key.items():
        stratum, uid = key.split("|", 1)
        cand = manifest["candidate_targets"]["atlas" if stratum == ATLAS_FW else "crosswalk"]
        r = {t: CODE[g] for t, g in (ref[key] or {}).items()}
        targets = list(dict.fromkeys(list(cand) + list(r) + list(coder)))
        a = np.array([r.get(t, 0) for t in targets])
        b = np.array([coder.get(t, 0) for t in targets])
        per_unit_pairs.append(confusion(a, b))
        best = lambda d, X=None: max([v for t, v in d.items() if X is None or t in X], default=0)  # noqa: E731
        unit_rows.append(dict(key=key, stratum=stratum, ref_ext=best(r), cod_ext=best(coder),
                              ref_leg=best(r, legacy), cod_leg=best(coder, legacy)))
        strata.append(stratum)
    P = np.array(per_unit_pairs)
    strata = np.array(strata)
    out = OrderedDict()
    out["pair_level"] = OrderedDict(all=stats_from_confusion(P.sum(axis=0)))
    out["pair_level"]["all"]["ci95_bootstrap"] = boot_ci(P, strata, B, seed)
    for lab, sel in (("crosswalk_units", strata != ATLAS_FW), ("atlas_mitigations", strata == ATLAS_FW)):
        if sel.any():
            out["pair_level"][lab] = stats_from_confusion(P[sel].sum(axis=0))
            out["pair_level"][lab]["ci95_bootstrap"] = boot_ci(P[sel], strata[sel], B, seed)
    out["pair_level"]["by_stratum"] = OrderedDict((s, stats_from_confusion(P[strata == s].sum(axis=0)))
                                                  for s in dict.fromkeys(strata))
    out["unit_level"] = OrderedDict()
    for X in ("ext", "leg"):
        U = np.array([confusion([u[f"ref_{X}"]], [u[f"cod_{X}"]]) for u in unit_rows])
        blk = OrderedDict(all=stats_from_confusion(U.sum(axis=0)))
        blk["all"]["ci95_bootstrap"] = boot_ci(U, strata, B, seed)
        for lab, sel in (("crosswalk_units", strata != ATLAS_FW), ("atlas_mitigations", strata == ATLAS_FW)):
            if sel.any():
                blk[lab] = stats_from_confusion(U[sel].sum(axis=0))
        blk["by_stratum"] = OrderedDict((s, stats_from_confusion(U[strata == s].sum(axis=0))) for s in dict.fromkeys(strata))
        out["unit_level"]["X_" + X] = blk
    # coverage on the sample
    cov = OrderedDict()
    rng = np.random.default_rng(seed)
    for s in dict.fromkeys(strata):
        us = [u for u in unit_rows if u["stratum"] == s]
        blk = OrderedDict(n_units=len(us))
        for X in ("leg", "ext"):
            for w, L in lut.items():
                ra = np.array([L[u[f"ref_{X}"]] for u in us])
                ca = np.array([L[u[f"cod_{X}"]] for u in us])
                d = ca - ra
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
                blk[f"X_{X}|{w}"] = OrderedDict(authors=float(ra.mean()), coder=float(ca.mean()), difference=float(d.mean()),
                                                difference_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        cov[s] = blk
    out["coverage_on_sample"] = cov
    return out, unit_rows


def disagreements(rows_by_key, ref):
    out = []
    for key, coder in rows_by_key.items():
        r = {t: CODE[g] for t, g in (ref[key] or {}).items()}
        name = {0: "none", 1: "P", 2: "F"}
        for t in dict.fromkeys(list(r) + list(coder)):
            if r.get(t, 0) != coder.get(t, 0):
                out.append({"stratum": key.split("|", 1)[0], "id": key.split("|", 1)[1], "target": t,
                            "authors": name[r.get(t, 0)], "coder": name[coder.get(t, 0)]})
    return out


# ============================================================================ self-test
def selftest():
    nan = np.nan
    K11 = np.array([[1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan], [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
                    [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan], [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan]])
    exp = {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797}       # Krippendorff (2011)
    for lv, e in exp.items():
        got = krippendorff_alpha(K11, lv)
        assert abs(got - e) < 5e-4, (lv, got, e)
    WK = np.array([[nan, nan, nan, nan, nan, 3, 4, 1, 2, 1, 1, 3, 3, nan, 3], [1, nan, 2, 1, 3, 3, 4, 3, nan, nan, nan, nan, nan, nan, nan],
                   [nan, nan, 2, 1, 3, 4, 4, nan, 2, 1, 1, 3, 3, nan, 4]])
    assert abs(krippendorff_alpha(WK, "nominal") - 0.691) < 5e-4 and abs(krippendorff_alpha(WK, "interval") - 0.811) < 5e-4
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 300))
        a = rng.integers(0, 3, n)
        b = np.where(rng.random(n) < 0.7, a, rng.integers(0, 3, n))
        C = confusion(a, b)
        for lv in ("nominal", "ordinal", "interval"):
            g1, g2 = alpha_two_coders(C, lv), krippendorff_alpha(np.vstack([a, b]).astype(float), lv)
            assert (np.isnan(g1) and np.isnan(g2)) or abs(g1 - g2) < 1e-10, (lv, g1, g2)
        try:
            from sklearn.metrics import cohen_kappa_score
            for w in (None, "linear", "quadratic"):
                assert abs(kappa_from_confusion(C, w) - cohen_kappa_score(a, b, weights=w, labels=[0, 1, 2])) < 1e-10
        except ImportError:
            pass
    C = confusion([0, 1, 2, 2], [0, 1, 2, 2])
    assert abs(kappa_from_confusion(C, "linear") - 1) < 1e-12 and abs(alpha_two_coders(C) - 1) < 1e-12
    print("selftest passed: Krippendorff (2011) example 0.743/0.815/0.849/0.797, Wikipedia example 0.691/0.811, "
          "two-coder alpha = general alpha, kappa = scikit-learn (if installed)")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet")
    ap.add_argument("--root", default=os.path.abspath(os.path.join(HERE, "..", "..")))
    ap.add_argument("--manifest", default=os.path.join(HERE, "sample_manifest.json"))
    ap.add_argument("--out")
    ap.add_argument("--disagreements")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lenient", action="store_true", help="ignore unknown target codes instead of stopping")
    ap.add_argument("--allow-incomplete", action="store_true", help="score coded rows only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.sheet:
        ap.error("--sheet is required (or --selftest)")
    t0 = time.time()
    manifest = json.load(open(a.manifest, encoding="utf8"))
    rows = read_sheet(a.sheet)
    ref_now = load_reference(a.root, manifest)
    refs = OrderedDict(current_specification=ref_now)
    same = digest(ref_now) == manifest["reference_mapping_sha256"]
    if not same:
        kp = os.path.join(a.root, manifest["files"]["reference_key"])
        if os.path.exists(kp):
            k = json.load(open(kp, encoding="utf8"))["mapping"]
            refs["frozen_at_preparation"] = OrderedDict(
                (key, None if v.strip() == "NA" else OrderedDict(tok.rsplit(":", 1) for tok in v.split())) for key, v in k.items())
    expected = [f"{s}|{u}" for s, v in manifest["strata"].items() for u in v["sampled"]]
    coded, missing, na_flags, bad_all, seen = OrderedDict(), [], [], [], set()
    for i, r in enumerate(rows, start=2):
        fw = r["framework"].strip()
        stratum = ATLAS_FW if fw.startswith(ATLAS_FW) else fw
        key = f"{stratum}|{r['id'].strip()}"
        if key not in ref_now:
            raise SystemExit(f"row {i}: unit {key!r} is not in the sample manifest")
        seen.add(key)
        if str(r.get("coded", "")).strip().lower() not in ("1", "x", "yes", "y", "true"):
            missing.append(key)
            continue
        allowed = set(manifest["candidate_targets"]["atlas" if stratum == ATLAS_FW else "crosswalk"])
        if str(r.get("coder_flag", "")).strip().upper() == "NA":
            na_flags.append(key)
            coded[key] = OrderedDict()
            continue
        cd, bad = parse_coding(r.get("coding", ""), allowed, a.lenient, f"row {i} ({key})")
        bad_all += [f"{key}: {b}" for b in bad]
        coded[key] = cd
    missing += [k for k in expected if k not in seen]
    if missing and not a.allow_incomplete:
        raise SystemExit(f"{len(missing)} units are not marked as coded (column 'coded' = 1), e.g. {missing[:5]}; "
                         f"use --allow-incomplete to score the coded rows only")
    res = OrderedDict(
        meta=OrderedDict(script=os.path.basename(__file__), generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                         python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
                         sheet=os.path.abspath(a.sheet), sheet_sha256=hashlib.sha256(open(a.sheet, "rb").read()).hexdigest(),
                         bootstrap=a.bootstrap, seed=a.seed),
        completeness=OrderedDict(units_expected=len(expected), units_scored=len(coded), not_coded=missing,
                                 coder_flagged_NA=na_flags, ignored_tokens=bad_all),
        reference=OrderedDict(digest_at_preparation=manifest["reference_mapping_sha256"], digest_current=digest(ref_now),
                              current_equals_prepared=same))
    for lab, ref in refs.items():
        sc, units = score(coded, ref, manifest, a.bootstrap, a.seed)
        res[f"agreement_vs_{lab}"] = sc
    res["meta"]["runtime_s"] = round(time.time() - t0, 2)
    out = a.out or os.path.splitext(os.path.abspath(a.sheet))[0] + "_agreement.json"
    json.dump(res, open(out, "w", encoding="utf8"), indent=1)
    if a.disagreements:
        dis = disagreements(coded, refs["current_specification"])
        with open(a.disagreements, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stratum", "id", "target", "authors", "coder"])
            w.writeheader()
            w.writerows(dis)
    sc = res["agreement_vs_current_specification"]
    p, u = sc["pair_level"]["all"], sc["unit_level"]["X_ext"]["all"]
    print(f"units scored {len(coded)} of {len(expected)}; reference unchanged since preparation: {same}")
    print(f"pair level (n={p['n_items']}): kappa_lin={p['kappa_linear']:.3f} {p['ci95_bootstrap']['kappa_linear']}, "
          f"kappa_quad={p['kappa_quadratic']:.3f}, alpha_ord={p['alpha_ordinal']:.3f} {p['ci95_bootstrap']['alpha_ordinal']}, "
          f"PSA={p['positive_specific_agreement']:.3f}")
    print(f"unit level X_ext (n={u['n_items']}): kappa_lin={u['kappa_linear']:.3f} {u['ci95_bootstrap']['kappa_linear']}, "
          f"alpha_ord={u['alpha_ordinal']:.3f} {u['ci95_bootstrap']['alpha_ordinal']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
