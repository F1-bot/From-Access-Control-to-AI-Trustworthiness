"""ATLAS coverage analysis -> data/results/atlas_coverage.json"""
import json, yaml, numpy as np
from collections import defaultdict
from aitcert.atlas import Atlas, load_map, technique_coverage, case_coverage
from aitcert.model import Catalogue
ROOT = "../.."
A = Atlas(f"{ROOT}/data/raw/atlas-data/dist/v6/ATLAS-2026.09.yaml")
M = load_map("spec/atlas_mitigation_map.yaml")
cat = Catalogue.load(True)
rows = lambda f: ["G:" + r["id"] for d in yaml.safe_load(open(f, encoding="utf8"))["dimensions"].values() for r in d["rows"]]
leg_rows, ai_rows = rows("spec/legacy_guarantees.yaml"), rows("spec/ai_guarantees.yaml")
leg_svc = [c for c, s in cat.services.items() if s.origin == "legacy"]
ai_svc = [c for c, s in cat.services.items() if s.origin == "ai"]
legacy, ext = leg_svc + leg_rows, leg_svc + leg_rows + ai_svc + ai_rows
unknown = sorted({x for v in M.values() for x in v} - set(ext))
assert not unknown, unknown
has_mit = [t for t in A.techniques if A.mitigations_of(t)]
tl, te = technique_coverage(A, M, legacy), technique_coverage(A, M, ext)
def summ(tc, ts):
    v = np.array([tc[t] for t in ts]); return {"n": len(ts), "addressed": float((v > 0).mean()), "full": float((v >= 1).mean()), "weighted": float(v.mean())}
res = {"atlas_release": "2026.09", "n_tactics": len(A.tactics), "n_techniques": len(A.techniques),
       "n_top_techniques": sum(1 for t in A.techniques if t not in A.parent), "n_subtechniques": len(A.parent),
       "n_mitigations": len(A.mitigations), "n_case_studies": len(A.case_studies), "n_mitigated_techniques": len(has_mit),
       "all": {"legacy": summ(tl, A.techniques), "extended": summ(te, A.techniques)},
       "mitigated": {"legacy": summ(tl, has_mit), "extended": summ(te, has_mit)}}
# by tactic
bt = defaultdict(list)
for t in has_mit:
    for ta in A.tactics_of(t):
        bt[ta].append(t)
res["by_tactic"] = {ta: {"name": A.tactics[ta]["name"], "legacy": summ(tl, ts), "extended": summ(te, ts)} for ta, ts in sorted(bt.items())}
# case studies
cl, ce = case_coverage(A, tl), case_coverage(A, te)
res["case_studies"] = {cs: {"name": A.case_studies[cs]["name"], "type": A.case_studies[cs].get("type"), "date": str(A.case_studies[cs].get("date")),
                            "legacy": cl[cs], "extended": ce[cs]} for cs in cl}
res["case_means"] = {k: {"legacy": float(np.mean([v[k] for v in cl.values()])), "extended": float(np.mean([v[k] for v in ce.values()]))}
                     for k in ("weighted", "addressed", "full")}
# mitigation categories: does each mitigation map (F) to legacy / AI targets?
cats = defaultdict(lambda: {"n": 0, "legacy_F": 0, "legacy_any": 0, "ai_F": 0, "ai_any": 0, "unmapped": 0})
for mid, m in A.mitigations.items():
    for c in m.get("categories", []):
        r = cats[c]; r["n"] += 1
        mp = M.get(mid, {})
        lf = any(w >= 1 for x, w in mp.items() if x in legacy); la = any(x in legacy for x in mp)
        af = any(w >= 1 for x, w in mp.items() if x in ai_svc + ai_rows); aa = any(x in ai_svc + ai_rows for x in mp)
        r["legacy_F"] += lf; r["legacy_any"] += la; r["ai_F"] += af; r["ai_any"] += aa; r["unmapped"] += (not mp)
res["mitigation_categories"] = dict(cats)
# marginal contribution of each AI service (drop in weighted coverage of mitigated techniques when removed)
base = summ(te, has_mit)["weighted"]
res["ai_marginal"] = {}
for s in ai_svc + ["AI-rows"]:
    X = [x for x in ext if x != s] if s != "AI-rows" else [x for x in ext if x not in ai_rows]
    res["ai_marginal"][s] = base - summ(technique_coverage(A, M, X), has_mit)["weighted"]
res["technique_coverage"] = {t: {"name": A.techniques[t]["name"], "legacy": tl[t], "extended": te[t], "tactics": sorted(A.tactics_of(t)),
                                 "mitigations": sorted(A.mitigations_of(t))} for t in A.techniques}
json.dump(res, open(f"{ROOT}/data/results/atlas_coverage.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("all", "mitigated", "case_means", "mitigation_categories", "ai_marginal")}, indent=1))
