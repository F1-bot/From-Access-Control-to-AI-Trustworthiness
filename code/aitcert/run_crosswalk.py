"""Crosswalk coverage analysis -> data/results/crosswalk_coverage.json + supplementary CSV/XLSX."""
import json, yaml, os
import pandas as pd
from aitcert.model import Catalogue
from aitcert.crosswalk import coverage, unit_score
import importlib.util
spec = importlib.util.spec_from_file_location("cw", "spec/crosswalk_data.py"); cw = importlib.util.module_from_spec(spec); spec.loader.exec_module(cw)
ROOT = "../.."
cat = Catalogue.load(True)
rows = lambda f: ["G:" + r["id"] for d in yaml.safe_load(open(f, encoding="utf8"))["dimensions"].values() for r in d["rows"]]
leg = [c for c, s in cat.services.items() if s.origin == "legacy"] + rows("spec/legacy_guarantees.yaml") + ["X:PROFILE"]
ai = [c for c, s in cat.services.items() if s.origin == "ai"] + rows("spec/ai_guarantees.yaml") + ["X:AIPROFILE"]
ext = leg + ai
# --- validation of identifiers against the official inventories
rmf = [s["id"] for s in json.load(open(f"{ROOT}/data/processed/nist_ai_rmf_subcategories.json", encoding="utf8"))["subcategories"]]
assert sorted(cw.NIST_AI_RMF) == sorted(rmf), set(rmf) ^ set(cw.NIST_AI_RMF)
csf = [s["id"] for s in json.load(open(f"{ROOT}/data/processed/nist_csf2.json", encoding="utf8"))["subcategories"]]
assert set(cw.CSF2) <= set(csf)
act_ids = {r["id"] for r in json.load(open(f"{ROOT}/data/processed/aiact_paragraphs.json", encoding="utf8"))["requirements"]}
assert set(cw.EU_AI_ACT) <= act_ids, set(cw.EU_AI_ACT) - act_ids
assert len(cw.ISO_42001_ANNEX_A) == 38 and len(cw.OWASP_LLM_2025) == 10
targets = {t for d in (cw.NIST_AI_RMF, cw.EU_AI_ACT, cw.ISO_42001_ANNEX_A, cw.OWASP_LLM_2025, cw.CSF2) for v in d.values() for t in (cw.parse(v) or {})}
assert targets <= set(ext), targets - set(ext)
prio = {r["id"]: r for r in json.load(open(f"{ROOT}/data/processed/nist_ir8596_priorities.json"))["subcategories"]}
fw = {"NIST AI RMF 1.0": {k: cw.parse(v) for k, v in cw.NIST_AI_RMF.items()},
      "EU AI Act Arts. 9-15": {k: cw.parse(v) for k, v in cw.EU_AI_ACT.items()},
      "ISO/IEC 42001 Annex A": {k: cw.parse(v) for k, v in cw.ISO_42001_ANNEX_A.items()},
      "OWASP LLM Top 10 (2025)": {k: cw.parse(v) for k, v in cw.OWASP_LLM_2025.items()},
      "NIST CSF 2.0": {k: cw.parse(cw.CSF2.get(k, "")) for k in csf}}
wts = {k: {1: 3.0, 2: 2.0, 3: 1.0}[prio[k]["secure"]] for k in csf}
high = {k: fw["NIST CSF 2.0"][k] for k in csf if prio[k]["secure"] == 1}
fw["NIST CSF 2.0 (Cyber AI Profile, Secure: High)"] = high
res = {}
for name, mp in fw.items():
    r = {}
    for label, X in (("legacy", leg), ("ai_only", ai), ("extended", ext)):
        r[label] = {f"w{w}": coverage(mp, X, w)["weighted"] for w in (0.25, 0.5, 0.75)}
        r[label]["strict"] = coverage(mp, X, strict=True)["weighted"]
        r[label]["addressed"] = coverage(mp, X)["addressed"]
        r[label]["n"] = coverage(mp, X)["n"]
    if name == "NIST CSF 2.0":
        r["priority_weighted"] = {lab: coverage(mp, X, 0.5, weights=wts)["weighted"] for lab, X in (("legacy", leg), ("extended", ext))}
    res[name] = r
# by AI RMF function
by_fn = {}
for fn in ("GOVERN", "MAP", "MEASURE", "MANAGE"):
    sub = {k: v for k, v in fw["NIST AI RMF 1.0"].items() if k.startswith(fn)}
    by_fn[fn] = {lab: coverage(sub, X)["weighted"] for lab, X in (("legacy", leg), ("extended", ext))} | {"n": len(sub)}
res["NIST AI RMF by function"] = by_fn
by_art = {}
for a in ("9", "10", "11", "12", "13", "14", "15"):
    sub = {k: v for k, v in fw["EU AI Act Arts. 9-15"].items() if k.startswith(f"Art.{a}(")}
    by_art[f"Art. {a}"] = {lab: coverage(sub, X)["weighted"] for lab, X in (("legacy", leg), ("extended", ext))} | {"n": coverage(sub, ext)["n"]}
res["EU AI Act by article"] = by_art
os.makedirs(f"{ROOT}/data/results", exist_ok=True)
json.dump(res, open(f"{ROOT}/data/results/crosswalk_coverage.json", "w"), indent=1)
# supplementary long table
recs = []
for name, mp in fw.items():
    if "Secure: High" in name:
        continue
    for k, m in mp.items():
        recs.append({"framework": name, "requirement": k, "mapping": "NA" if m is None else " ".join(f"{t}:{'F' if s >= 1 else 'P'}" for t, s in m.items()),
                     "legacy_score": unit_score(m, set(leg)), "extended_score": unit_score(m, set(ext))})
df = pd.DataFrame(recs)
os.makedirs(f"{ROOT}/supplementary", exist_ok=True)
df.to_csv(f"{ROOT}/supplementary/S2_crosswalk_mapping.csv", index=False)
for name, r in res.items():
    if "legacy" in r:
        print(f"{name:48s} n={r['legacy']['n']:3d}  legacy w0.5={r['legacy']['w0.5']:.3f} strict={r['legacy']['strict']:.3f} | ext w0.5={r['extended']['w0.5']:.3f} strict={r['extended']['strict']:.3f} [{r['extended']['w0.25']:.3f}-{r['extended']['w0.75']:.3f}]")
print(json.dumps({k: res[k] for k in ("NIST AI RMF by function", "EU AI Act by article")}, indent=0))
print("CSF priority-weighted:", res["NIST CSF 2.0"]["priority_weighted"])
