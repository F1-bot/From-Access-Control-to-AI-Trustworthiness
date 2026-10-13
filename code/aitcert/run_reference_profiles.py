"""Reference AI-extended profiles for representative high-risk domains. Each profile = a legacy base profile
from ND TZI 2.5-005-99 + required class-I levels + assurance level; the closure under necessary
conditions is computed and validated. Output: data/results/reference_profiles.json"""
import json
from aitcert.model import Catalogue, profile_id

ROOT = "../.."
cat = Catalogue.load(True)
base = {p["id"]: p["services"] for p in json.load(open(f"{ROOT}/data/processed/nd_tzi_2.5-005-99_profiles.json", encoding="utf8"))["profiles"]}
DOMAINS = [
    {"id": "3.CDI.ICS", "domain": "Attack/anomaly detector for an industrial control system (critical infrastructure; Annex III(2) when used as a safety component)",
     "base": "3.CD.3", "ai": {"ID": 3, "IR": 2, "IV": 3, "IE": 2, "IP": 2, "IO": 2, "IM": 2}, "G": 4},
    {"id": "3.KCDI.MED", "domain": "Medical-imaging triage (high-risk via Annex I, Regulation (EU) 2017/745)",
     "base": "3.KCD.3", "ai": {"ID": 3, "IR": 2, "IV": 4, "IE": 2, "IB": 2, "IP": 2, "IO": 2, "IM": 2}, "G": 5},
    {"id": "3.KCDI.CRS", "domain": "Creditworthiness scoring of natural persons (Annex III(5)(b))",
     "base": "3.KCD.3", "ai": {"ID": 2, "IR": 1, "IV": 2, "IE": 2, "IB": 3, "IP": 2, "IO": 1, "IM": 2}, "G": 4},
    {"id": "3.KCDI.GOV", "domain": "LLM-based agent for access to public services (Annex III(5)(a); Art. 50 transparency)",
     "base": "3.KCD.2", "ai": {"ID": 2, "IR": 3, "IV": 4, "IE": 1, "IB": 2, "IP": 2, "IO": 2, "IM": 2, "IA": 2}, "G": 4},
    {"id": "3.KCDI.RBI", "domain": "Remote biometric identification (Annex III(1)(a); Art. 14(5) dual verification)",
     "base": "3.KCD.4", "ai": {"ID": 3, "IR": 3, "IV": 3, "IE": 2, "IB": 3, "IP": 3, "IO": 3, "IM": 3}, "G": 5},
]
out = []
for d in DOMAINS:
    p0 = {**base[d["base"]], **d["ai"]}
    cl, g = cat.closure(p0, d["G"])
    added = {k: v for k, v in cl.items() if p0.get(k, 0) < v}
    viol = cat.violations(cl, g)
    out.append({**d, "closure": cl, "G_closure": g, "added_by_closure": added, "valid": not viol, "violations": viol,
                "ai_formatted": cat.fmt({k: v for k, v in cl.items() if cat.services[k].origin == "ai"}),
                "n_services": len(cl), "dominates_base": cat.dominates(base[d["base"]], cl)})
    print(d["id"], "valid", not viol, "G", g, "added", added, "|", cat.fmt({k: v for k, v in cl.items() if cat.services[k].origin == "ai"}))
# dominance relations among reference profiles (partial order)
for a in out:
    for b in out:
        if a is not b and cat.dominates(a["closure"], b["closure"]):
            print(f"{a['id']} <= {b['id']}")
json.dump(out, open(f"{ROOT}/data/results/reference_profiles.json", "w"), indent=1)
