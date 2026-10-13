"""Extract proposed priorities (1=High, 2=Moderate, 3=Foundational) per CSF 2.0 subcategory and
focus area (Secure, Defend, Thwart) from NIST IR 8596 iprd (Cyber AI Profile, Dec 2025)."""
import json, re, sys
ROOT = sys.argv[1]
t = open(f"{ROOT}/data/raw/frameworks/NIST.IR.8596.txt", encoding="utf8").read()
t = re.sub(r"\n\d+ \n", "\n", t)  # line numbers
start = t.find("2.3. Cyber AI Profile: GOVERN")
body = t[start:t.find("References", t.find("Cyber AI Profile – RECOVER", start))] if start > 0 else t
parts = re.split(r"\n((?:GV|ID|PR|DE|RS|RC)\.[A-Z]{2}-\d{2}): ", body)
csf = json.load(open(f"{ROOT}/data/processed/nist_csf2.json", encoding="utf8"))
valid = {s["id"] for s in csf["subcategories"]}
rows, anomalies = {}, []
for k in range(1, len(parts), 2):
    sid, blk = parts[k], parts[k + 1]
    pr = re.findall(r"Proposed Priority:\s*(\d)", blk)
    if sid in rows:  # continuation of a row split across pages
        rows[sid]["raw"].extend(pr)
    else:
        rows[sid] = {"raw": pr}
out = []
for sid, r in rows.items():
    pr = r["raw"]
    rec = {"id": sid, "in_csf2": sid in valid}
    if len(pr) >= 3:
        rec.update({"secure": int(pr[0]), "defend": int(pr[1]), "thwart": int(pr[2])})
    else:
        anomalies.append((sid, pr))
    out.append(rec)
json.dump({"source": "NIST IR 8596 iprd (Dec 2025), Tables 1-6", "doi": "10.6028/NIST.IR.8596.iprd",
           "priority_codes": {"1": "High", "2": "Moderate", "3": "Foundational"}, "subcategories": out,
           "parse_anomalies": anomalies}, open(f"{ROOT}/data/processed/nist_ir8596_priorities.json", "w"), indent=1)
from collections import Counter
print("rows", len(out), "valid", sum(r["in_csf2"] for r in out), "anomalies", len(anomalies), anomalies[:10])
print("Secure priorities", Counter(r.get("secure") for r in out))
print("Defend", Counter(r.get("defend") for r in out), "Thwart", Counter(r.get("thwart") for r in out))
missing = valid - set(rows)
print("CSF subcategories missing from profile:", sorted(missing))
