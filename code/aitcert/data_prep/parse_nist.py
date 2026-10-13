"""Extract NIST AI RMF 1.0 subcategories (from NIST AI 100-1 PDF text) and NIST CSF 2.0
categories/subcategories (from the CPRT JSON export) into processed JSON files."""
import json, re, sys
ROOT = sys.argv[1]

# ---- AI RMF ----
txt = open(f"{ROOT}/data/raw/frameworks/NIST.AI.100-1.txt", encoding="utf8").read()
# remove running headers/footers and table furniture
txt = re.sub(r"Page \d+\nNIST AI 100-1\nAI RMF 1\.0\n", "\n", txt)
txt = re.sub(r"Table \d+: Categories and subcategories for the \w+ function\.( \(Continued\))?\n", "\n", txt)
txt = txt.replace("Categories\nSubcategories\n", "\n").replace("Continued on next page\n", "\n")
pat = re.compile(r"\b(GOVERN|MAP|MEASURE|MANAGE) (\d)\.(\d+):")
cat_pat = re.compile(r"\n(GOVERN|MAP|MEASURE|MANAGE) (\d):\s")
starts = [(m.start(), m.group(1), int(m.group(2)), int(m.group(3))) for m in pat.finditer(txt)]
seen, subs = set(), []
for k, (pos, fn, c, s) in enumerate(starts):
    if (fn, c, s) in seen:
        continue
    end = starts[k + 1][0] if k + 1 < len(starts) else pos + 600
    seg = txt[pos:end]
    cm = cat_pat.search(seg)  # cut at the next category block
    if cm:
        seg = seg[:cm.start()]
    seg = re.sub(r"-\n", "", seg)
    seg = re.sub(r"\s+", " ", seg).strip()
    seg = re.sub(r"^\w+ \d\.\d+:\s*", "", seg)
    seg = re.split(r"\s\d\.\d (?:Map|Measure|Manage|Govern)", seg)[0].strip()
    seg = re.split(r"(?<=\.)\s+(?=[A-Z][a-z]+ [a-z]+ [a-z]+ .*?(are|is) in place|Appendix|\d+ )", seg)[0] if len(seg) > 400 else seg
    seen.add((fn, c, s))
    subs.append({"id": f"{fn} {c}.{s}", "function": fn, "category": f"{fn} {c}", "text": seg})
order = {"GOVERN": 0, "MAP": 1, "MEASURE": 2, "MANAGE": 3}
subs.sort(key=lambda r: (order[r["function"]], int(r["id"].split()[1].split(".")[0]), int(r["id"].split(".")[1])))
json.dump({"source": "NIST AI 100-1 (AI RMF 1.0), Tables 1-4", "doi": "10.6028/NIST.AI.100-1", "subcategories": subs},
          open(f"{ROOT}/data/processed/nist_ai_rmf_subcategories.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("AI RMF subcategories:", len(subs))

# ---- CSF 2.0 ----
d = json.load(open(f"{ROOT}/data/raw/frameworks/csf2_cprt.json", encoding="utf8"))
els = d["response"]["elements"]["elements"]
funcs = [e for e in els if e.get("element_type") == "function"]
cats = [e for e in els if e.get("element_type") == "category"]
subc = [e for e in els if e.get("element_type") == "subcategory"]
out = {"source": "NIST CSF 2.0 (CSWP 29), CPRT JSON export", "doi": "10.6028/NIST.CSWP.29",
       "functions": [{"id": e["element_identifier"], "title": e.get("title"), "text": e.get("text")} for e in funcs],
       "categories": [{"id": e["element_identifier"], "title": e.get("title"), "text": e.get("text")} for e in cats if (e.get("text") or "").strip()],
       "subcategories": [{"id": e["element_identifier"], "text": e.get("text")} for e in subc if (e.get("text") or "").strip()]}
json.dump(out, open(f"{ROOT}/data/processed/nist_csf2.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("CSF 2.0:", len(out["functions"]), "functions,", len(out["categories"]), "categories,", len(out["subcategories"]), "subcategories")

# ---- post-fixes (robust cleaning) ----
p = f"{ROOT}/data/processed/nist_ai_rmf_subcategories.json"
d = json.load(open(p, encoding="utf8"))
for s in d["subcategories"]:
    s["text"] = re.split(r"\s\d\.\d (?:Map|Measure|Manage|Govern)\s", s["text"])[0].strip()
    s["text"] = re.split(r"\s(?:Appendix|Table \d)", s["text"])[0].strip()
json.dump(d, open(p, "w", encoding="utf8"), ensure_ascii=False, indent=1)
p = f"{ROOT}/data/processed/nist_csf2.json"
d = json.load(open(p, encoding="utf8"))
active = {c["id"] for c in d["categories"]}
d["subcategories"] = [s for s in d["subcategories"]
                      if re.fullmatch(r"[A-Z]{2}\.[A-Z]{2}-\d{2}", s["id"]) and s["id"][:5] in active]
json.dump(d, open(p, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("post-fix: CSF 2.0 active subcategories:", len(d["subcategories"]))

# ---- CSF 2.0: keep only subcategories without a withdraw_reason ----
raw = json.load(open(f"{ROOT}/data/raw/frameworks/csf2_cprt.json", encoding="utf8"))["response"]["elements"]
withdrawn = {e["element_identifier"][3:] for e in raw["elements"] if e.get("element_type") == "withdraw_reason"}
d = json.load(open(p, encoding="utf8"))
d["subcategories"] = [s for s in d["subcategories"] if s["id"] not in withdrawn]
json.dump(d, open(p, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("CSF 2.0 non-withdrawn subcategories:", len(d["subcategories"]))
p2 = f"{ROOT}/data/processed/nist_ai_rmf_subcategories.json"
d2 = json.load(open(p2, encoding="utf8"))
for s in d2["subcategories"]:
    s["text"] = re.split(r"\s\d\. AI RMF Profiles|\s6\. ", s["text"])[0].strip()
json.dump(d2, open(p2, "w", encoding="utf8"), ensure_ascii=False, indent=1)
