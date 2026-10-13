"""Parse EU AI Act article pages (consolidated text from artificialintelligenceact.eu,
which marks Digital-Omnibus amendments) into paragraph/point-level records.

For every paragraph the *current* (amended) wording and, where it differs, the
*original* 2024 wording are stored. Glossary tooltips injected by the site are removed.

Usage: python parse_aiact.py <dir_with_artN.html> <out.json>
"""
import json
import os
import re
import sys

from lxml import html as lh

SRC, OUT = sys.argv[1], sys.argv[2]


def text_of(span):
    span = lh.fromstring(lh.tostring(span))  # private copy
    for tip in span.xpath('.//span[contains(@class,"term-tip")]'):
        tip.getparent().remove(tip)
    return re.sub(r"\s+", " ", span.text_content()).strip()


def parse_article(path, num):
    doc = lh.parse(path).getroot()
    out = []
    for div in doc.xpath('//div[contains(@class,"para") and starts-with(@id,"p-article-%d-")]' % num):
        cls = div.get("class", "")
        depth = int(re.search(r"depth-(\d+)", cls).group(1)) if "depth-" in cls else 0
        label = div.xpath('./span[@class="para-label"]')
        label = label[0].text_content().strip() if label else ""
        cur = div.xpath('./span[contains(@class,"para-text") and not(contains(@class,"om-old"))]')
        old = div.xpath('./span[contains(@class,"para-text") and contains(@class,"om-old")]')
        status = "amended" if "om-amended" in cls else ("new" if "om-new" in cls or "om-insert" in cls else ("deleted" if "om-delete" in cls else "original"))
        out.append({
            "anchor": div.get("id"),
            "depth": depth,
            "label": label,
            "status": status,
            "text": text_of(cur[0]) if cur else "",
            "original_text": text_of(old[0]) if old else None,
        })
    title = doc.xpath('//h1')
    return out


records, meta = [], {}
files = sorted((f for f in os.listdir(SRC) if f.endswith(".html")), key=lambda x: int(re.findall(r"\d+", x)[0]))
for fn in files:
    num = int(re.findall(r"\d+", fn)[0])
    items = parse_article(os.path.join(SRC, fn), num)
    # build hierarchical ids: paragraph labels '1.' at depth 0, points '(a)' at depth 1, '(i)' at depth 2
    para = point = None
    for it in items:
        lab = it["label"].strip()
        if it["depth"] == 0:
            para, point = (lab.rstrip(".") or para), None
            rid = f"Art.{num}({para})" if para else f"Art.{num}"
        elif it["depth"] == 1:
            point = lab.strip("()")
            rid = f"Art.{num}({para})({point})" if para else f"Art.{num}({point})"
        else:
            rid = f"Art.{num}({para})({point})({lab.strip('()')})"
        it.update({"article": num, "id": rid})
        records.append(it)
    raw = open(os.path.join(SRC, fn), encoding="utf8", errors="ignore").read()
    m = re.search(r"Comes into force ([^<]+?)(?:, according to|<)", raw)
    t = re.search(r"<title>([^<]+)</title>", raw)
    meta[num] = {"page_title": t.group(1).strip() if t else "", "applicability": m.group(1).strip() if m else ""}

json.dump({"source": "https://artificialintelligenceact.eu/article/<n>/ (consolidated text incl. Digital Omnibus markup)",
           "retrieved": "2026-10-11", "meta": meta, "requirements": records},
          open(OUT, "w", encoding="utf8"), ensure_ascii=False, indent=1)
from collections import Counter
print(len(records), "records", dict(Counter(r["article"] for r in records)))
print(Counter(r["status"] for r in records))
