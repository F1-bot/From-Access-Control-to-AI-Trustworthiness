"""Parse all standard functional protection profiles from ND TZI 2.5-005-99 (UTF-8 text)
and transliterate service codes (ISO 9:1995) -> JSON."""
import json, re, sys
src, out = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf8").read()
t = t[t.find("1.К.1 =") - 10:t.rfind("Додаток А")]
t = re.sub(r"\n\s*\d+\s*\n", "\n", t)  # page numbers
TR = {"К": "K", "Ц": "C", "Д": "D", "Н": "N", "А": "A", "О": "O", "В": "V", "Р": "R", "С": "S", "З": "Z",
      "И": "I", "Т": "T", "П": "P"}
def tr(s):
    return "".join(TR.get(ch, ch) for ch in s)
profiles = []
for m in re.finditer(r"(\d)\s*\.\s*([КЦД]+)\s*\.\s*(\d+)\s*=\s*\{([^}]*)\}", t, flags=re.S):
    cls, sub, num, body = m.groups()
    items = [x.strip() for x in re.split(r"[,\s]+", body.replace("\n", " ")) if x.strip()]
    codes = []
    for it in items:
        mm = re.fullmatch(r"([А-ЯІ]{2})-(\d)", it)
        if not mm:
            raise ValueError(f"bad item {it!r} in {cls}.{sub}.{num}")
        codes.append((tr(mm.group(1)), int(mm.group(2)), it))
    dup = sorted({c for c, l, _ in codes if sum(1 for c2, _, _ in codes if c2 == c) > 1})
    prof = {}
    for c, l, _ in codes:
        prof[c] = max(prof.get(c, 0), l)
    profiles.append({"id": f"{cls}.{tr(sub)}.{num}", "id_uk": f"{cls}.{sub}.{num}", "class": int(cls),
                     "subclass": tr(sub), "number": int(num), "services": prof,
                     "listed_items": len(codes), "duplicates": dup})
json.dump({"source": "ND TZI 2.5-005-99, Section 7", "transliteration": "ISO 9:1995", "profiles": profiles},
          open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
from collections import Counter
print("profiles:", len(profiles), Counter(p["class"] for p in profiles))
print("by subclass:", Counter(f'{p["class"]}.{p["subclass"]}' for p in profiles))
print("duplicates:", [(p["id"], p["duplicates"]) for p in profiles if p["duplicates"]])
