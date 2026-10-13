"""Parse the HAI 21.03 attack table (technical details PDF, pages 37-39) into a CSV with attack ID,
primitive(s), target controller(s), target point(s) and duration, and align the attacks with the
labelled intervals of test1..test5 (chronological order within each file), as parse_hai_attacks.py
does for HAI 22.04. Only the 'attack' label column of the test files is read.
Usage: python parse_hai2103_attacks.py <project root>"""
import re
import sys

import fitz
import pandas as pd

ROOT = sys.argv[1]
doc = fitz.open(f"{ROOT}/data/raw/hai/hai_dataset_technical_details.pdf")
lines = []
for i in (36, 37, 38):
    lines += [l.strip() for l in doc[i].get_text().split("\n") if l.strip()]
lines = lines[lines.index("50 attacks were conducted, including 25 attack primitives and 25 combinations of attacks") + 1:] \
    if "50 attacks were conducted, including 25 attack primitives and 25 combinations of attacks" in lines else \
    lines[next(i for i, l in enumerate(lines) if l.startswith("The 50 attacks were conducted")) + 1:]
start = lines.index("Target Point(s)")
lines = lines[start + 1:]
stop = next((i for i, l in enumerate(lines) if l.startswith("HAI 20.07")), len(lines))
lines = lines[:stop]
attacks, cur = [], None
k = 0
while k < len(lines):
    l = lines[k]
    if re.fullmatch(r"A\d{3}", l):
        cur = {"id": l, "primitives": []}
        attacks.append(cur)
        k += 1
        continue
    if re.fullmatch(r"AP\d+", l) and cur is not None:
        ctrl, tgt = lines[k + 1], lines[k + 2]
        cur["primitives"].append({"ap": l, "controller": ctrl.replace(" ", ""),
                                  "targets": [t.strip(" .") for t in re.split(r"[,.]\s*", tgt) if t.strip(" .")]})
        k += 3
        continue
    if re.fullmatch(r"\d{1,2}:\d{2}", l) and cur is not None and "start" not in cur:
        cur["start"] = l
        if k + 1 < len(lines) and re.fullmatch(r"\d+", lines[k + 1]):
            cur["duration"] = int(lines[k + 1])
            k += 2
            continue
    k += 1
rows = []
for a in attacks:
    procs = sorted({p["controller"][:2] for p in a["primitives"]})
    rows.append({"attack_id": a["id"], "test_file": f"test{a['id'][1]}", "n_primitives": len(a["primitives"]),
                 "primitives": ";".join(p["ap"] for p in a["primitives"]),
                 "controllers": ";".join(p["controller"] for p in a["primitives"]),
                 "target_points": ";".join(",".join(p["targets"]) for p in a["primitives"]),
                 "target_processes": ";".join(procs), "table_start": a.get("start"), "table_duration_s": a.get("duration")})
df = pd.DataFrame(rows)
assert len(df) == 50, len(df)
aligned = []
for f in ["test1", "test2", "test3", "test4", "test5"]:
    lab = pd.read_csv(f"{ROOT}/data/raw/hai/hai-21.03/{f}.csv.gz", usecols=["time", "attack"])
    a = lab["attack"].values
    starts = [i for i in range(len(a)) if a[i] == 1 and (i == 0 or a[i - 1] == 0)]
    ends = [i for i in range(len(a)) if a[i] == 1 and (i == len(a) - 1 or a[i + 1] == 0)]
    sub = df[df.test_file == f].reset_index(drop=True)
    assert len(sub) == len(starts), (f, len(sub), len(starts))
    for j, (s, e) in enumerate(zip(starts, ends)):
        r = sub.loc[j].to_dict()
        r.update({"label_start_idx": s, "label_end_idx": e, "label_start": lab.time[s], "label_duration_s": e - s + 1})
        aligned.append(r)
out = pd.DataFrame(aligned)
out.to_csv(f"{ROOT}/data/processed/hai2103_attacks.csv", index=False)
print(len(out), "attacks; duration mismatch (table vs label):", int((out.table_duration_s != out.label_duration_s).sum()))
print(out[["attack_id", "primitives", "target_processes", "target_points", "table_start", "label_start",
           "table_duration_s", "label_duration_s"]].to_string())
