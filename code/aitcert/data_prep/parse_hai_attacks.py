"""Parse the HAI 22.04 attack table (technical details PDF, pages 34-36) into a CSV with attack ID,
primitive(s), target controller(s), target point(s), and duration; then align attacks with the
labelled intervals of test1..test4 (chronological order within each file)."""
import fitz, re, sys, json
import pandas as pd
ROOT = sys.argv[1]
doc = fitz.open(f"{ROOT}/data/raw/hai/hai_dataset_technical_details.pdf")
lines = []
for i in (33, 34, 35):
    lines += [l.strip() for l in doc[i].get_text().split("\n") if l.strip()]
start = lines.index("Target Point(s)")
lines = lines[start + 1:]
attacks, cur = [], None
k = 0
while k < len(lines):
    l = lines[k]
    if re.fullmatch(r"A\d{3}", l):
        cur = {"id": l, "primitives": []}
        attacks.append(cur); k += 1; continue
    if re.fullmatch(r"AP\d+", l) and cur is not None:
        ctrl, tgt = lines[k + 1], lines[k + 2]
        cur["primitives"].append({"ap": l, "controller": ctrl.replace(" ", ""), "targets": [t.strip(" .") for t in re.split(r"[,.]\s*", tgt) if t.strip(" .")]})
        k += 3; continue
    if re.fullmatch(r"\d{1,2}:\d{2}", l) and cur is not None and "start" not in cur:
        cur["start"] = l
        if k + 1 < len(lines) and re.fullmatch(r"\d+", lines[k + 1]):
            cur["duration"] = int(lines[k + 1]); k += 2; continue
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
# align with labelled intervals
aligned = []
for f in ["test1", "test2", "test3", "test4"]:
    lab = pd.read_csv(f"{ROOT}/data/raw/hai/hai-22.04/{f}.csv", usecols=["timestamp", "Attack"])
    a = lab["Attack"].values
    starts = [i for i in range(len(a)) if a[i] == 1 and (i == 0 or a[i - 1] == 0)]
    ends = [i for i in range(len(a)) if a[i] == 1 and (i == len(a) - 1 or a[i + 1] == 0)]
    sub = df[df.test_file == f].reset_index(drop=True)
    assert len(sub) == len(starts), (f, len(sub), len(starts))
    for j, (s, e) in enumerate(zip(starts, ends)):
        r = sub.loc[j].to_dict()
        r.update({"label_start_idx": s, "label_end_idx": e, "label_start": lab.timestamp[s], "label_duration_s": e - s + 1})
        aligned.append(r)
out = pd.DataFrame(aligned)
out.to_csv(f"{ROOT}/data/processed/hai2204_attacks.csv", index=False)
print(len(out), "attacks; duration mismatch (table vs label):", int((out.table_duration_s != out.label_duration_s).sum()))
print(out[["attack_id", "primitives", "target_processes", "table_duration_s", "label_duration_s", "label_start"]].to_string())
