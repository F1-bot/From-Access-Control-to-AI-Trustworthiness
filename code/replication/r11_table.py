"""R11: the replication table of the manuscript (Table tab:replication), generated from results/r_assessment.json (alpha = 0.05)
so that no number is transcribed by hand. Prints the Markdown table. Usage: python r11_table.py"""
import r_lib as L

A = L.load_json("r_assessment.json")["by_alpha"]["0.05"]
e1 = L.load_json("r_e1.json")
e2 = L.load_json("r_e2.json")
e3 = L.load_json("r_e3.json")
e5 = L.load_json("r_e5.json")["results"]["calibrated"]
f2, f3 = (lambda x: f"{x:.2f}"), (lambda x: f"{x:.3f}")


def cell(tag):
    t = A[tag]["tests"]
    v = e1["variants"][tag]
    out = {}
    r = t["IV-2.recall"]
    out["recall"] = f"{r['k']}/{r['n']}, cluster-robust LCB {f3(r['lcb'])}: {r['decision']}"
    fr = t["IV-2.far"]
    out["far"] = f"{fr['count']} ({f3(fr['estimate'])} per h), UCB {f3(fr['ucb'])}: {fr['decision']}"
    ld = t["IO-2.load"]
    out["load"] = f"{f2(ld['estimate'])}, UCB {f2(ld['ucb'])}: {ld['decision']}"
    x = t["IE-2.xfid"]
    out["xfid"] = (f"{x['absolute']['k']}/{x['absolute']['n']}, LCB {f3(x['absolute']['lcb'])}; better than the constant explainer in "
                   f"{x['comparison']['b']} attacks and worse in {'none' if x['comparison']['c'] == 0 else x['comparison']['c']}, "
                   f"p = {x['comparison']['p_dependence_robust']:.4g}: {x['decision']}")
    s = t["IR-2.asr"]
    out["asr"] = f"strongest attack: {s['k']}/{s['n']}, UCB {f3(s['ucb'])}: {s['decision']}"
    p = t["IP-2.ext"]
    out["ext"] = f"operational windows, largest UCB {f3(p['max_ucb95'])}: {p['decision']}"
    return out


b, h = cell("baseline_s0"), cell("hardened_s0")
sh = A["baseline_s0"]["tests"]
pd_, dg, dr, im = sh["ID-3.pdet"], sh["ID-3.pdeg"], sh["IV-3.drift"], sh["IM-2.sig"]
rows = [
    ("IV-2", "Event recall ≥ 0.50 (50 attacks)", b["recall"], h["recall"]),
    ("IV-2", "False alarms ≤ 0.25 per h (108.3 h)", b["far"], h["far"]),
    ("IO-2", "Alerts ≤ 6 per operator-hour", b["load"], h["load"]),
    ("IE-2", "Process localisation ≥ 0.60 and better than the constant explainer", b["xfid"], h["xfid"]),
    ("IR-2", "Attack success ≤ 0.20 (TM-A, ε ≤ 0.5)", b["asr"], h["asr"]),
    ("ID-3", "Poison detection ≥ 0.80 at clean removal ≤ 0.05 (β = 1%)",
     f"{f2(pd_['estimate'])} (20 traces, LCB {f2(pd_['lcb'])}), clean removal {pd_['clean_removal']['estimate']:.4f}: {pd_['decision']}", "as baseline (shared pipeline)"),
    ("ID-3", "Recall degradation ≤ 0.10 (30 attacks)",
     f"{dg['n10']} lost, {dg['n01']} gained (three-seed vote), UCB {f2(dg['ucb'])}: {dg['decision']}", "not tested (drill run with the baseline recipe only)"),
    ("IP-2", "Extraction fidelity ≤ 0.90 (Q = 10⁴, label-only)", b["ext"], h["ext"]),
    ("IV-3", "Hourly drift alarms: TPR ≥ 0.90, FPR ≤ 0.05",
     f"{dr['tpr']['k']}/{dr['tpr']['n']} and {dr['fpr']['k']}/{dr['fpr']['n']}; dependence-robust FPR UCB {f3(dr['fpr']['ucb'])}: {dr['decision']}", "as baseline (shared monitor)"),
    ("IM-2", "Signature verified; forgeries rejected", f"4 of 4 tests as required: {im['decision']}", "as baseline"),
]
print("| Level | Requirement (threshold) | Baseline | Hardened |")
print("|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")
for tag in L.VARIANTS:
    for reading in ("demonstrated", "conditional"):
        print(tag, reading, A[tag]["results"][reading]["formatted"])
