"""Blinded material for an independent second coder of the crosswalk and of the ATLAS mitigation map.

Writes (nothing existing is modified):
    supplementary/second_coder/to_coder/coding_sheet.csv       blinded sheet (no mapping of the authors)
    supplementary/second_coder/to_coder/coder_instructions.md  definitions of full / partial / none, examples
    supplementary/second_coder/to_coder/targets_glossary.csv   all candidate targets with descriptions
    supplementary/second_coder/sample_manifest.json            sample design, ids, candidate lists, digest of the
                                                               authors' mapping of the sampled units (not the mapping)
    data/results/second_coder_reference_key.json               frozen copy of the authors' mapping of the sampled
                                                               units (for scoring; must NOT be sent to the coder)
Sample: stratified by framework, ceil(25%) of the applicable units of each of the five crosswalk frameworks
(numpy default_rng(seed=0), one generator used for the frameworks in a fixed order), plus every ATLAS
mitigation that the authors mapped (non-empty map). Re-run: cd code/aitcert && python make_second_coder.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict

import numpy as np

import sens_lib as L

OUT = os.path.join(L.ROOT, "supplementary", "second_coder")
KEY = os.path.join(L.ROOT, "data", "results", "second_coder_reference_key.json")
ATLAS_FW = "MITRE ATLAS mitigations"
SHEET_COLUMNS = ["item", "id", "framework", "unit_text", "candidate_targets", "coding", "coder_flag", "coded", "notes"]

# Worked examples for the instructions. Only units that are NOT in the sample are used, and only if the
# authors' current mapping still equals the stated coding (checked at run time).
EXAMPLES = [   # (framework, unit, authors' coding, reason); the first MAX_EXAMPLES valid ones are used
    ("EU AI Act Arts. 9-15", "Art.12(1)", "NR:F",
     "The legacy registration (audit) service NR requires the automatic recording of security-relevant events in a "
     "protected log, which is the whole outcome of the unit: full."),
    ("NIST AI RMF 1.0", "MAP 2.2", "IE:F IO:P",
     "IE-1 requires a model or system card that states capabilities and limitations, and instructions for use: full. "
     "IO-1 requires that oversight measures are documented in the instructions for use, which covers only the part of "
     "the unit on how outputs are overseen by humans: partial."),
    ("NIST AI RMF 1.0", "GOVERN 2.1", "NO:P IO:P",
     "Separation of duties (NO) and the assignment of oversight roles (IO-1) give the evaluated system defined roles, but "
     "neither requires the organisation-wide documentation of responsibilities for AI risk: partial for both."),
    ("EU AI Act Arts. 9-15", "Art.12(2)", "NR:P IE:P G:OE-AI-1:F",
     "The AI assurance row OE-AI-1 requires a post-deployment monitoring plan with performance, drift and incident "
     "indicators, which is what the recorded events must serve: full. Registration NR provides the logging mechanism but "
     "does not require these AI-specific purposes, and IE-2 requires explanations to be recorded with decisions: partial "
     "for both."),
    ("ISO/IEC 42001 Annex A", "A.7.5", "ID:F",
     "Documented data provenance is required by ID-1 (and attested provenance by ID-4): full."),
    ("NIST AI RMF 1.0", "GOVERN 1.1", "",
     "Understanding and documenting legal and regulatory requirements is an organisational governance outcome. No "
     "service, assurance row or profile mechanism requires it of the evaluated system or its developer: no target."),
    ("NIST CSF 2.0", "PR.AA-01", "NI:F",
     "Management of identities and credentials of users, services and hardware is required by the identification and "
     "authentication service NI: full."),
    ("OWASP LLM Top 10 (2025)", "LLM10", "DR:F IA:P",
     "Quotas and prevention of resource capture are required by the legacy resource-usage service DR: full. Agency "
     "bounding requires iteration, resource and spending limits for agents (IA-1), which covers only the agentic part "
     "of the risk: partial."),
    ("NIST AI RMF 1.0", "MEASURE 2.11", "IB:F",
     "Evaluation of fairness and bias is required by the bias-management service IB (quantified group parity, IB-2): full."),
    ("NIST CSF 2.0", "GV.OC-01", "",
     "The organisational mission and its influence on risk management are organisational context: no target."),
]
MAX_EXAMPLES = 7
ATLAS_NONE_EXAMPLE = ("AML.M0000",
                      "Limiting the public release of technical information about the AI stack of an organisation is an organisational "
                      "publication policy, not a property of an evaluated system: no target. (The authors excluded this "
                      "mitigation and AML.M0001 from the sample for this reason.)")


# ============================================================================ unit texts
def clean_md(s: str) -> str:
    s = re.sub(r"\[\[[^\]]*\]\]", "", s)                                 # ATLAS citation keys [[ref]]
    s = re.sub(r"\[([^\]]+)\]\((/mitigations/|/techniques/)?([^)]*)\)",
               lambda m: f"{m.group(1)} ({m.group(3)})" if m.group(2) else m.group(1), s)
    s = s.replace("**", "")
    return re.sub(r"\s+", " ", s).strip()


def aiact_texts(unit_ids):
    recs = json.load(open(L.PATHS["aiact"], encoding="utf8"))["requirements"]
    units = set(unit_ids)
    deleted = "Not present after the amendment."

    def fmt(r):
        lab = (r.get("label") or "").strip()
        return (lab + " " if lab else "") + r["text"].strip()

    def parent(uid):
        m = re.match(r"^(.*)\([^()]+\)$", uid)
        p = m.group(1) if m else None
        return p if p and re.match(r"^Art\.\d+\(\d+\)", p) else None

    out = {}
    for uid in unit_ids:
        ctx, p = [], parent(uid)
        while p:
            if p not in units:
                ctx.insert(0, " ".join(fmt(r) for r in recs if r["id"] == p and r["text"].strip() != deleted))
            p = parent(p)
        inner = [u for u in units if u != uid and u.startswith(uid + "(")]
        body = [fmt(r) for r in recs if (r["id"] == uid or (r["id"].startswith(uid + "(") and
                                                          not any(r["id"] == u or r["id"].startswith(u + "(") for u in inner)))
                and r["text"].strip() != deleted]
        art = re.match(r"^Art\.(\d+)(.*)$", uid)
        head = f"EU AI Act, Article {art.group(1)}{art.group(2)}"
        c = (" [Introductory text: " + " ".join(ctx) + "]") if ctx else ""
        out[uid] = f"{head}.{c} {' '.join(body)}".strip()
    return out


def unit_texts(fw, cw, atlas):
    rmf = {s["id"]: s for s in json.load(open(L.PATHS["rmf"], encoding="utf8"))["subcategories"]}
    csfd = json.load(open(L.PATHS["csf"], encoding="utf8"))
    csf = {s["id"]: s for s in csfd["subcategories"]}
    csf_cat = {c["id"]: c for c in csfd["categories"]}
    act = aiact_texts(list(fw["EU AI Act Arts. 9-15"]))
    out = {}
    for u in fw["NIST AI RMF 1.0"]:
        out[("NIST AI RMF 1.0", u)] = f"NIST AI RMF 1.0, {u} (function {rmf[u]['function']}). {rmf[u]['text']}"
    for u in fw["EU AI Act Arts. 9-15"]:
        out[("EU AI Act Arts. 9-15", u)] = act[u]
    for u in fw["ISO/IEC 42001 Annex A"]:
        out[("ISO/IEC 42001 Annex A", u)] = (f"ISO/IEC 42001:2023 Annex A, control {u} \"{cw.ISO_42001_TITLES[u]}\". Title only: read "
                                             f"the control in ISO/IEC 42001:2023 Annex A (and its guidance in Annex B).")
    for u in fw["OWASP LLM Top 10 (2025)"]:
        out[("OWASP LLM Top 10 (2025)", u)] = (f"OWASP Top 10 for LLM Applications 2025, {u}:2025 \"{cw.OWASP_TITLES[u]}\". Title "
                                               f"only: read the entry (description, prevention and mitigation strategies) in "
                                               f"the OWASP document.")
    for u in fw["NIST CSF 2.0"]:
        c = csf_cat[u.split("-")[0]]
        out[("NIST CSF 2.0", u)] = f"NIST CSF 2.0, {u} (category {c['id']} {c['title']}). {csf[u]['text']}"
    for m, d in atlas.mitigations.items():
        out[(ATLAS_FW, m)] = f"MITRE ATLAS mitigation {m} \"{d['name']}\". {clean_md(d.get('description', ''))}"
    return out


# ============================================================================ targets
def candidate_lists(E):
    groups = [("Legacy services", E.leg_svc), ("Class-I AI services", E.ai_svc), ("Legacy assurance rows", E.leg_rows),
              ("AI assurance rows", E.ai_rows), ("Profile mechanisms", E.prof_leg + E.prof_ai)]
    cw = [x for _, xs in groups for x in xs]
    at = [x for x in cw if not x.startswith("X:")]

    def cell(allowed):
        parts = []
        for g, xs in groups:
            xs = [x for x in xs if x in allowed]
            if xs:
                parts.append(f"[{g}] " + " | ".join(f"{x} {E.describe(x)}" for x in xs))
        return " || ".join(parts)
    return cw, at, cell(set(cw)), cell(set(at))


def glossary_rows(E):
    rows = []
    for x in E.X_ext:
        g = E.group[x]
        lvl = ""
        if x in E.cat.services:
            s = E.cat.services[x]
            if s.origin == "ai":       # class I: level names with their requirements (spec/ai_services.yaml)
                lvl = " ".join(f"Level {l} ({v.name}): {' '.join(v.requirements)}" for l, v in sorted(s.levels.items()))
            else:                      # legacy services: the specification holds level names only
                lvl = "; ".join(f"{l}: {v.name}" for l, v in sorted(s.levels.items())) + \
                      " (level requirements: ND TZI 2.5-004-99, Sections 6-9)"
        elif x in E.rows_info:
            fp = E.rows_info[x]["first_plus"]
            lvl = f"first required at G-{fp}" if fp else "not required at any G level"
        rows.append({"code": x, "group": g, "description": E.describe(x, long=True), "levels_or_first_required": lvl,
                     "candidate_for": "crosswalk units only" if x.startswith("X:") else "crosswalk units and ATLAS mitigations"})
    return rows


# ============================================================================ main
def canon_digest(key: dict) -> str:
    s = json.dumps({k: sorted(f"{t}:{g}" for t, g in v.items()) for k, v in sorted(key.items())}, sort_keys=True)
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--key", default=KEY)
    args = ap.parse_args()
    t0 = time.time()
    E = L.Elements()
    cwmod = L.load_crosswalk_module()
    fw, _, _, raw = L.load_crosswalk(cwmod)
    atlas_path, release = L.atlas_yaml_path()
    A = L.Atlas(atlas_path)
    grades, _ = L.load_atlas_grades()
    texts = unit_texts(fw, cwmod, A)
    cand_cw, cand_at, cell_cw, cell_at = candidate_lists(E)

    # ---------------------------------------------------------------- stratified sample
    rng = np.random.default_rng(args.seed)
    strata = OrderedDict()
    for name in L.FW_ORDER:
        units = [u for u, m in fw[name].items() if m is not None]          # applicable units, canonical order
        n = int(math.ceil(args.fraction * len(units)))
        idx = np.sort(rng.choice(len(units), size=n, replace=False))
        strata[name] = OrderedDict(N_applicable=len(units), n_sampled=n, sampled=[units[i] for i in idx])
    atlas_units = [m for m in A.mitigations if grades.get(m)]
    strata[ATLAS_FW] = OrderedDict(N_applicable=len(atlas_units), n_sampled=len(atlas_units), sampled=atlas_units,
                                   note="all mitigations with a non-empty mapping (census, not a sample)")
    n_cw = sum(strata[n]["n_sampled"] for n in L.FW_ORDER)
    N_cw = sum(strata[n]["N_applicable"] for n in L.FW_ORDER)

    # ---------------------------------------------------------------- reference key (authors' mapping, kept apart)
    key = OrderedDict()
    for name in L.FW_ORDER:
        for u in strata[name]["sampled"]:
            key[f"{name}|{u}"] = OrderedDict(fw[name][u])
    for m in atlas_units:
        key[f"{ATLAS_FW}|{m}"] = OrderedDict(grades[m])
    digest = canon_digest(key)

    # ---------------------------------------------------------------- sheet
    os.makedirs(os.path.join(args.out, "to_coder"), exist_ok=True)
    rows, item = [], 0
    for name in list(L.FW_ORDER) + [ATLAS_FW]:
        for u in strata[name]["sampled"]:
            item += 1
            rows.append({"item": item, "id": u, "framework": name if name != ATLAS_FW else f"{ATLAS_FW} ({release})",
                         "unit_text": texts[(name, u)], "candidate_targets": cell_at if name == ATLAS_FW else cell_cw,
                         "coding": "", "coder_flag": "", "coded": "", "notes": ""})
    sheet = os.path.join(args.out, "to_coder", "coding_sheet.csv")
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    gl = os.path.join(args.out, "to_coder", "targets_glossary.csv")
    with open(gl, "w", encoding="utf-8-sig", newline="") as f:
        g = glossary_rows(E)
        w = csv.DictWriter(f, fieldnames=list(g[0]), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(g)

    # ---------------------------------------------------------------- examples (non-sampled units only)
    sampled = {(n, u) for n in strata for u in strata[n]["sampled"]}
    ex_used, ex_skipped = [], []
    for name, u, coding, why in EXAMPLES:
        cur = fw[name].get(u)
        cur_s = " ".join(f"{t}:{g}" for t, g in (cur or {}).items())
        if (name, u) in sampled:
            ex_skipped.append(f"{name} {u}: in the sample")
        elif cur is None or L.parse_grade_string(coding) != cur:
            ex_skipped.append(f"{name} {u}: current mapping '{cur_s}' differs from the example '{coding}'")
        elif len(ex_used) >= MAX_EXAMPLES:
            ex_skipped.append(f"{name} {u}: not needed")
        else:
            ex_used.append((name, u, coding, why))
    assert not ({(n, u) for n, u, _, _ in ex_used} & sampled), "an example unit is in the sample"
    at_ok = not grades.get(ATLAS_NONE_EXAMPLE[0]) and ATLAS_NONE_EXAMPLE[0] in A.mitigations
    instr = os.path.join(args.out, "to_coder", "coder_instructions.md")
    with open(instr, "w", encoding="utf8") as f:
        f.write(instructions_md(E, rows, strata, ex_used, at_ok, texts, release, n_cw, N_cw))

    # ---------------------------------------------------------------- manifest and key
    t1 = time.time()
    meta = L.meta_block(os.path.abspath(__file__), t0, t1, {"sample": args.seed},
                        [L.PATHS[k] for k in ("atlas_map", "crosswalk", "legacy_guarantees", "ai_guarantees",
                                              "legacy_services", "ai_services", "rmf", "csf", "aiact")] + [atlas_path],
                        notes="numpy.random.default_rng(seed).choice(N, n, replace=False) per framework, frameworks in the "
                              "order " + ", ".join(L.FW_ORDER) + "; ids sorted back into canonical order.")
    manifest = OrderedDict(
        meta=meta, design=OrderedDict(fraction=args.fraction, rounding="ceil", seed=args.seed, stratified_by="framework",
                                      n_crosswalk_sampled=n_cw, N_crosswalk_applicable=N_cw,
                                      share_crosswalk_sampled=round(n_cw / N_cw, 4), n_atlas=len(atlas_units),
                                      n_total=len(rows), atlas_release=release),
        strata=strata,
        candidate_targets=OrderedDict(crosswalk=cand_cw, atlas=cand_at),
        target_groups=OrderedDict((x, E.group[x]) for x in E.X_ext),
        legacy_groups=["legacy_service", "legacy_row", "legacy_profile"],
        reference_mapping_sha256=digest,
        reference_mapping_digest_rule="sha256 of json.dumps({'<framework>|<id>': sorted('<target>:<grade>')}, sort_keys=True)",
        examples_in_instructions=[f"{n} {u}" for n, u, _, _ in ex_used] + ([f"{ATLAS_FW} {ATLAS_NONE_EXAMPLE[0]}"] if at_ok else []),
        examples_skipped=ex_skipped,
        files=OrderedDict(sheet=L.rel(sheet), instructions=L.rel(instr), glossary=L.rel(gl), reference_key=L.rel(args.key)))
    json.dump(L.jsonable(manifest), open(os.path.join(args.out, "sample_manifest.json"), "w", encoding="utf8"), indent=1)
    keyout = OrderedDict(
        warning="Authors' mapping of the sampled units. Do NOT send to the second coder.",
        meta=meta, reference_mapping_sha256=digest,
        mapping=OrderedDict((k, " ".join(f"{t}:{g}" for t, g in v.items())) for k, v in key.items()))
    os.makedirs(os.path.dirname(args.key), exist_ok=True)
    json.dump(keyout, open(args.key, "w", encoding="utf8"), indent=1)
    print(f"{len(rows)} units ({n_cw} of {N_cw} applicable crosswalk units = {n_cw / N_cw:.1%}, {len(atlas_units)} ATLAS "
          f"mitigations); examples used {len(ex_used)}, skipped {ex_skipped}; digest {digest[:16]}")


def instructions_md(E, rows, strata, examples, atlas_ok, texts, release, n_cw, N_cw) -> str:
    n_items = len(rows)
    strata_lines = "\n".join(f"| {n} | {s['n_sampled']} | {s['N_applicable']} |" for n, s in strata.items())
    ex_lines = []
    for name, u, coding, why in examples:
        t = texts[(name, u)]
        t = t if len(t) <= 420 else t[:417].rsplit(" ", 1)[0] + " ..."
        ex_lines.append(f"**{u}** ({name}). *{t}*\n\n- Coding: `{coding or '(empty)'}`\n- Reason: {why}\n")
    if atlas_ok:
        ex_lines.append(f"**{ATLAS_NONE_EXAMPLE[0]}** (ATLAS mitigation). *{texts[(ATLAS_FW, ATLAS_NONE_EXAMPLE[0])][:300]} ...*\n\n"
                        f"- Coding: `(empty)`\n- Reason: {ATLAS_NONE_EXAMPLE[1]}\n")
    return f"""# Coding instructions for the second coder

Version generated by `code/aitcert/make_second_coder.py`. Please read these instructions in full before you start.

## 1. Purpose

The article maps requirement units of five AI and cybersecurity frameworks, and the mitigations of MITRE ATLAS
(release {release}), to the elements of graded security-evaluation criteria (ND TZI 2.5-004-99 lineage) and of
their AI extension (service class *I* and AI assurance rows). You will code a sample of the same units
**independently**. Your coding will be compared with the authors' coding by weighted Cohen's kappa and
Krippendorff's alpha.

Please do **not** look at the authors' mapping (supplementary table S2, the files `crosswalk_data.py` or
`atlas_mitigation_map.yaml`, or Figures and tables of the article that show coverage) and do not discuss
the units with the authors until you have returned the completed sheet.

## 2. Material

- `coding_sheet.csv`: {n_items} units, one per row ({n_cw} of the {N_cw} applicable crosswalk units, a
  stratified random sample, and all {strata['MITRE ATLAS mitigations']['n_sampled']} ATLAS mitigations except the two
  organisational publication policies AML.M0000 and AML.M0001). Columns: `item`, `id`, `framework`, `unit_text`, `candidate_targets` (the targets you may
  use, with short names), and four columns for you: `coding`, `coder_flag`, `coded`, `notes`.
- `targets_glossary.csv`: every target with a full description: legacy functional services with their levels,
  class-*I* AI services with purpose, levels and the requirements of each level, legacy and AI assurance rows
  with the guarantee level G-k at which they are first required, and the two profile mechanisms.
- Appendix A of the article (level requirements of class *I*) and, if you have it, ND TZI 2.5-004-99.
- ISO/IEC 42001:2023 and the OWASP Top 10 for LLM Applications 2025: the sheet gives only the titles of their
  units (copyright); please read the full control or entry before coding it.

| Stratum | Units in your sheet | Applicable units |
|---|---|---|
{strata_lines}

## 3. The decision

For every unit, decide for each candidate target whether the target **requires** the outcome or measure that
the unit describes. Code from the point of view of an evaluation of a product (the evaluated system and its
developer), at catalogue level: a service counts if **any** of its levels requires the outcome, and an
assurance row counts if it is required at **any** guarantee level. You do not need to name the level, but you
may note it in `notes`.

- **F (full)**: the target explicitly requires the outcome or measure as a whole, so that an evaluation against
  this target would check it.
- **P (partial)**: the target directly contributes to the outcome, implements an essential part of it, or
  requires it only for part of its scope or in a weaker form, but it does not require the whole outcome.
- **none** (leave out): the target is unrelated, only thematically similar, or relevant only through
  organisational processes that the criteria do not require of the evaluated product.

Section 5 gives worked examples of all three grades.

Rules:

1. Credit a target only when the criteria themselves require (F) or directly contribute to (P) the outcome for
   the evaluated system or its developer. Organisational governance outcomes without a product-level
   counterpart (organisation-wide policies, workforce, culture, legal monitoring, external reporting by the organisation)
   are not covered by any target: leave the coding empty.
2. A unit may have several targets, including several F. Record every target that is F or P.
3. Use the most specific target that applies, but also record other targets that independently meet the rule.
   Do not add a target only because it is a precondition of another target (for example, NC-1 is a necessary
   condition of every service and is not credited for that reason).
4. The profile mechanisms are credited only for units whose outcome is a mechanism of the criteria themselves:
   `X:PROFILE` when the outcome is met by the legacy rating and profile scheme (choosing required service levels
   and a guarantee level); `X:AIPROFILE` when the outcome is the risk-proportionate selection of AI requirements
   or the statistical acceptance of quantitative claims (thresholds, confidence bounds at level 1-alpha_G).
   ATLAS mitigations are coded against services and assurance rows only.
5. Judge the unit text, not your expectation of how much the criteria should cover.

## 4. How to record

- `coding`: the targets with their grade, separated by spaces, written exactly as in the glossary, e.g.
  `NR:F IE:P G:OE-AI-1:F`. Assurance rows start with `G:`; the profile mechanisms are `X:PROFILE` and
  `X:AIPROFILE`. Leave `coding` empty if no target is F or P.
- `coder_flag`: write `NA` only if you think the unit is not applicable to the evaluation of a system (for
  example, an obligation of the Commission); leave `coding` empty in that case.
- `coded`: write `1` when you have finished the row, also when `coding` is empty. Rows without `1` are treated as
  not yet coded.
- `notes` (optional): the level you had in mind, doubts, or alternative codings.

Save the file as CSV (UTF-8) without changing the columns `item`, `id` and `framework`. Expected effort is two to
four minutes per unit.

## 5. Worked examples (units that are not in your sheet)

{chr(10).join(ex_lines)}
## 6. After coding

Return the sheet as `coding_sheet_<your initials>.csv`. The authors will compute agreement with the script
`score_second_coder.py`. Disagreements will then be discussed and resolved, and both the independent and the
consensus results will be reported.
"""


if __name__ == "__main__":
    main()
