"""Helpers for the sensitivity analyses of the ATLAS threat-coverage and crosswalk analyses
(run_sensitivity.py) and for the second-coder material (make_second_coder.py).

Every specification file (spec/*.yaml, spec/crosswalk_data.py), the ATLAS release, the
framework inventories in data/processed and the reference profiles are read at run time, so the
analyses can be re-run unchanged after the specification is revised. Nothing in aitcert/, spec/
or any existing result file is modified by this module.

Grades are handled as ordinal codes 0 (none) < 1 (partial, P) < 2 (full, F). Because every
weighting used in the article (w(F) = 1, w(P) = w in [0, 1], strict: w(P) = 0) is a non-decreasing
function of the code, max_x w(m, x) = w(max_x code(m, x)); all coverage measures are therefore
computed from the best code per unit / mitigation.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
from collections import OrderedDict
from itertools import combinations
from math import factorial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))            # code/aitcert
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))       # project root
SPEC = os.path.join(HERE, "spec")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from aitcert.atlas import Atlas  # noqa: E402
from aitcert.model import Catalogue  # noqa: E402

GRADES = ("F", "P")
CODE = {"P": 1, "F": 2}
GRADE_OF_CODE = {1: "P", 2: "F"}
SVC_LEVEL_RE = re.compile(r"^([A-Z]{2})-(\d)$")       # level-specific target, e.g. 'IR-2' (none at present)

PATHS = {
    "atlas_map": os.path.join(SPEC, "atlas_mitigation_map.yaml"),
    "crosswalk": os.path.join(SPEC, "crosswalk_data.py"),
    "legacy_guarantees": os.path.join(SPEC, "legacy_guarantees.yaml"),
    "ai_guarantees": os.path.join(SPEC, "ai_guarantees.yaml"),
    "legacy_services": os.path.join(SPEC, "legacy_services.yaml"),
    "ai_services": os.path.join(SPEC, "ai_services.yaml"),
    "rmf": os.path.join(ROOT, "data", "processed", "nist_ai_rmf_subcategories.json"),
    "csf": os.path.join(ROOT, "data", "processed", "nist_csf2.json"),
    "aiact": os.path.join(ROOT, "data", "processed", "aiact_paragraphs.json"),
    "ir8596": os.path.join(ROOT, "data", "processed", "nist_ir8596_priorities.json"),
    "ndtzi_profiles": os.path.join(ROOT, "data", "processed", "nd_tzi_2.5-005-99_profiles.json"),
    "reference_profiles": os.path.join(ROOT, "data", "results", "reference_profiles.json"),
    "atlas_coverage": os.path.join(ROOT, "data", "results", "atlas_coverage.json"),
    "crosswalk_coverage": os.path.join(ROOT, "data", "results", "crosswalk_coverage.json"),
}

FW_ORDER = ["NIST AI RMF 1.0", "EU AI Act Arts. 9-15", "ISO/IEC 42001 Annex A", "OWASP LLM Top 10 (2025)", "NIST CSF 2.0"]
FW_SHORT = {"NIST AI RMF 1.0": "AI RMF", "EU AI Act Arts. 9-15": "AI Act", "ISO/IEC 42001 Annex A": "ISO 42001",
            "OWASP LLM Top 10 (2025)": "OWASP LLM", "NIST CSF 2.0": "CSF 2.0"}
CSF_HIGH = "NIST CSF 2.0 (Cyber AI Profile, Secure: High)"


# ============================================================================ bookkeeping
def rel(path: str) -> str:
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:                       # different drive (e.g. a scratch directory)
        return os.path.abspath(path).replace(os.sep, "/")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def software_versions() -> Dict[str, str]:
    import scipy
    import pandas
    import networkx
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pandas.__version__, "pyyaml": yaml.__version__,
            "networkx": networkx.__version__}


def meta_block(script: str, t0: float, t1: float, seeds: Dict, inputs: Iterable[str], notes: str = "") -> Dict:
    return {"script": rel(script), "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "runtime_s": round(t1 - t0, 3), "software": software_versions(), "seeds": seeds,
            "inputs_sha256": {rel(p): sha256(p) for p in inputs if os.path.exists(p)}, "notes": notes}


def jsonable(o):
    """Recursively convert numpy scalars/arrays for json.dump."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ============================================================================ specification
def atlas_yaml_path(map_path: str = PATHS["atlas_map"]) -> Tuple[str, str]:
    """Locate the ATLAS release named in the 'source' field of the mitigation map."""
    src = str(yaml.safe_load(open(map_path, encoding="utf8")).get("source", ""))
    m = re.search(r"release\s+(\d{4}\.\d{2}(?:\.\d+)?)", src)
    if m:
        cands = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "atlas-data", "dist", "*", f"ATLAS-{m.group(1)}.yaml")))
        if cands:
            return cands[-1], m.group(1)
    return os.path.join(ROOT, "data", "raw", "atlas-data", "dist", "v6", "ATLAS-2026.09.yaml"), "2026.09"


def guarantee_rows(path: str) -> "OrderedDict[str, Dict]":
    """Assurance rows 'G:<id>' with their text and the first guarantee level whose mark starts with '+'."""
    d = yaml.safe_load(open(path, encoding="utf8"))
    out: "OrderedDict[str, Dict]" = OrderedDict()
    for dim, dd in d["dimensions"].items():
        for r in dd["rows"]:
            marks = [str(x) for x in r["marks"]]
            first = next((i + 1 for i, mk in enumerate(marks) if mk.startswith("+")), None)
            out["G:" + r["id"]] = {"dimension": dim, "text": r["text"], "marks": marks, "first_plus": first}
    return out


class Elements:
    """Criteria elements (targets): services, assurance rows and profile mechanisms."""
    PROFILE_DESC = {
        "X:PROFILE": "Legacy rating and profile scheme (functional profiles of ND TZI 2.5-005-99: class, subclass, "
                     "required service levels, with a guarantee level G-1..G-7).",
        "X:AIPROFILE": "Risk-proportionate derivation of AI-extended profiles (class-I levels chosen by risk, closure "
                       "under necessary conditions) with probabilistic acceptance of quantitative requirements "
                       "(one-sided confidence bound at level 1-alpha_G must clear the threshold tau).",
    }

    def __init__(self):
        self.cat = Catalogue.load(True)
        self.leg_svc = [c for c, s in self.cat.services.items() if s.origin == "legacy"]
        self.ai_svc = [c for c, s in self.cat.services.items() if s.origin == "ai"]
        self.leg_rows_info = guarantee_rows(PATHS["legacy_guarantees"])
        self.ai_rows_info = guarantee_rows(PATHS["ai_guarantees"])
        self.leg_rows = list(self.leg_rows_info)
        self.ai_rows = list(self.ai_rows_info)
        self.rows_info = OrderedDict(list(self.leg_rows_info.items()) + list(self.ai_rows_info.items()))
        self.prof_leg, self.prof_ai = ["X:PROFILE"], ["X:AIPROFILE"]
        self.X_leg = self.leg_svc + self.leg_rows + self.prof_leg
        self.X_ai = self.ai_svc + self.ai_rows + self.prof_ai          # class I + AI rows (+ AI profile mechanism)
        self.X_ext = self.X_leg + self.X_ai
        self.group = {}
        for lst, g in ((self.leg_svc, "legacy_service"), (self.leg_rows, "legacy_row"), (self.prof_leg, "legacy_profile"),
                       (self.ai_svc, "ai_service"), (self.ai_rows, "ai_row"), (self.prof_ai, "ai_profile")):
            for x in lst:
                self.group[x] = g
        assert len(set(self.X_ext)) == len(self.X_ext), "duplicate element codes"

    def counts(self) -> Dict[str, int]:
        return {"legacy_services": len(self.leg_svc), "ai_services": len(self.ai_svc), "legacy_rows": len(self.leg_rows),
                "ai_rows": len(self.ai_rows), "profile_mechanisms": 2, "X_leg": len(self.X_leg), "X_ai_only": len(self.X_ai),
                "X_ext": len(self.X_ext)}

    def describe(self, x: str, long: bool = False) -> str:
        """Short (or long) description of a target for coders."""
        if x in self.cat.services:
            s = self.cat.services[x]
            lv = "; ".join(f"{l} {v.name}" for l, v in sorted(s.levels.items()))
            if not long:
                return f"{s.name}"
            pur = f" Purpose: {s.purpose}" if s.purpose else ""
            org = "class-I AI service" if s.origin == "ai" else "legacy functional service (ND TZI 2.5-004-99)"
            return f"{s.name} [{org}].{pur} Levels: {lv}."
        if x in self.rows_info:
            r = self.rows_info[x]
            first = f"G-{r['first_plus']}" if r["first_plus"] else "never"
            if not long:
                t = r["text"]
                return (t if len(t) <= 90 else t[:87].rsplit(" ", 1)[0] + "...") + f" (from {first})"
            org = "AI assurance row" if x in self.ai_rows_info else "legacy assurance row (ND TZI 2.5-004-99, Sect. 10)"
            return f"{r['text']} [{org}; dimension: {r['dimension'].replace('_', ' ')}; first required at {first}]"
        if x in self.PROFILE_DESC:
            return self.PROFILE_DESC[x] if long else self.PROFILE_DESC[x].split(" (")[0]
        raise KeyError(x)


def parse_grade_string(s: str) -> Optional["OrderedDict[str, str]"]:
    """'IR:F G:TE-AI-3:P' -> {'IR': 'F', 'G:TE-AI-3': 'P'}; 'NA' -> None; '' -> {} (not covered)."""
    s = (s or "").strip()
    if s == "NA":
        return None
    out: "OrderedDict[str, str]" = OrderedDict()
    for tok in s.split():
        tgt, g = tok.rsplit(":", 1)
        if g not in GRADES:
            raise ValueError(f"bad grade in {tok!r}")
        if tgt not in out or CODE[g] > CODE[out[tgt]]:
            out[tgt] = g
    return out


def load_atlas_grades(path: str = PATHS["atlas_map"]) -> Tuple["OrderedDict[str, OrderedDict]", Dict[str, str]]:
    d = yaml.safe_load(open(path, encoding="utf8"))["mitigations"]
    grades, rationale = OrderedDict(), {}
    for m, v in d.items():
        g = OrderedDict()
        for x, s in (v.get("map") or {}).items():
            if not isinstance(x, str):
                raise ValueError(f"{m}: target {x!r} is not a string (quote YAML keys such as 'NO')")
            if str(s) not in GRADES:
                raise ValueError(f"{m}: bad grade {s!r} for {x}")
            g[x] = str(s)
        grades[m] = g
        rationale[m] = str(v.get("rationale", ""))
    return grades, rationale


def load_crosswalk_module(path: str = PATHS["crosswalk"]):
    spec = importlib.util.spec_from_file_location("cw_spec_sensitivity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_crosswalk(cw=None):
    """Frameworks exactly as in run_crosswalk.py: {framework: {unit: None (NA) | {target: 'F'|'P'}}},
    CSF priority weights (3/2/1 for high/moderate/foundational, Cyber AI Profile 'Secure') and the
    ids of the high-priority CSF subcategories. Raw mapping strings are returned for traceability."""
    cw = cw or load_crosswalk_module()
    rmf = [s["id"] for s in json.load(open(PATHS["rmf"], encoding="utf8"))["subcategories"]]
    csf = [s["id"] for s in json.load(open(PATHS["csf"], encoding="utf8"))["subcategories"]]
    act_ids = {r["id"] for r in json.load(open(PATHS["aiact"], encoding="utf8"))["requirements"]}
    assert sorted(cw.NIST_AI_RMF) == sorted(rmf), set(rmf) ^ set(cw.NIST_AI_RMF)
    assert set(cw.CSF2) <= set(csf), set(cw.CSF2) - set(csf)
    assert set(cw.EU_AI_ACT) <= act_ids, set(cw.EU_AI_ACT) - act_ids
    raw = OrderedDict()
    raw["NIST AI RMF 1.0"] = OrderedDict(cw.NIST_AI_RMF)
    raw["EU AI Act Arts. 9-15"] = OrderedDict(cw.EU_AI_ACT)
    raw["ISO/IEC 42001 Annex A"] = OrderedDict(cw.ISO_42001_ANNEX_A)
    raw["OWASP LLM Top 10 (2025)"] = OrderedDict(cw.OWASP_LLM_2025)
    raw["NIST CSF 2.0"] = OrderedDict((k, cw.CSF2.get(k, "")) for k in csf)
    fw = OrderedDict((name, OrderedDict((k, parse_grade_string(v)) for k, v in mp.items())) for name, mp in raw.items())
    # consistency with the specification's own parser
    for name, mp in raw.items():
        for k, v in mp.items():
            ref, mine = cw.parse(v), fw[name][k]
            assert (ref is None) == (mine is None), (name, k)
            if ref is not None:
                assert ref == {t: (1.0 if g == "F" else 0.5) for t, g in mine.items()}, (name, k, ref, mine)
    prio = {r["id"]: r for r in json.load(open(PATHS["ir8596"], encoding="utf8"))["subcategories"]}
    weights = {k: {1: 3.0, 2: 2.0, 3: 1.0}[prio[k]["secure"]] for k in csf}
    high = [k for k in csf if prio[k]["secure"] == 1]
    return fw, weights, high, raw


def level_targets(mappings: Iterable[Dict[str, Optional[Dict[str, str]]]]) -> List[str]:
    """Level-specific targets such as 'IR-2' (service at level >= 2) found in the mappings."""
    out = set()
    for mp in mappings:
        for m in mp.values():
            for t in (m or {}):
                if SVC_LEVEL_RE.match(t):
                    out.add(t)
    return sorted(out)


# ============================================================================ grade tables
class GradeTable:
    """Units x targets ordinal codes (0 none, 1 P, 2 F). Units mapped to None (NA) are excluded."""

    def __init__(self, mapping: Dict[str, Optional[Dict[str, str]]], targets: Sequence[str], label: str = ""):
        self.targets = list(targets)
        self.tidx = {t: j for j, t in enumerate(self.targets)}
        self.units = [u for u, m in mapping.items() if m is not None]
        self.na_units = [u for u, m in mapping.items() if m is None]
        self.codes = np.zeros((len(self.units), len(self.targets)), dtype=np.int8)
        unknown = set()
        for i, u in enumerate(self.units):
            for t, g in mapping[u].items():
                if t not in self.tidx:
                    unknown.add(t)
                    continue
                self.codes[i, self.tidx[t]] = CODE[g]
        if unknown:
            raise KeyError(f"{label}: targets not in the criteria: {sorted(unknown)}")

    def n_grades(self) -> int:
        return int((self.codes > 0).sum())


def xmask(targets: Sequence[str], members: Iterable[str], levels: Optional[Dict[str, int]] = None,
          cat: Optional[Catalogue] = None) -> np.ndarray:
    """Boolean mask of the targets contained in X. Level-specific targets 'SS-n' are contained iff
    service SS is in X at level >= n (levels given) or, at catalogue level (levels None), iff SS is in X
    and has a level n."""
    mem = set(members)
    out = np.zeros(len(targets), dtype=bool)
    for j, t in enumerate(targets):
        m = SVC_LEVEL_RE.match(t)
        if m and t not in mem:
            s, n = m.group(1), int(m.group(2))
            if levels is not None:
                out[j] = levels.get(s, 0) >= n
            else:
                out[j] = s in mem and (cat is None or n in cat.services[s].levels)
        else:
            out[j] = t in mem
    return out


def best_codes(codes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return codes[:, mask].max(axis=1) if mask.any() else np.zeros(codes.shape[0], dtype=np.int8)


def lut(w: float = 0.5, strict: bool = False) -> np.ndarray:
    return np.array([0.0, 0.0 if strict else w, 1.0])


# ============================================================================ crosswalk
class CrosswalkModel:
    """All crosswalk frameworks stacked into one table so that one perturbation acts on all grades."""

    def __init__(self, fw: Dict[str, Dict], csf_weights: Dict[str, float], csf_high: List[str], targets: Sequence[str]):
        self.fw_names = list(fw)
        self.targets = list(targets)
        self.tables = OrderedDict((n, GradeTable(fw[n], targets, n)) for n in self.fw_names)
        self.codes = np.vstack([t.codes for t in self.tables.values()])
        self.rows = OrderedDict()
        start = 0
        for n, t in self.tables.items():
            self.rows[n] = np.arange(start, start + len(t.units))
            start += len(t.units)
        self.unit_ids = [(n, u) for n, t in self.tables.items() for u in t.units]
        csf = self.tables["NIST CSF 2.0"]
        self.csf_w = np.array([csf_weights[u] for u in csf.units])
        hi = set(csf_high)
        self.rows[CSF_HIGH] = self.rows["NIST CSF 2.0"][[u in hi for u in csf.units]]
        self.groups_rmf = {fn: self.rows["NIST AI RMF 1.0"][[u.startswith(fn) for u in self.tables["NIST AI RMF 1.0"].units]]
                           for fn in ("GOVERN", "MAP", "MEASURE", "MANAGE")}
        act = self.tables["EU AI Act Arts. 9-15"].units
        arts = sorted({re.match(r"Art\.(\d+)\(", u).group(1) for u in act}, key=int)
        self.groups_act = {f"Art. {a}": self.rows["EU AI Act Arts. 9-15"][[u.startswith(f"Art.{a}(") for u in act]] for a in arts}

    def summary(self, mask: np.ndarray, codes: Optional[np.ndarray] = None, full: bool = True) -> Dict:
        c = self.codes if codes is None else codes
        best = best_codes(c, mask)
        out = OrderedDict()
        for n in self.fw_names + [CSF_HIGH]:
            b = best[self.rows[n]]
            r = OrderedDict(n=int(len(b)), w0_5=float(lut(0.5)[b].mean()))
            if full:
                r["w0_25"] = float(lut(0.25)[b].mean())
                r["w0_75"] = float(lut(0.75)[b].mean())
                r["strict"] = float((b == 2).mean())
                r["addressed"] = float((b > 0).mean())
            out[n] = r
        b = best[self.rows["NIST CSF 2.0"]]
        out["NIST CSF 2.0"]["priority_weighted_w0_5"] = float(np.average(lut(0.5)[b], weights=self.csf_w))
        if full:
            out["NIST AI RMF by function"] = {k: {"n": int(len(v)), "w0_5": float(lut(0.5)[best[v]].mean())} for k, v in self.groups_rmf.items()}
            out["EU AI Act by article"] = {k: {"n": int(len(v)), "w0_5": float(lut(0.5)[best[v]].mean())} for k, v in self.groups_act.items()}
        return out

    def unit_values(self, w: float = 0.5, codes: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrix units x targets of s_w(r, x) (the value of each single target for each unit)."""
        c = self.codes if codes is None else codes
        return lut(w)[c]


# ============================================================================ ATLAS
class AtlasModel:
    def __init__(self, atlas: Atlas, grades: Dict[str, Dict[str, str]], targets: Sequence[str]):
        self.atlas = atlas
        self.techs = list(atlas.techniques)
        self.tix = {t: i for i, t in enumerate(self.techs)}
        self.mits = list(atlas.mitigations)
        self.mix = {m: j for j, m in enumerate(self.mits)}
        self.extra_in_map = sorted(set(grades) - set(self.mits))          # in the map but not in this ATLAS release
        self.missing_in_map = sorted(set(self.mits) - set(grades))        # in ATLAS but not in the map (treated as unmapped)
        if self.extra_in_map:
            raise KeyError(f"mitigations in the map but not in ATLAS: {self.extra_in_map}")
        self.table = GradeTable(OrderedDict((m, dict(grades.get(m, {}))) for m in self.mits), targets, "ATLAS map")
        self.targets = list(targets)
        self.codes = self.table.codes
        self.in_scope = np.array([bool(grades.get(m)) for m in self.mits])     # deliberately unmapped = out of scope
        T = np.zeros((len(self.techs), len(self.mits)), dtype=bool)
        for t in self.techs:
            for m in atlas.mitigations_of(t):
                T[self.tix[t], self.mix[m]] = True
        self.T = T
        self.mitigated = T.any(axis=1)
        self.cases = list(atlas.employs)
        C = np.zeros((len(self.cases), len(self.techs)))
        self.case_n = np.zeros(len(self.cases))
        for k, cs in enumerate(self.cases):
            uniq = list(dict.fromkeys(atlas.employs[cs]))
            self.case_n[k] = len(uniq)
            for t in uniq:
                if t in self.tix:
                    C[k, self.tix[t]] = 1.0
        self.C = C
        self.categories = OrderedDict()
        for m in self.mits:
            for c in atlas.mitigations[m].get("categories", []):
                self.categories.setdefault(c, []).append(m)

    # ---- per-technique quantities
    def per_technique(self, mask: np.ndarray, codes: Optional[np.ndarray] = None, w: float = 0.5,
                      scope: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """max / mean / all-partial / all-full aggregators over M(t) (restricted to `scope` mitigations if given).
        Techniques without (in-scope) mitigations score 0 under every aggregator."""
        c = self.codes if codes is None else codes
        mb = best_codes(c, mask)                                   # best code per mitigation
        T = self.T if scope is None else (self.T & scope[None, :])
        has = T.any(axis=1)
        a = lut(w)[mb]
        cmax = np.where(T, mb[None, :], 0).max(axis=1)
        cmin = np.where(T, mb[None, :], 9).min(axis=1)
        cmin[~has] = 0
        nm = T.sum(axis=1)
        mean = np.where(has, (T * a[None, :]).sum(axis=1) / np.maximum(nm, 1), 0.0)
        return {"max_w": lut(w)[cmax], "addressed": (cmax >= 1).astype(float), "full": (cmax >= 2).astype(float),
                "mean_w": mean, "all_partial": (cmin >= 1).astype(float), "all_full": (cmin >= 2).astype(float), "has": has}

    def case_values(self, per_t: np.ndarray) -> np.ndarray:
        return (self.C @ per_t) / self.case_n

    def summary(self, mask: np.ndarray, codes: Optional[np.ndarray] = None, full: bool = True) -> Dict:
        pt = self.per_technique(mask, codes)
        mt = self.mitigated
        out = OrderedDict()
        out["mitigated"] = OrderedDict(n=int(mt.sum()), addressed=float(pt["addressed"][mt].mean()),
                                       full=float(pt["full"][mt].mean()), weighted=float(pt["max_w"][mt].mean()))
        if full:
            out["all"] = OrderedDict(n=len(self.techs), addressed=float(pt["addressed"].mean()), full=float(pt["full"].mean()),
                                     weighted=float(pt["max_w"].mean()))
            out["case_means"] = OrderedDict((k, float(self.case_values(pt[v]).mean()))
                                            for k, v in (("weighted", "max_w"), ("addressed", "addressed"), ("full", "full")))
        return out

    def mitigation_best(self, mask: np.ndarray, codes: Optional[np.ndarray] = None) -> np.ndarray:
        return best_codes(self.codes if codes is None else codes, mask)

    def element_values(self, w: float = 0.5, codes: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrix techniques x targets: cov(t | {x}) = max_{m in M(t)} w(m, x)."""
        c = self.codes if codes is None else codes
        Wm = lut(w)[c]                                             # mitigations x targets
        return (self.T[:, :, None] * Wm[None, :, :]).max(axis=1)


# ============================================================================ perturbation
def reassign(codes: np.ndarray, rng: np.random.Generator, frac: float = 0.1) -> Tuple[np.ndarray, int]:
    """Reassign round(frac * N) of the N existing grades, chosen uniformly without replacement:
    F -> P or none, P -> F or none, each alternative with probability 1/2."""
    idx = np.argwhere(codes > 0)
    k = int(round(frac * len(idx)))
    sel = idx[rng.choice(len(idx), size=k, replace=False)]
    coin = rng.integers(0, 2, size=k)
    cur = codes[sel[:, 0], sel[:, 1]]
    new = np.where(cur == 2, np.where(coin == 0, 1, 0), np.where(coin == 0, 2, 0)).astype(codes.dtype)
    out = codes.copy()
    out[sel[:, 0], sel[:, 1]] = new
    return out, k


def add_random(codes: np.ndarray, rng: np.random.Generator, k: int, row_ok: np.ndarray, col_ok: np.ndarray) -> np.ndarray:
    """Omission-error model: assign grade F or P (probability 1/2 each) to k uniformly chosen ungraded cells."""
    cand = np.argwhere((codes == 0) & row_ok[:, None] & col_ok[None, :])
    sel = cand[rng.choice(len(cand), size=k, replace=False)]
    out = codes.copy()
    out[sel[:, 0], sel[:, 1]] = np.where(rng.integers(0, 2, size=k) == 0, 2, 1)
    return out


def pct(v: Sequence[float], qs=(2.5, 50, 97.5)) -> Dict[str, float]:
    v = np.asarray(v, dtype=float)
    out = {f"p{q:g}": float(np.percentile(v, q)) for q in qs}
    out["mean"] = float(v.mean())
    out["min"] = float(v.min())
    out["max"] = float(v.max())
    return out


# ============================================================================ attribution
def shapley_max_rows(V: np.ndarray) -> np.ndarray:
    """Exact Shapley values for each row of V of the game v(S) = max_{i in S} V[row, i] (v(empty) = 0,
    V >= 0). Airport-game formula (Littlechild and Owen 1973): with the values sorted ascending,
    phi_(i) = sum_{j <= i} (a_(j) - a_(j-1)) / (n - j + 1)."""
    V = np.asarray(V, dtype=float)
    n = V.shape[1]
    order = np.argsort(V, axis=1, kind="stable")
    a = np.take_along_axis(V, order, axis=1)
    inc = np.diff(np.concatenate([np.zeros((V.shape[0], 1)), a], axis=1), axis=1)
    cum = np.cumsum(inc / (n - np.arange(n))[None, :], axis=1)
    phi = np.empty_like(V)
    np.put_along_axis(phi, order, cum, axis=1)
    return phi


def shapley_bruteforce(values: Sequence[float]) -> np.ndarray:
    """Exact Shapley values of v(S) = max_{i in S} values[i] by enumeration (for verification, n <= 12)."""
    vals = list(values)
    n = len(vals)
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            wgt = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in combinations(others, r):
                vs = max([vals[j] for j in S], default=0.0)
                phi[i] += wgt * (max(vs, vals[i]) - vs)
    return phi


# ============================================================================ agreement statistics
def weighted_kappa(a: Sequence[int], b: Sequence[int], k: int = 3, weights: Optional[str] = "linear") -> float:
    """Cohen's kappa for two raters on ordinal categories 0..k-1 with linear, quadratic or no (None) weights."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    O = np.zeros((k, k))
    np.add.at(O, (a, b), 1.0)
    O /= O.sum()
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        Wd = (i != j).astype(float)
    elif weights == "linear":
        Wd = np.abs(i - j) / (k - 1)
    elif weights == "quadratic":
        Wd = ((i - j) / (k - 1)) ** 2
    else:
        raise ValueError(weights)
    de = (Wd * E).sum()
    return float("nan") if de == 0 else float(1.0 - (Wd * O).sum() / de)


def krippendorff_alpha(data: np.ndarray, level: str = "ordinal") -> float:
    """Krippendorff's alpha for a coders x units matrix (np.nan = missing), metric 'nominal', 'ordinal',
    'interval' or 'ratio' (Krippendorff 2011, coincidence-matrix formulation)."""
    data = np.asarray(data, dtype=float)
    vals = np.unique(data[~np.isnan(data)])
    if len(vals) < 2:
        return float("nan")
    vix = {v: i for i, v in enumerate(vals)}
    K = len(vals)
    o = np.zeros((K, K))
    for u in range(data.shape[1]):
        col = data[:, u]
        col = col[~np.isnan(col)]
        m = len(col)
        if m < 2:
            continue
        for x in range(m):
            for y in range(m):
                if x != y:
                    o[vix[col[x]], vix[col[y]]] += 1.0 / (m - 1)
    nc = o.sum(axis=1)
    n = nc.sum()
    i, j = np.indices((K, K))
    if level == "nominal":
        d2 = (i != j).astype(float)
    elif level == "interval":
        d2 = (vals[i] - vals[j]) ** 2
    elif level == "ratio":
        s = vals[i] + vals[j]
        d2 = np.where(s == 0, 0.0, ((vals[i] - vals[j]) / np.where(s == 0, 1, s)) ** 2)
    elif level == "ordinal":
        cs = np.concatenate([[0.0], np.cumsum(nc)])
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        d2 = (cs[hi + 1] - cs[lo] - (nc[i] + nc[j]) / 2.0) ** 2
    else:
        raise ValueError(level)
    De = (np.outer(nc, nc) * d2).sum()
    return float("nan") if De == 0 else float(1.0 - (n - 1) * (o * d2).sum() / De)
