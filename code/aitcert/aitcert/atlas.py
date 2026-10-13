"""Threat-coverage analysis against MITRE ATLAS.

Coverage of a technique t by a set X of services/assurance rows:
    cov(t | X) = max_{m in M(t)} max_{x in X} w(m, x),   w in {0, 0.5 (P), 1 (F)},
where M(t) are the ATLAS mitigations linked to t by a 'mitigates' relationship (mitigations of a
parent technique are inherited by its sub-techniques). A case study c employing techniques T(c)
has coverage  Cov(c | X) = mean_{t in T(c)} cov(t | X); a technique is 'addressed' if cov > 0.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Iterable, List, Set

import yaml

W_STRENGTH = {"F": 1.0, "P": 0.5}


class Atlas:
    def __init__(self, path: str):
        d = yaml.safe_load(open(path, encoding="utf8"))
        self.tactics = d["tactics"]
        self.techniques = d["techniques"]
        self.mitigations = d["mitigations"]
        self.case_studies = d["case-studies"]
        rel = d["relationships"]
        self.mitigates: Dict[str, Set[str]] = defaultdict(set)   # technique -> mitigations
        self.employs: Dict[str, List[str]] = defaultdict(list)   # case study -> techniques (procedure order)
        self.achieves: Dict[str, Set[str]] = defaultdict(set)    # technique -> tactics
        self.parent: Dict[str, str] = {}
        for src, groups in rel.items():
            for rtype, lst in groups.items():
                for r in lst:
                    if rtype == "mitigates":
                        self.mitigates[r["target"]].add(r["source"])
                    elif rtype == "employs":
                        self.employs[r["source"]].append(r["target"])
                    elif rtype == "achieves":
                        self.achieves[r["source"]].add(r["target"])
                    elif rtype == "specializes":
                        self.parent[r["source"]] = r["target"]

    def mitigations_of(self, t: str) -> Set[str]:
        ms = set(self.mitigates.get(t, set()))
        if t in self.parent:
            ms |= self.mitigates.get(self.parent[t], set())
        return ms

    def tactics_of(self, t: str) -> Set[str]:
        return self.achieves.get(t) or self.achieves.get(self.parent.get(t, ""), set())


def load_map(path: str) -> Dict[str, Dict[str, float]]:
    d = yaml.safe_load(open(path, encoding="utf8"))["mitigations"]
    return {m: {x: W_STRENGTH[s] for x, s in v["map"].items()} for m, v in d.items()}


def technique_coverage(atlas: Atlas, mmap: Dict[str, Dict[str, float]], X: Iterable[str]) -> Dict[str, float]:
    X = set(X)
    out = {}
    for t in atlas.techniques:
        best = 0.0
        for m in atlas.mitigations_of(t):
            for x, w in mmap.get(m, {}).items():
                if x in X:
                    best = max(best, w)
        out[t] = best
    return out


def case_coverage(atlas: Atlas, tcov: Dict[str, float]) -> Dict[str, Dict]:
    out = {}
    for cs, techs in atlas.employs.items():
        uniq = list(dict.fromkeys(techs))
        vals = [tcov.get(t, 0.0) for t in uniq]
        out[cs] = {"n_techniques": len(uniq), "weighted": sum(vals) / len(vals) if vals else 0.0,
                   "addressed": sum(v > 0 for v in vals) / len(vals) if vals else 0.0,
                   "full": sum(v >= 1 for v in vals) / len(vals) if vals else 0.0}
    return out
