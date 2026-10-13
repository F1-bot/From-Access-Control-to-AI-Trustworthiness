"""Formal model of graded functional-service security criteria (ND TZI 2.5-004-99 lineage)
and its AI-trustworthiness extension (class I).

Definitions (cf. Section 3.2 of the article):
  * A catalogue  S  is a finite set of services; each s has levels L(s) = {1..n_s}.
  * A requirement r = (s, l) means "service s at level >= l"; G-k means "assurance level >= k".
  * N(s, l) is the set of necessary conditions of level l of service s.
  * A profile P : S -/-> N is a partial map (service -> required level), optionally with an
    assurance level g(P).
  * Levels are hierarchical: the effective conditions N*(s, l) are the stated conditions of levels 1..l of s
    plus the global condition (NC-1 for every service other than NC). This cumulative reading makes N*
    monotone in the level, which Proposition 1 needs; the per-level reading of the printed tables
    (legacy_cumulative=False) is non-monotone for NC-1 -> NC-2 and KK-2 -> KK-3.
  * P is *valid* iff it is closed under N*:  for all s in dom(P) and every (s', l') in N*(s, P(s)):
    P(s') >= l' (P(s') = 0 outside dom(P)), and every G-k in N*(s, P(s)) satisfies g(P) >= k.
  * P1 <= P2 (dominance) iff dom(P1) is a subset of dom(P2) and P1(s) <= P2(s) for all s in dom(P1).
  * cl(P) is the least valid profile dominating P (least fixed point, Proposition 1); int(P) is the greatest
    valid profile dominated by P at a fixed assurance level (used for ratings).
"""
from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import yaml

SPEC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spec")
REQ_RE = re.compile(r"^([A-Z]{2})-(\d)$")
G_RE = re.compile(r"^G-(\d)$")


@dataclass(frozen=True)
class Level:
    service: str
    level: int
    name: str
    requires: Tuple[str, ...] = ()
    requirements: Tuple[str, ...] = ()
    acceptance: Tuple[dict, ...] = ()


@dataclass
class Service:
    code: str
    code_uk: str
    group: str
    name: str
    name_uk: str
    origin: str  # 'legacy' or 'ai'
    levels: Dict[int, Level] = field(default_factory=dict)
    cumulative: bool = False
    purpose: str = ""
    threats: dict = field(default_factory=dict)

    @property
    def max_level(self) -> int:
        return max(self.levels)


class Catalogue:
    """Service catalogue with necessary-condition semantics."""

    def __init__(self, services: Dict[str, Service], global_requires: Dict[str, List[str]]):
        self.services = services
        self.global_requires = global_requires  # service-origin -> extra requirements on every level

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, include_ai: bool = True, spec_dir: str = SPEC_DIR, legacy_cumulative: bool = True) -> "Catalogue":
        leg = yaml.safe_load(open(os.path.join(spec_dir, "legacy_services.yaml"), encoding="utf8"))
        services: Dict[str, Service] = {}
        for code, s in leg["services"].items():
            svc = Service(code=code, code_uk=s["uk"], group=s["group"], name=s["name"], name_uk=s["name_uk"],
                          origin="legacy", cumulative=legacy_cumulative)
            for lvl, d in s["levels"].items():
                svc.levels[int(lvl)] = Level(code, int(lvl), d["name"], tuple(d.get("requires", [])))
            services[code] = svc
        if include_ai:
            ai = yaml.safe_load(open(os.path.join(spec_dir, "ai_services.yaml"), encoding="utf8"))
            for code, s in ai["services"].items():
                svc = Service(code=code, code_uk=s["uk"], group="I", name=s["name"], name_uk=s["name_uk"],
                              origin="ai", cumulative=True, purpose=s.get("purpose", "").strip(),
                              threats=s.get("threats", {}))
                for lvl, d in s["levels"].items():
                    svc.levels[int(lvl)] = Level(code, int(lvl), d["name"], tuple(d.get("requires", [])),
                                                 tuple(d.get("requirements", [])), tuple(d.get("acceptance", [])))
                services[code] = svc
        return cls(services, {"legacy": ["NC-1"], "ai": ["NC-1"]})

    # ------------------------------------------------------------------ semantics
    def necessary(self, code: str, level: int) -> Tuple[Dict[str, int], int]:
        """Effective necessary conditions N*(code, level): (service -> min level, min assurance level).
        For cumulative services (all services by default), conditions of all lower levels are inherited."""
        svc = self.services[code]
        levels = range(1, level + 1) if svc.cumulative else [level]
        req: Dict[str, int] = {}
        g = 0
        items = list(itertools.chain.from_iterable(svc.levels[l].requires for l in levels))
        if code != "NC":
            items += self.global_requires[svc.origin]
        for it in items:
            m, mg = REQ_RE.match(it), G_RE.match(it)
            if m:
                s2, l2 = m.group(1), int(m.group(2))
                if s2 == code:
                    continue
                req[s2] = max(req.get(s2, 0), l2)
            elif mg:
                g = max(g, int(mg.group(1)))
            else:
                raise ValueError(f"Unparsable requirement {it!r} in {code}-{level}")
        return req, g

    def check_references(self) -> List[str]:
        """All referenced services/levels must exist."""
        errs = []
        for code, svc in self.services.items():
            for lvl in svc.levels:
                req, _ = self.necessary(code, lvl)
                for s2, l2 in req.items():
                    if s2 not in self.services:
                        errs.append(f"{code}-{lvl}: unknown service {s2}")
                    elif l2 not in self.services[s2].levels:
                        errs.append(f"{code}-{lvl}: {s2} has no level {l2}")
        return errs

    def dependency_graph(self, level_nodes: bool = True) -> nx.DiGraph:
        """Directed graph u -> v meaning 'u requires v'."""
        g = nx.DiGraph()
        for code, svc in self.services.items():
            for lvl in svc.levels:
                u = f"{code}-{lvl}" if level_nodes else code
                g.add_node(u, origin=svc.origin, group=svc.group)
                req, gmin = self.necessary(code, lvl)
                for s2, l2 in req.items():
                    v = f"{s2}-{l2}" if level_nodes else s2
                    g.add_edge(u, v)
                if gmin and level_nodes:
                    g.add_edge(u, f"G-{gmin}")
        return g

    # ------------------------------------------------------------------ profiles
    def closure(self, profile: Dict[str, int], g: int = 0) -> Tuple[Dict[str, int], int]:
        """Least valid profile dominating `profile` (least fixed point of the necessary-condition map)."""
        p = dict(profile)
        changed = True
        while changed:
            changed = False
            for code, lvl in list(p.items()):
                req, gmin = self.necessary(code, lvl)
                for s2, l2 in req.items():
                    if p.get(s2, 0) < l2:
                        p[s2] = l2
                        changed = True
                if gmin > g:
                    g = gmin
                    changed = True
        return p, g

    def violations(self, profile: Dict[str, int], g: Optional[int] = None) -> List[str]:
        """Unmet necessary conditions of a profile. If g is None, assurance conditions are reported
        as 'requires G-k' notes rather than violations (standard profiles carry no assurance level)."""
        out = []
        for code, lvl in profile.items():
            if code not in self.services:
                out.append(f"unknown service {code}")
                continue
            if lvl not in self.services[code].levels:
                out.append(f"{code}-{lvl}: level does not exist")
                continue
            req, gmin = self.necessary(code, lvl)
            for s2, l2 in req.items():
                if profile.get(s2, 0) < l2:
                    out.append(f"{code}-{lvl} requires {s2}-{l2} (profile has {s2}-{profile.get(s2, 0) or 'none'})")
            if g is not None and gmin > g:
                out.append(f"{code}-{lvl} requires G-{gmin} (profile has G-{g})")
        return out

    def required_assurance(self, profile: Dict[str, int]) -> int:
        return max([self.necessary(c, l)[1] for c, l in profile.items()] + [0])

    def interior(self, profile: Dict[str, int], g: int = 0) -> Dict[str, int]:
        """Greatest valid profile dominated by `profile` at assurance level g (greatest fixed point of the
        deflationary map that lowers every level whose effective conditions are unmet). Used to turn the
        levels whose requirements were demonstrated into a rating that respects the necessary conditions."""
        p = {s: l for s, l in profile.items() if l > 0}
        changed = True
        while changed:
            changed = False
            for code in list(p):
                lvl = p[code]
                while lvl > 0:
                    req, gmin = self.necessary(code, lvl)
                    if gmin <= g and all(p.get(s2, 0) >= l2 for s2, l2 in req.items()):
                        break
                    lvl -= 1
                if lvl != p[code]:
                    changed = True
                    if lvl == 0:
                        del p[code]
                    else:
                        p[code] = lvl
        return p

    def monotonicity_violations(self) -> List[str]:
        """Pairs (s, l) -> (s, l+1) where N*(s, l+1) does not dominate N*(s, l); empty iff N* is monotone."""
        out = []
        for code, svc in self.services.items():
            for lvl in sorted(svc.levels)[:-1]:
                (r1, g1), (r2, g2) = self.necessary(code, lvl), self.necessary(code, lvl + 1)
                lost = [f"{s}-{l}" for s, l in r1.items() if r2.get(s, 0) < l] + ([f"G-{g1}"] if g2 < g1 else [])
                if lost:
                    out.append(f"{code}-{lvl} -> {code}-{lvl + 1}: loses {', '.join(lost)}")
        return out

    @staticmethod
    def dominates(p_low: Dict[str, int], p_high: Dict[str, int]) -> bool:
        return all(p_high.get(s, 0) >= l for s, l in p_low.items())

    # ------------------------------------------------------------------ notation
    def order(self) -> List[str]:
        """Canonical order of services: K, C, D, N groups as in the standard, then class I."""
        grp = {"K": 0, "C": 1, "D": 2, "N": 3, "I": 4}
        return sorted(self.services, key=lambda c: (grp[self.services[c].group], list(self.services).index(c)))

    def fmt(self, profile: Dict[str, int], uk: bool = False) -> str:
        items = [f"{(self.services[c].code_uk if uk else c)}-{profile[c]}" for c in self.order() if c in profile]
        return "{ " + ", ".join(items) + " }"


def profile_id(as_class: int, subclass: str, number: int, version: str = "") -> str:
    """Identifier <class>.<subclass>.<number>[.<version>]; subclass letters ordered K, C, D, I."""
    order = "KCDI"
    sub = "".join(sorted(set(subclass), key=order.index))
    return f"{as_class}.{sub}.{number}" + (f".{version}" if version else "")


def parse_profile(text: str) -> Dict[str, int]:
    """Parse '{ KD-2, NR-3, ... }' into a dict."""
    out: Dict[str, int] = {}
    for c, l in re.findall(r"([A-Z]{2})-(\d)", text):
        out[c] = max(out.get(c, 0), int(l))
    return out
