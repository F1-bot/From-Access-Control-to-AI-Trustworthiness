"""Crosswalk between external AI frameworks and the criteria (legacy services/assurance rows and
the AI extension). Each requirement unit maps to targets with strength F (full) or P (partial);
'NA' marks units that are not applicable to the evaluation of a system (e.g. obligations of the
Commission); an empty string means 'not covered'.
Targets: legacy services (KD..NP), AI services (ID..IA), assurance rows 'G:<row>' and the profile
mechanisms 'X:PROFILE' (legacy rating/profile scheme) and 'X:AIPROFILE' (risk-proportionate AI
profile derivation with probabilistic acceptance).
Mapping principle: we credit a target only when the criteria themselves REQUIRE (F) or directly
contribute to (P) the outcome for the evaluated system or its developer; organisational governance
outcomes without a product-level counterpart are left uncovered."""

NIST_AI_RMF = {
    "GOVERN 1.1": "", "GOVERN 1.2": "G:DE-AI-1:P", "GOVERN 1.3": "X:PROFILE:P X:AIPROFILE:P", "GOVERN 1.4": "",
    "GOVERN 1.5": "", "GOVERN 1.6": "IM:P", "GOVERN 1.7": "G:DE-AI-1:P",
    "GOVERN 2.1": "NO:P IO:P", "GOVERN 2.2": "IO:P", "GOVERN 2.3": "",
    "GOVERN 3.1": "", "GOVERN 3.2": "IO:F NO:P",
    "GOVERN 4.1": "", "GOVERN 4.2": "G:DO-AI-1:P IE:P", "GOVERN 4.3": "G:TE-1:P G:TE-AI-1:P G:OE-AI-1:P",
    "GOVERN 5.1": "", "GOVERN 5.2": "",
    "GOVERN 6.1": "IM:P", "GOVERN 6.2": "IM:P",
    "MAP 1.1": "G:DS-1:P G:DS-AI-1:F IV:F", "MAP 1.2": "", "MAP 1.3": "", "MAP 1.4": "",
    "MAP 1.5": "X:AIPROFILE:P G:DS-AI-1:P", "MAP 1.6": "G:DS-1:P G:DS-AI-1:P",
    "MAP 2.1": "G:DS-AI-1:P IE:P", "MAP 2.2": "IE:F IO:P", "MAP 2.3": "G:DS-AI-2:F IV:P ID:P",
    "MAP 3.1": "", "MAP 3.2": "G:DO-AI-1:P", "MAP 3.3": "G:DS-AI-1:F", "MAP 3.4": "IO:P", "MAP 3.5": "IO:F",
    "MAP 4.1": "IM:P ID:P", "MAP 4.2": "IM:F",
    "MAP 5.1": "X:AIPROFILE:P", "MAP 5.2": "",
    "MEASURE 1.1": "X:AIPROFILE:F G:TE-AI-1:P", "MEASURE 1.2": "G:TE-AI-2:P IV:P", "MEASURE 1.3": "G:TE-2:P G:TE-AI-4:F",
    "MEASURE 2.1": "G:TE-1:P G:TE-2:P G:TE-AI-1:F G:DO-AI-1:F", "MEASURE 2.2": "", "MEASURE 2.3": "IV:F",
    "MEASURE 2.4": "NR:P NT:P IV:F G:OE-AI-1:F", "MEASURE 2.5": "IV:F IE:P", "MEASURE 2.6": "DS:P DV:P IV:P G:AR-AI-2:F",
    "MEASURE 2.7": "NC:P G:TE-4:P IR:F ID:F IP:F IM:F G:TE-AI-3:F", "MEASURE 2.8": "NR:P IE:F", "MEASURE 2.9": "IE:F",
    "MEASURE 2.10": "KD:P KA:P IP:F", "MEASURE 2.11": "IB:F", "MEASURE 2.12": "", "MEASURE 2.13": "X:AIPROFILE:P G:TE-AI-4:P",
    "MEASURE 3.1": "G:OE-AI-1:P IV:P", "MEASURE 3.2": "", "MEASURE 3.3": "IE:P",
    "MEASURE 4.1": "", "MEASURE 4.2": "IV:P", "MEASURE 4.3": "IV:F G:OE-AI-1:P",
    "MANAGE 1.1": "X:PROFILE:P X:AIPROFILE:P IV:P", "MANAGE 1.2": "", "MANAGE 1.3": "", "MANAGE 1.4": "G:DO-AI-1:F",
    "MANAGE 2.1": "G:AR-AI-2:P", "MANAGE 2.2": "IV:P G:OE-AI-3:P", "MANAGE 2.3": "CO:P DV:P G:OE-AI-3:P",
    "MANAGE 2.4": "IO:F IA:P", "MANAGE 3.1": "IM:F", "MANAGE 3.2": "IM:F IV:P",
    "MANAGE 4.1": "NR:P G:OE-AI-1:F IV:P", "MANAGE 4.2": "G:OE-AI-3:P G:TE-AI-2:P", "MANAGE 4.3": "NR:P DV:P G:OE-AI-1:F",
}

EU_AI_ACT = {  # requirement units of Chapter III, Section 2 (consolidated text incl. Regulation (EU) 2026/1744)
    "Art.9(1)": "G:DE-AI-1:P X:AIPROFILE:P", "Art.9(2)": "X:AIPROFILE:P G:OE-AI-1:P", "Art.9(3)": "NA",
    "Art.9(4)": "", "Art.9(5)": "G:DO-AI-1:P IE:P", "Art.9(6)": "IV:F G:TE-AI-1:P", "Art.9(7)": "NA",
    "Art.9(8)": "X:AIPROFILE:F G:DS-AI-2:F IV:F", "Art.9(9)": "IB:P", "Art.9(10)": "NA",
    "Art.10(1)": "ID:F IV:P", "Art.10(2)(a)": "G:DS-AI-1:P", "Art.10(2)(b)": "ID:F", "Art.10(2)(c)": "ID:F",
    "Art.10(2)(d)": "ID:P IV:P", "Art.10(2)(e)": "ID:P IV:P", "Art.10(2)(f)": "IB:F", "Art.10(2)(g)": "IB:F",
    "Art.10(2)(h)": "ID:P", "Art.10(3)": "ID:P IB:P IV:P", "Art.10(4)": "IV:P G:DS-AI-1:P", "Art.10(6)": "ID:P",
    "Art.11(1)": "G:DO-1:P G:DO-AI-1:F", "Art.11(2)": "NA", "Art.11(3)": "NA",
    "Art.12(1)": "NR:F", "Art.12(2)": "NR:P IE:P G:OE-AI-1:F", "Art.12(3)": "NR:P IO:P",
    "Art.13(1)": "IE:F", "Art.13(2)": "G:DO-1:P IE:F", "Art.13(3)(a)": "G:DO-1:P IE:P",
    "Art.13(3)(b)": "IE:F IV:P IR:P", "Art.13(3)(c)": "G:OE-AI-3:F", "Art.13(3)(d)": "IO:F IE:P",
    "Art.13(3)(e)": "G:DO-1:P G:DO-AI-1:P", "Art.13(3)(f)": "NR:P G:DO-AI-1:P",
    "Art.14(1)": "IO:F", "Art.14(2)": "IO:P", "Art.14(3)": "IO:F IA:P", "Art.14(4)(a)": "IE:F IV:P IO:P",
    "Art.14(4)(b)": "IO:F", "Art.14(4)(c)": "IE:F", "Art.14(4)(d)": "IO:F", "Art.14(4)(e)": "IO:F", "Art.14(5)": "IO:F",
    "Art.15(1)": "NC:P DS:P IV:F IR:F", "Art.15(2)": "NA", "Art.15(3)": "IV:F IE:F",
    "Art.15(4)": "DS:P DV:P IV:P ID:P G:AR-AI-2:P", "Art.15(5)": "KD:P KA:P CA:P NC:P ID:F IR:F IP:F IM:F",
}

ISO_42001_ANNEX_A = {
    "A.2.2": "", "A.2.3": "", "A.2.4": "",
    "A.3.2": "NO:P IO:P", "A.3.3": "",
    "A.4.2": "IM:P G:DO-AI-1:P", "A.4.3": "ID:F", "A.4.4": "IM:P G:DE-AI-2:P", "A.4.5": "G:DE-AI-4:P", "A.4.6": "",
    "A.5.2": "X:AIPROFILE:P", "A.5.3": "G:DO-AI-1:P", "A.5.4": "IB:P", "A.5.5": "",
    "A.6.1.2": "G:DS-AI-1:P", "A.6.1.3": "G:DE-1:P G:DE-AI-1:F", "A.6.2.2": "G:DS-1:P G:DS-AI-1:F",
    "A.6.2.3": "G:DS-3:P G:DS-4:P G:DO-AI-1:P", "A.6.2.4": "G:TE-1:P IV:F G:TE-AI-1:F", "A.6.2.5": "G:OE-1:P G:OE-AI-2:F",
    "A.6.2.6": "NR:P G:OE-AI-1:F IV:F", "A.6.2.7": "G:DO-1:P G:DO-AI-1:F", "A.6.2.8": "NR:F IA:P",
    "A.7.2": "ID:F", "A.7.3": "ID:F", "A.7.4": "ID:P IB:P IV:P", "A.7.5": "ID:F", "A.7.6": "ID:F",
    "A.8.2": "G:DO-1:P IE:F", "A.8.3": "", "A.8.4": "G:OE-AI-1:P", "A.8.5": "IE:P",
    "A.9.2": "IO:P", "A.9.3": "", "A.9.4": "G:DS-AI-1:F IE:F",
    "A.10.2": "", "A.10.3": "IM:F", "A.10.4": "IE:P",
}
ISO_42001_TITLES = {
    "A.2.2": "AI policy", "A.2.3": "Alignment with other organizational policies", "A.2.4": "Review of the AI policy",
    "A.3.2": "AI roles and responsibilities", "A.3.3": "Reporting of concerns",
    "A.4.2": "Resource documentation", "A.4.3": "Data resources", "A.4.4": "Tooling resources",
    "A.4.5": "System and computing resources", "A.4.6": "Human resources",
    "A.5.2": "AI system impact assessment process", "A.5.3": "Documentation of AI system impact assessments",
    "A.5.4": "Assessing AI system impact on individuals or groups of individuals", "A.5.5": "Assessing societal impacts of AI systems",
    "A.6.1.2": "Objectives for responsible development of AI system", "A.6.1.3": "Processes for responsible AI system design and development",
    "A.6.2.2": "AI system requirements and specification", "A.6.2.3": "Documentation of AI system design and development",
    "A.6.2.4": "AI system verification and validation", "A.6.2.5": "AI system deployment", "A.6.2.6": "AI system operation and monitoring",
    "A.6.2.7": "AI system technical documentation", "A.6.2.8": "AI system recording of event logs",
    "A.7.2": "Data for development and enhancement of AI system", "A.7.3": "Acquisition of data", "A.7.4": "Quality of data for AI systems",
    "A.7.5": "Data provenance", "A.7.6": "Data preparation",
    "A.8.2": "System documentation and information for users", "A.8.3": "External reporting", "A.8.4": "Communication of incidents",
    "A.8.5": "Information for interested parties",
    "A.9.2": "Processes for responsible use of AI systems", "A.9.3": "Objectives for responsible use of AI system", "A.9.4": "Intended use of the AI system",
    "A.10.2": "Allocating responsibilities", "A.10.3": "Suppliers", "A.10.4": "Customers",
}

OWASP_LLM_2025 = {
    "LLM01": "IR:F IA:F", "LLM02": "KD:P KA:P IP:F", "LLM03": "CV:P IM:F", "LLM04": "ID:F IM:F",
    "LLM05": "IA:P IR:P", "LLM06": "IA:F IO:P", "LLM07": "IP:P", "LLM08": "KA:P ID:P IP:P", "LLM09": "IV:F IE:P", "LLM10": "DR:F IA:P",
}
OWASP_TITLES = {
    "LLM01": "Prompt Injection", "LLM02": "Sensitive Information Disclosure", "LLM03": "Supply Chain", "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling", "LLM06": "Excessive Agency", "LLM07": "System Prompt Leakage",
    "LLM08": "Vector and Embedding Weaknesses", "LLM09": "Misinformation", "LLM10": "Unbounded Consumption",
}

CSF2 = {  # unmapped subcategories default to '' (not covered)
    "GV.SC-04": "IM:P", "GV.SC-05": "IM:P", "GV.SC-06": "IM:P", "GV.SC-07": "IM:P",
    "ID.AM-02": "IM:P", "ID.AM-07": "ID:F", "ID.AM-08": "G:DE-1:P G:DE-AI-1:P",
    "ID.RA-01": "G:TE-3:P G:TE-4:P IM:P G:TE-AI-3:F", "ID.RA-03": "G:DS-AI-1:P", "ID.RA-04": "X:AIPROFILE:P",
    "ID.RA-07": "G:DE-4:P G:OE-AI-3:F", "ID.RA-09": "CV:P G:OE-2:P IM:F", "ID.IM-02": "G:TE-3:P G:TE-AI-2:P",
    "PR.AA-01": "NI:F", "PR.AA-02": "NI:P", "PR.AA-03": "NI:F NV:F", "PR.AA-04": "NK:P NV:P",
    "PR.AA-05": "KD:F KA:F NO:F IA:P", "PR.AT-01": "IO:P", "PR.AT-02": "IO:P",
    "PR.DS-01": "KA:F CA:F KD:P CD:P ID:P", "PR.DS-02": "KV:F CV:F", "PR.DS-10": "KO:P IP:P", "PR.DS-11": "CO:P DV:P",
    "PR.PS-01": "G:DE-4:F G:DE-AI-2:F", "PR.PS-02": "DZ:P G:OE-AI-3:P", "PR.PS-03": "DZ:P", "PR.PS-04": "NR:F",
    "PR.PS-05": "NC:P IM:F", "PR.PS-06": "G:DE-1:P G:DE-2:F G:DE-AI-1:F",
    "PR.IR-01": "KA:P NV:P", "PR.IR-03": "DS:F DV:F", "PR.IR-04": "DR:F",
    "DE.CM-01": "NR:P", "DE.CM-03": "NR:F", "DE.CM-06": "IM:P", "DE.CM-09": "NR:P NT:P NC:P IV:F IR:P",
    "DE.AE-02": "NR:P", "DE.AE-03": "NR:P", "DE.AE-06": "NR:P IO:P",
    "RS.AN-06": "NR:P", "RS.AN-07": "NR:P ID:P", "RS.MI-01": "NR:P IA:P IO:P",
    "RC.RP-03": "CO:P", "RC.RP-05": "DV:P NC:P",
}


def parse(s: str):
    """'IR:F G:TE-AI-3:F' -> {'IR': 1.0, 'G:TE-AI-3': 1.0}; 'NA' -> None."""
    if s.strip() == "NA":
        return None
    out = {}
    for tok in s.split():
        tgt, st = tok.rsplit(":", 1)
        out[tgt] = 1.0 if st == "F" else 0.5
    return out
