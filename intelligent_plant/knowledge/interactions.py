"""Hard, rule-based interaction and contraindication checks (Table 6.1,
Chapter 13.1, Appendix C.1). These are deterministic by design: no learned
model may override them (Chapter 13.4)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InteractionRule:
    rule_id: str
    triggers: tuple[str, ...]          # lower-case drug names / class tokens
    cannabinoid: str                   # "CBD", "THC", "any"
    mechanism: str
    direction: str
    action: str
    severity: str                      # "major" | "moderate"
    refs: tuple[int, ...]


RULES: tuple[InteractionRule, ...] = (
    InteractionRule("DDI-CLOBAZAM", ("clobazam",), "CBD", "CBD inhibits CYP2C19",
                    "~3x increase in N-desmethylclobazam", "Monitor sedation; clobazam dose reduction often needed "
                    "(prescriber decision).", "major", (93, 27)),
    InteractionRule("DDI-VALPROATE", ("valproate", "valproic acid", "divalproex"), "CBD",
                    "Not primarily PK", "Increased risk of transaminase elevation",
                    "Check LFTs before and during treatment.", "major", (27,)),
    InteractionRule("DDI-WARFARIN", ("warfarin",), "any", "CBD/THC inhibit CYP2C9", "Case reports of raised INR",
                    "Increase INR monitoring frequency.", "major", (102,)),
    InteractionRule("DDI-TACROLIMUS", ("tacrolimus", "sirolimus", "everolimus", "cyclosporine"), "CBD",
                    "CYP3A4", "Raised exposure reported", "Therapeutic drug monitoring.", "major", (103, 27)),
    InteractionRule("DDI-CYP-MODULATOR",
                    ("ketoconazole", "itraconazole", "clarithromycin", "ritonavir", "fluconazole", "fluvoxamine",
                     "rifampin", "rifampicin", "carbamazepine", "phenytoin", "st johns wort"),
                    "CBD", "Strong CYP3A4/CYP2C19 inhibitor or inducer alters CBD clearance", "Both directions",
                    "Consider CBD dose adjustment (prescriber decision).", "moderate", (27,)),
    InteractionRule("DDI-CNS-DEPRESSANT",
                    ("alcohol", "opioid", "oxycodone", "morphine", "hydrocodone", "fentanyl", "benzodiazepine",
                     "diazepam", "lorazepam", "alprazolam", "clonazepam", "zolpidem", "gabapentin", "pregabalin",
                     "codeine", "tramadol", "methadone", "buprenorphine", "hydromorphone", "tapentadol"),
                    "any", "Pharmacodynamic", "Additive sedation and psychomotor impairment",
                    "Counsel explicitly; driving risk.", "moderate", (27,)),
    InteractionRule("DDI-EPIDIOLEX-SUBSTRATES",
                    ("lamotrigine", "morphine", "lorazepam", "phenytoin", "celecoxib", "repaglinide",
                     "omeprazole", "citalopram", "clopidogrel"),
                    "CBD", "CBD may inhibit UGT1A9/UGT2B7/CYP2C8/CYP2C9/CYP2C19",
                    "Possible increased substrate exposure", "Consider substrate dose reduction per label.",
                    "moderate", (27,)),
)

# Narrow-therapeutic-index CYP2C9/2C19 substrates needing special attention (Chapter 13.1 step 3).
NARROW_TI = {"warfarin", "phenytoin", "clobazam", "tacrolimus", "valproate", "valproic acid", "divalproex",
             "cyclosporine", "sirolimus", "everolimus"}

# Spelling variants and brand names -> the generic token the rules use. Matching is
# deterministic; an unrecognised name is never evidence of safety (POLYPHARMACY note).
DRUG_ALIASES = {
    "ciclosporin": "cyclosporine", "cyclosporin": "cyclosporine", "coumadin": "warfarin", "jantoven": "warfarin",
    "depakote": "divalproex", "depakene": "valproic acid", "onfi": "clobazam", "sympazan": "clobazam",
    "prograf": "tacrolimus", "dilantin": "phenytoin", "tegretol": "carbamazepine", "hypericum": "st johns wort",
}

# Cannabinoid names -> rule tokens. Unrecognised or empty input FAILS CLOSED (both checked).
CANNABINOID_ALIASES = {"CBD": {"CBD"}, "CANNABIDIOL": {"CBD"}, "EPIDIOLEX": {"CBD"},
                       "THC": {"THC"}, "DELTA-9-THC": {"THC"}, "DELTA 9 THC": {"THC"}, "D9-THC": {"THC"},
                       "TETRAHYDROCANNABINOL": {"THC"}, "DRONABINOL": {"THC"}, "NABILONE": {"THC"},
                       "NABIXIMOLS": {"THC", "CBD"}, "SATIVEX": {"THC", "CBD"}}


@dataclass
class InteractionAlert:
    rule_id: str
    drug: str
    severity: str
    message: str
    action: str
    refs: tuple[int, ...]
    overridable_by_model: bool = False       # always False: hard rule


def _norm(s: str) -> str:
    n = " ".join(s.lower().replace("\u2019", "").replace("'", "").replace(".", " ").replace("-", " ").split())
    for alias, generic in DRUG_ALIASES.items():
        if alias in n:
            n = f"{n} ({generic})"     # keep what the patient wrote, add the generic token for matching
    return n


def _cannabinoids(cannabinoids: set[str]) -> set[str]:
    can: set[str] = set()
    for c in cannabinoids:
        can |= CANNABINOID_ALIASES.get(str(c).strip().upper(), set())
    return can if can else {"CBD", "THC"}      # unknown or empty -> assume both (fail closed)


def check_interactions(medications: list[str], cannabinoids: set[str]) -> list[InteractionAlert]:
    """Deterministic interaction check. `cannabinoids` e.g. {"CBD","THC"}."""
    meds = [_norm(m) for m in medications]
    can = _cannabinoids(cannabinoids)
    alerts: list[InteractionAlert] = []
    seen = set()
    for rule in RULES:
        if rule.cannabinoid != "any" and rule.cannabinoid not in can:
            continue
        for med in meds:
            for trig in rule.triggers:
                if trig in med and (rule.rule_id, med) not in seen:
                    seen.add((rule.rule_id, med))
                    alerts.append(InteractionAlert(
                        rule.rule_id, med, rule.severity,
                        f"{med}: {rule.mechanism} -> {rule.direction}.", rule.action, rule.refs))
    if len(meds) >= 5:
        alerts.append(InteractionAlert("POLYPHARMACY", ",".join(meds), "moderate",
                                       f"{len(meds)} concurrent medications: most combinations never studied.",
                                       "Full medication review; absence of a documented interaction is not "
                                       "evidence of safety.", (27,)))
    order = {"major": 0, "moderate": 1}
    return sorted(alerts, key=lambda a: order.get(a.severity, 2))


@dataclass
class PatientRiskProfile:
    age: int | None
    psychosis_history_personal: bool = False
    psychosis_history_family_strong: bool = False
    unstable_cardiovascular: bool = False
    pregnant_or_lactating: bool = False
    substance_use_disorder_history: bool = False
    medications: list[str] = field(default_factory=list)
    cyp2c9_genotype: str | None = None


@dataclass
class ScreenFlag:
    code: str
    level: str          # "stop" (hard: clinician must address), "caution", "consider"
    message: str
    refs: tuple[int, ...]


def screen_contraindications(p: PatientRiskProfile) -> list[ScreenFlag]:
    """Appendix C.1. Age <25 is a risk consideration, not a categorical rule."""
    f: list[ScreenFlag] = []
    if p.age is None:   # cannot rule out a paediatric patient: fail closed, keep screening the rest
        f.append(ScreenFlag("AGE_UNKNOWN", "stop", "Age not recorded: confirm age before any recommendation.", (110,)))
    elif p.age < 18:
        f.append(ScreenFlag("PEDIATRIC", "stop", "Pediatric patient: outside this tool's validated scope.", (110,)))
    if p.pregnant_or_lactating:
        f.append(ScreenFlag("PREGNANCY", "stop", "Pregnancy/lactation: professional bodies advise against use.", (109,)))
    if p.psychosis_history_personal:
        f.append(ScreenFlag("PSYCHOSIS", "stop", "Personal history of psychotic illness.", (104,)))
    if p.psychosis_history_family_strong:
        f.append(ScreenFlag("PSYCHOSIS_FAMILY", "caution", "Strong family history of psychosis: avoid high-THC "
                            "products; harm reduction, not gatekeeping.", (104,)))
    if p.unstable_cardiovascular:
        f.append(ScreenFlag("CARDIO", "stop", "Unstable cardiovascular disease.", (105,)))
    if p.substance_use_disorder_history:
        f.append(ScreenFlag("SUD", "caution", "Substance use disorder history; CUD risk ~22%.", (106,)))
    if p.age is not None and 18 <= p.age < 25:
        f.append(ScreenFlag("AGE_UNDER_25", "consider", "Age under 25: weigh developmental risk against the "
                            "indication.", (20,)))
    if any(_norm(m) in NARROW_TI or any(n in _norm(m) for n in NARROW_TI) for m in p.medications):
        f.append(ScreenFlag("NARROW_TI", "caution", "Narrow-therapeutic-index CYP2C9/2C19 substrate on board.", (27,)))
    return f
