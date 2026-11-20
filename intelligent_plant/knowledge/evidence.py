"""Evidence vocabulary (Table P.1) and the indication evidence map (Chapter 5,
NASEM 2017). Every clinical output links to an EvidenceRef."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Grade(IntEnum):
    EVIDENCE_AGAINST = -1
    PRECLINICAL = 0
    INSUFFICIENT = 1
    LIMITED = 2
    MODERATE = 3
    SUBSTANTIAL = 4
    CONCLUSIVE = 5


GRADE_MEANING = {
    Grade.CONCLUSIVE: "Strong evidence; many supportive findings, no credible opposing findings",
    Grade.SUBSTANTIAL: "Strong evidence; few or no credible opposing findings",
    Grade.MODERATE: "Some evidence; several supportive findings, a few opposing",
    Grade.LIMITED: "Weak evidence; supportive findings from fair-quality studies or mixed findings",
    Grade.INSUFFICIENT: "No or insufficient evidence to support or refute",
    Grade.PRECLINICAL: "Mechanistic, in vitro or animal data only",
    Grade.EVIDENCE_AGAINST: "Well-designed evidence against the claim",
}


@dataclass(frozen=True)
class EvidenceRef:
    claim: str
    grade: Grade
    refs: tuple[int, ...]
    note: str = ""

    def cite(self) -> str:
        r = ",".join(map(str, self.refs))
        return f"{self.claim} [{self.grade.name.lower()}; refs {r}]{' - ' + self.note if self.note else ''}"


# Indication -> evidence. Keys are normalised indication codes.
INDICATIONS: dict[str, EvidenceRef] = {
    "chronic_pain_adult": EvidenceRef(
        "Cannabis for chronic pain in adults", Grade.SUBSTANTIAL, (20, 13, 97),
        "Benefit is SMALL on average (BMJ 2021: 32 RCTs, n=5,174); weak recommendation for non-inhaled "
        "products after standard care fails."),
    "cinv": EvidenceRef("Oral cannabinoids for chemotherapy-induced nausea and vomiting",
                        Grade.CONCLUSIVE, (20,)),
    "ms_spasticity": EvidenceRef("Oral cannabinoids for patient-reported MS spasticity",
                                 Grade.SUBSTANTIAL, (20,)),
    "dravet": EvidenceRef("Purified CBD (FDA-approved) for Dravet syndrome seizures", Grade.CONCLUSIVE,
                          (9,), "Specialist-managed, FDA-approved product only; outside this tool's scope."),
    "lennox_gastaut": EvidenceRef("Purified CBD (FDA-approved) for LGS drop seizures", Grade.CONCLUSIVE,
                                  (10, 11), "Specialist-managed, FDA-approved product only."),
    "tsc_seizures": EvidenceRef("Purified CBD (FDA-approved) for TSC-associated seizures",
                                Grade.CONCLUSIVE, (12,), "Specialist-managed, FDA-approved product only."),
    "ptsd": EvidenceRef("Smoked cannabis for PTSD", Grade.LIMITED, (14,),
                        "Well-run RCT (n=80): no preparation beat placebo."),
}

# Indications handled only by specialists with FDA-approved product: the tool refuses, loudly.
SPECIALIST_ONLY = {"dravet", "lennox_gastaut", "tsc_seizures"}

ENTOURAGE_CLAIMS = {
    "limonene_reduces_thc_anxiety": EvidenceRef("d-Limonene reduces THC-induced anxiety (vaporized, acute)",
                                                Grade.LIMITED, (116,), "Well designed factorial trial, n=20."),
    "bcp_cb2_agonist": EvidenceRef("beta-Caryophyllene is a functional CB2 agonist", Grade.PRECLINICAL, (117,)),
    "cbd_attenuates_thc": EvidenceRef("CBD attenuates THC cognitive/psychotic effects",
                                      Grade.EVIDENCE_AGAINST, (15,), "RCT across 4 ratios, n=46: no effect."),
    "terpenes_modulate_cb_receptors": EvidenceRef("Terpenes broadly modulate CB1/CB2 at consumed concentrations",
                                                  Grade.EVIDENCE_AGAINST, (118,)),
    "whole_plant_superior": EvidenceRef("Whole-plant extracts outperform matched isolates in humans",
                                        Grade.INSUFFICIENT, ()),
}

HARMS = {
    "cud": EvidenceRef("Cannabis use disorder ~22% (18-26%) of people who use; ~33% young weekly/daily users",
                       Grade.SUBSTANTIAL, (106,)),
    "psychosis_daily_high_potency": EvidenceRef(
        "Daily high-potency (THC>=10%) use: adjusted OR ~4.8 for first-episode psychosis", Grade.MODERATE, (104,),
        "Case-control; causality not established by these data alone."),
    "cardiovascular_daily": EvidenceRef("Daily use: aOR 1.25 MI, 1.42 stroke (cross-sectional, self-report)",
                                        Grade.LIMITED, (105,)),
    "pregnancy": EvidenceRef("Prenatal exposure associated with lower birth weight; avoid in pregnancy/lactation",
                             Grade.SUBSTANTIAL, (109,)),
    "pediatric_ingestion": EvidenceRef("Pediatric unintentional ingestion is a genuine emergency",
                                       Grade.SUBSTANTIAL, (110,)),
}


def evidence_for(indication: str) -> EvidenceRef:
    return INDICATIONS.get(indication, EvidenceRef(f"Cannabinoids for {indication}", Grade.INSUFFICIENT, (20,),
                                                   "Not in the curated evidence map; treat as insufficient."))
