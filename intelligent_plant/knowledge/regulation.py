"""Regulatory rules engine driven by configuration (Chapter 14) plus
change detection over regulatory text (Chapter 12.3).

Boundary: a model's summary of a regulation is a LEAD, never a legal
conclusion. Every interpretation that would drive an operational decision is
routed to counsel through the approval queue."""
from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path


def load_jurisdictions(path: str | Path | None = None) -> dict:
    if path:
        return json.loads(Path(path).read_text())
    return json.loads(resources.files("intelligent_plant.config").joinpath("jurisdictions.json").read_text())


@dataclass
class HempVerdict:
    compliant: bool
    reasons: list[str]
    rule_basis: str


def hemp_rule_on(config: dict, on: date) -> dict:
    rules = config["jurisdictions"]["US-FEDERAL"]["hemp"]["rules_by_effective_date"]
    merged: dict = {}
    for r in sorted(rules, key=lambda r: r["effective"]):
        if date.fromisoformat(r["effective"]) <= on:
            merged.update({k: v for k, v in r.items() if v is not None or k not in merged})
    return merged


def check_hemp_product(config: dict, on: date, delta9_pct: float, total_thc_pct: float,
                       container_total_thc_mg: float | None, contains_synthesized: bool) -> HempVerdict:
    rule = hemp_rule_on(config, on)
    reasons = []
    level = total_thc_pct if rule.get("basis") == "total_thc" else delta9_pct
    if level is None or math.isnan(level):
        reasons.append(f"{rule.get('basis')} not measured; compliance cannot be shown.")
    elif level > rule.get("max_pct", 0.3):
        reasons.append(f"{rule.get('basis')} {level:.3f}% exceeds {rule.get('max_pct')}% dry weight.")
    cap = rule.get("per_container_total_thc_mg")
    if cap is not None:   # unknown or unmeasurable totals fail closed once a cap applies
        if container_total_thc_mg is None or math.isnan(container_total_thc_mg):
            reasons.append(f"Per-container total THC unknown; the {cap} mg cap applies.")
        elif container_total_thc_mg > cap:
            reasons.append(f"{container_total_thc_mg} mg total THC per container exceeds {cap} mg cap.")
    if rule.get("exclude_non_naturally_produced_cannabinoids") and contains_synthesized:
        reasons.append("Contains cannabinoids synthesized outside the plant or not naturally produced.")
    return HempVerdict(not reasons, reasons, f"rules effective on {on.isoformat()}: {rule}")


def federal_schedule(config: dict, category: str) -> str:
    return config["jurisdictions"]["US-FEDERAL"]["categories"][category]["schedule"]


def subject_to_280e(config: dict, category: str) -> bool:
    fed = config["jurisdictions"]["US-FEDERAL"]
    return fed["categories"][category]["schedule"] in fed["irc_280e_applies_to_schedules"]


@dataclass
class RegulatoryChange:
    source: str
    added: list[str]
    removed: list[str]
    operational_areas: list[str]
    status: str = "lead_requires_primary_source_verification"


AREA_KEYWORDS = {
    "testing": ("test", "laboratory", "potency", "contaminant", "pesticide", "coa"),
    "packaging_labeling": ("label", "packag", "child-resistant", "warning"),
    "cultivation": ("plant", "canopy", "cultivat", "harvest"),
    "tracking": ("track", "trace", "inventory", "manifest", "transfer"),
    "taxation": ("tax", "280e", "excise"),
    "scheduling": ("schedule", "dea", "registration"),
    "hemp": ("hemp", "total thc", "per container"),
}


def diff_regulation(source: str, old_text: str, new_text: str) -> RegulatoryChange:
    o, n = old_text.splitlines(), new_text.splitlines()
    added, removed = [], []
    for line in difflib.unified_diff(o, n, lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].strip())
    text = " ".join(added + removed).lower()
    areas = [a for a, kws in AREA_KEYWORDS.items() if any(k in text for k in kws)]
    return RegulatoryChange(source, added, removed, areas)
