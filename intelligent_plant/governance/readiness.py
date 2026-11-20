"""Production readiness: the reviews that only people can do, made enforceable.

Each required item names the accountable ROLE and the ARTIFACT that role reviewed. A sign-off records who,
when, and the SHA-256 fingerprint of exactly what they reviewed. If the artifact changes afterwards, the
fingerprint no longer matches and the sign-off is STALE: approving one wording and shipping another is how
reviews silently stop meaning anything.

Software cannot sign these. `record_signoff` is meant to be run by the named person after doing the review.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path


def _package_file(name: str) -> Path:
    return Path(str(resources.files("intelligent_plant.config").joinpath(name)))


def _repo_doc(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / name


@dataclass(frozen=True)
class RequiredReview:
    item: str
    role: str
    artifact: str            # logical name, resolved by `artifacts`
    what_to_check: str


REQUIRED: tuple[RequiredReview, ...] = (
    RequiredReview("patient_messages_clinical", "clinical_lead", "patient_messages",
                   "Every patient message is clinically correct, safe and consistent with your protocols."),
    RequiredReview("patient_messages_usability", "patient_advisory", "patient_messages",
                   "Patients (ideally 5 or more, including low-literacy and older readers) read each message and "
                   "could say what it means and what to do next."),
    RequiredReview("facility_calibration", "facility_engineer", "facility_calibration",
                   "Room model, decarboxylation losses and nutrient ranges were fitted from this facility's data "
                   "and the fit report is acceptable."),
    RequiredReview("nutrient_ranges", "head_grower", "facility_calibration",
                   "The EC/pH ranges match the facility's SOPs for each stage."),
    RequiredReview("regulatory_legal", "regulatory_counsel", "regulatory_review",
                   "Every question in docs/REGULATORY_REVIEW.md has a written answer and the system may be used "
                   "as described (FDA CDS status, DEA/state licensing, TCPA consent, advertising claims)."),
    RequiredReview("privacy_security", "privacy_officer", "regulatory_review",
                   "HIPAA status, BAAs, encryption at rest, access control, backups and breach response are in "
                   "place as described in docs/REGULATORY_REVIEW.md."),
)


def default_artifacts(calibration_path: str | Path | None = None,
                      regulatory_review_path: str | Path | None = None) -> dict[str, Path | None]:
    return {
        "patient_messages": _package_file("patient_messages.json"),
        "facility_calibration": Path(calibration_path) if calibration_path else None,
        "regulatory_review": Path(regulatory_review_path) if regulatory_review_path
        else _repo_doc("REGULATORY_REVIEW.md"),
    }


def artifact_fingerprint(path: Path) -> str:
    """SHA-256 of the reviewed content. For JSON, keys starting with '_' (comments) and 'status' are
    excluded, and keys are canonically ordered, so formatting or a status note does not invalidate a review
    but any change to the substance does."""
    data = path.read_bytes()
    if path.suffix == ".json":
        doc = json.loads(data)
        doc = {k: v for k, v in doc.items() if not k.startswith("_") and k != "status"}
        data = json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


@dataclass
class ItemStatus:
    item: str
    role: str
    status: str              # ok | missing | stale | artifact_missing
    detail: str


@dataclass
class ReadinessReport:
    items: list[ItemStatus] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(i.status == "ok" for i in self.items)

    def summary(self) -> str:
        lines = [f"{'READY' if self.ready else 'NOT READY'} for production use"]
        for i in self.items:
            lines.append(f"  [{i.status.upper():16}] {i.item} ({i.role}): {i.detail}")
        return "\n".join(lines)


def load_signoffs(path: str | Path) -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def check_readiness(signoffs_path: str | Path, artifacts: dict[str, Path | None]) -> ReadinessReport:
    signoffs = load_signoffs(signoffs_path)
    report = ReadinessReport()
    for req in REQUIRED:
        art = artifacts.get(req.artifact)
        if art is None or not Path(art).exists():
            report.items.append(ItemStatus(req.item, req.role, "artifact_missing",
                                           f"{req.artifact} not found; produce it first."))
            continue
        fp = artifact_fingerprint(Path(art))
        mine = [s for s in signoffs if s["item"] == req.item and s["role"] == req.role]
        if not mine:
            report.items.append(ItemStatus(req.item, req.role, "missing", req.what_to_check))
            continue
        latest = mine[-1]
        if latest["sha256"] != fp:
            report.items.append(ItemStatus(req.item, req.role, "stale",
                                           f"{req.artifact} changed after {latest['name']} signed on "
                                           f"{latest['date']}; re-review required."))
        else:
            report.items.append(ItemStatus(req.item, req.role, "ok", f"signed by {latest['name']} on {latest['date']}"))
    return report


def record_signoff(signoffs_path: str | Path, item: str, name: str, role: str, statement: str,
                   artifacts: dict[str, Path | None]) -> dict:
    req = next((r for r in REQUIRED if r.item == item), None)
    if req is None:
        raise ValueError(f"Unknown review item {item!r}; choose from {[r.item for r in REQUIRED]}.")
    if role != req.role:
        raise PermissionError(f"{item} must be signed by the {req.role}, not {role}.")
    if not name.strip() or not statement.strip():
        raise ValueError("A sign-off needs the reviewer's name and a statement of what they checked.")
    art = artifacts.get(req.artifact)
    if art is None or not Path(art).exists():
        raise FileNotFoundError(f"{req.artifact} does not exist yet.")
    rec = {"item": item, "role": role, "name": name.strip(), "statement": statement.strip(),
           "date": datetime.now(timezone.utc).date().isoformat(), "artifact": str(art),
           "sha256": artifact_fingerprint(Path(art))}
    signoffs = load_signoffs(signoffs_path)
    signoffs.append(rec)                     # append-only: earlier sign-offs stay as history
    Path(signoffs_path).write_text(json.dumps(signoffs, indent=2) + "\n")
    return rec
