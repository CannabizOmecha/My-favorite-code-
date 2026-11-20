"""Build the review packets that people (clinicians and patients) fill in.

* Clinician packet: every patient message as the patient will see it, its urgency, reading grade and the
  review questions, with space for required changes.
* Patient packet: the teach-back method (health-literacy best practice). Patients read each message and say
  in their own words what it means and what they would do. That tests understanding, not just approval.

Both packets print the fingerprint of the wording under review, which is what the sign-off binds to.
"""
from __future__ import annotations

from pathlib import Path

from .governance.readiness import artifact_fingerprint, default_artifacts
from .patient.communication import CATALOG, PatientMessenger, progress_summary, reading_grade

SAMPLE_PAYLOAD = {
    "adverse_effects": [{"name": "dizziness"}],
    "missing": ["score_post", "route"],
    "summary": progress_summary([{"score_pre": 7, "score_post": 5}] * 10, "sleep through the night"),
}

EVENT_CONTEXT = {
    "oral_redose": "Sent when a patient logs a second dose by mouth too soon after the first.",
    "adverse_effect": "Sent when a patient logs a moderate or severe side effect.",
    "multiple_changes": "Sent when the product, amount and route changed at the same time.",
    "missing_log_fields": "Sent when recent log entries are missing details.",
    "medication_check": "Sent when a newly listed medicine can interact with cannabis products.",
    "progress_summary": "Weekly summary built from the patient's own log (example shown).",
}


def build(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp = artifact_fingerprint(default_artifacts()["patient_messages"])
    m = PatientMessenger()
    rows = [(k, *m.compose(k, SAMPLE_PAYLOAD)) for k in CATALOG["messages"]]

    clin = [f"# Patient message review: clinical lead",
            "",
            f"Catalog version {CATALOG['version']} - fingerprint `{fp[:16]}`. Status: {CATALOG['status']}.",
            "",
            "Edit wording in `intelligent_plant/config/patient_messages.json`. Any edit changes the fingerprint,",
            "so this review must be repeated for the new wording.",
            "",
            "## Questions for every message",
            *[f"{i}. {q}" for i, q in enumerate(CATALOG["review_questions"], 1)],
            ""]
    for k, urgency, text in rows:
        clin += [f"## `{k}` ({urgency})",
                 f"*When:* {EVENT_CONTEXT.get(k, '')}  ",
                 f"*Reading grade:* {reading_grade(text):.1f} (limit {CATALOG['reading_grade_max']})",
                 "",
                 "> " + text,
                 "",
                 "- [ ] Approve as written    - [ ] Approve with changes    - [ ] Reject",
                 "",
                 "Required changes: ______________________________________________",
                 ""]
    clin += ["## Sign-off", "",
             "After the review, record it (binds your name to this exact wording):", "",
             "```bash",
             'intelligent-plant signoff patient_messages_clinical --name "<name>" --role clinical_lead \\',
             '    --statement "Reviewed every message in catalog version '
             f'{CATALOG["version"]} for clinical accuracy and safety."',
             "```", ""]

    pat = ["# Help us make our messages clear",
           "",
           "Thank you for helping. There are no wrong answers: if something is unclear, that is our mistake to fix.",
           "",
           "For each message, imagine your phone just showed it to you.",
           "",
           f"(Version {CATALOG['version']}, code `{fp[:8]}`)",
           ""]
    for i, (k, urgency, text) in enumerate(rows, 1):
        pat += [f"## Message {i}", "", "> " + text, "",
                "1. In your own words, what is this message telling you?", "", "   ____________________________", "",
                "2. What would you do right after reading it?", "", "   ____________________________", "",
                "3. Was any word or sentence confusing? Which one?", "", "   ____________________________", "",
                "4. How did it make you feel?  ( ) Calm  ( ) Worried  ( ) Confused  ( ) Other: ______", ""]
    pat += ["## For the study coordinator", "",
            "Teach-back scoring: a message passes when at least 4 of 5 readers state its meaning AND the right",
            "next step. Include older readers and readers with limited literacy. Record results, then:", "",
            "```bash",
            'intelligent-plant signoff patient_messages_usability --name "<coordinator>" --role patient_advisory \\',
            '    --statement "Teach-back with N patients; every message passed (results at <location>)."',
            "```", ""]

    paths = {"clinical": out / "patient_message_review_clinical.md",
             "patients": out / "patient_message_review_patients.md"}
    paths["clinical"].write_text("\n".join(clin))
    paths["patients"].write_text("\n".join(pat))
    return paths
