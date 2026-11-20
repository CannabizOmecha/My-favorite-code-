# CLAUDE.md — The Intelligent Plant

## North star

**The goal of this system is to improve the experience and outcomes of patients.** Everything upstream (genetics,
cultivation, processing, compliance, learning, quantum optimisation) exists to give the person at the end of the
supply chain a consistent, accurately labelled product, safe guidance, and a care team that has the information it
needs. When choosing what to build or how to build it, ask first: *what does this change for the patient?*

The manuscript's ordering still governs trade-offs: **Safety > Ethics > Accuracy > Efficiency > Profitability.**
Patient experience lives inside it. Dignity, plain language, respect for preferences and freedom from alert fatigue
are ethical obligations. They never override a safety rule.

## Rules that protect patients (tests enforce these; do not weaken them)

- Patient-facing text goes through `patient/communication.py`: reading grade ≤ 8, no internal IDs or jargon,
  "what to do now", and emergency guidance on every urgent message (`tests/test_patient_experience.py`).
- Urgent safety messages are always delivered. Routine messages are de-duplicated, capped per day and held during
  quiet hours.
- Patient preferences (psychoactivity, route, goals) shape recommendations within safety limits. A preference can
  narrow options; it can never widen them past a safety rule. Disagreements are surfaced for discussion, never
  silently overridden.
- Patients see their own progress, honestly, including "too little data to be sure" and "little change".
- Agents never originate dose numbers; dose selection and titration steps belong to the treating clinician.
- Hard interaction rules (Table 6.1) are deterministic and cannot be overridden by any learned model.
- Models learn only from measured outcomes, abstain when degraded, and disclose uncertainty.
- Genetic and behavioural data are for harm reduction, never gatekeeping or stigma (Ch 4.5).

## Working in this repo

- CI (`.github/workflows/tests.yml`) runs the full suite on Python 3.10–3.13 plus a build-and-install check on every
  pull request. Never skip, disable or weaken a test to make CI green; fix the cause.
- `pytest` must pass. Book worked examples are regression tests; change them only if the book's math was wrong, and
  say why.
- `python -m intelligent_plant demo` shows a full cycle, including "WHAT THE PATIENT SEES".
- Jurisdiction rules live in `intelligent_plant/config/jurisdictions.json`, not code.
- `simulation.py` is synthetic; never present its numbers as empirical.
