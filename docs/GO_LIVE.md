# Going live: the three human steps

The software is ready. These steps need people. Production mode (`mode="production"`) refuses to start until
every step below is signed off **against the exact version that was reviewed**. If anything reviewed is later
edited, its sign-off goes stale and must be repeated.

Check progress at any time:

```bash
intelligent-plant readiness --signoffs signoffs.json --calibration facility_calibration.json
```

---

## Step 1: Patient message review (clinical lead, then patients)

```bash
intelligent-plant review-pack --out review/
```

This writes two packets:

- **`patient_message_review_clinical.md`** for the clinical lead. Every message exactly as a patient sees
  it, with its reading level and review questions. Wording changes go into
  `intelligent_plant/config/patient_messages.json` (not code).
- **`patient_message_review_patients.md`** for a **teach-back** session with at least 5 patients, including
  older readers and readers with limited literacy. A message passes when at least 4 of 5 readers state its
  meaning and the right next step. Revise any that fail, then re-run both reviews on the new wording.

Sign-offs: `patient_messages_clinical` (role `clinical_lead`) and `patient_messages_usability`
(role `patient_advisory`).

## Step 2: Calibrate to your facility (facility engineer and head grower)

Export three CSV files. Templates with the exact columns are in `docs/calibration_templates/`.

| File | What | How much |
|---|---|---|
| `room_log.csv` | Each control step: temperature, RH, which equipment was on (0, 0.5 or 1), and the next reading | At least 200 steps of normal operation, during which cooling, dehumidifying and humidifying each cycled on and off |
| `volatile_retention.csv` | GC-measured terpene retention after heating | At least 3 temperatures (e.g. 95, 105, 115 °C), replicates welcome |
| `nutrients.csv` | Per batch: stage, mean EC, mean pH, whether it met quality targets (1/0) | At least 20 good batches per stage to check the SOP |

Plus `sop_ranges.json`: the head grower's EC/pH ranges per stage, which stay authoritative.

```bash
intelligent-plant calibrate --facility "Your Facility" \
    --room-log room_log.csv --volatile volatile_retention.csv \
    --nutrients nutrients.csv --sop sop_ranges.json --out facility_calibration.json
```

The tool **refuses** rather than guesses when the data cannot support a fit: too few rows, equipment that
never cycled, a stuck sensor, cooling that appears to raise temperature (often swapped columns), or fewer
than 3 temperatures. Fix the data; do not loosen the checks. Rows outside physical ranges are rejected and
listed.

Sign-offs: `facility_calibration` (role `facility_engineer`) after reading the fit report, and
`nutrient_ranges` (role `head_grower`).

## Step 3: Regulatory, legal and privacy review (counsel and privacy officer)

Give counsel and the privacy officer **`docs/REGULATORY_REVIEW.md`**. It describes what the system does, every
data element it holds, 17 questions that need written answers (DEA/state licensing, FDA device status of
patient-facing messages, HIPAA, 42 CFR Part 2, genetic privacy, TCPA consent, health claims) and the known
technical gaps (encryption at rest, authentication, backups, breach response).

Sign-offs: `regulatory_legal` (role `regulatory_counsel`) and `privacy_security` (role `privacy_officer`).

## Starting in production

```bash
export INTELLIGENT_PLANT_VAULT_KEY=...   # from your secrets manager; 64+ hex characters; never commit it
```

```python
from intelligent_plant.orchestrator import IntelligentEnterprise
ent = IntelligentEnterprise(
    mode="production",
    db_path="/secure/encrypted/records.db",       # on an encrypted volume
    vault_path="/secure/encrypted/vault.db",      # separate from records
    calibration_path="facility_calibration.json",
    signoffs_path="signoffs.json",
    regulatory_review_path="docs/REGULATORY_REVIEW.md",
)
```

If anything is missing, it raises `ProductionNotReady` and lists exactly what.
