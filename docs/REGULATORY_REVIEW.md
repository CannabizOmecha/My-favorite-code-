# Regulatory, Legal and Privacy Review Briefing

**For:** regulatory counsel and the privacy officer. **Status:** awaiting review.
**This is not legal advice.** It was prepared by the software's authors to make a qualified reviewer's
work fast and complete. Every question below needs a written answer from counsel before production use.
Production mode (`IntelligentEnterprise(mode="production")`) refuses to start until the `regulatory_legal`
and `privacy_security` sign-offs are recorded against the exact version of this document.

---

## 1. What the system does

The software runs a medical cannabis operation from genetics to patient outcome (based on C. A. Cassinelli,
*The Intelligent Plant*, 2026). Its goal is to improve patients' experience and outcomes.

| Function | Acts on its own? | Who decides |
|---|---|---|
| Climate, irrigation, lighting setpoints; sensor fail-safe | Yes (reversible, with alarms) | Head grower / facility engineer can reverse |
| COA audits, batch **holds**, hemp quarantine | Yes (reversible) | QA manager releases |
| Batch **release**, harvest, breeding, nutrient recipes | No: recommendation only | Named staff role |
| Drug-interaction alerts (rule-based, from the Epidiolex label and literature) | Yes: alert to clinician and pharmacist | Clinician acts |
| **Messages sent directly to patients** (re-dosing warning, side-effect hold, medication check, log reminders, weekly progress) | **Yes, automatically** | Wording reviewed by the clinical lead |
| Product suggestion, titration steps, continue/stop review | No: recommendation to the treating clinician | Treating clinician |
| Dose selection | **Never**: the software cannot originate a dose | Treating clinician |
| Regulatory interpretation | No: flags changes for counsel | Counsel |

## 2. Data inventory

| Data | Where | Identifiable? |
|---|---|---|
| Patient identity (name, DOB) | Separate vault database | Yes: direct identifiers |
| Pseudonymous patient ID (`P-` + HMAC of identity) | Record store | **Yes under HIPAA.** It is derived from identity and re-identifiable with the vault key, so it is not de-identified (45 CFR 164.514). |
| Age band, indication, consent version | Record store | PHI if HIPAA applies |
| Medications, adverse effects, symptom scores, doses taken | Record store / patient app logs | PHI; sensitive |
| CYP2C9 genotype (optional) | Record store | Genetic information: highest sensitivity |
| Psychosis history, substance use disorder history | In-memory screening inputs | Highly sensitive; possibly 42 CFR Part 2 |
| Cultivation, batch, COA, inventory, financial | Record store | Business confidential, not personal |

Records are append-only and hash-chained (tamper-evident). Corrections are new versions with a reason code.

## 3. Questions counsel must answer

### 3.1 Federal controlled substances and licensing
1. Does our operation hold, or need, the DEA registration required for state medical licensees under the April
   2026 order (91 Fed. Reg. 22714)? What recordkeeping does that registration add?
2. The order is under challenge (D.C. Cir. Nos. 26-1106, 26-1130). What is our contingency if it is vacated?
3. Is our tax position (§280E) as configured in `config/jurisdictions.json` correct for our category?

### 3.2 State rules
4. Which state medical program rules apply (in Oklahoma, the OMMA rules)? Which seed-to-sale system must we
   report to, and must this software's inventory records reconcile with it?
5. Do state rules restrict who may give dosing or product guidance to patients, and do our automated patient
   messages fall within those limits?
6. Are our testing panels, labeling and packaging checks complete for our state? (Add them to the config.)

### 3.3 FDA: is any part of this a medical device?
7. Clinician-facing features are designed to meet the four non-device clinical decision support criteria of
   FD&C Act §520(o)(1)(E): no image or signal analysis, medical information only, support for a health care
   professional, and a basis the professional can review independently. Does counsel agree?
8. **Patient-facing messages are not directed to a health care professional**, so they likely fall outside
   those criteria. Are they device software functions? If so, does FDA enforcement discretion apply, or must
   they be restricted, for example by having a clinician release each message? (Check FDA's current CDS
   and general wellness guidance, including any revisions after the 2022 final CDS guidance.)
9. Do the learning models (yield, outcome-by-chemotype) or product suggestions change that analysis?

### 3.4 Privacy and security
10. Are we a HIPAA covered entity or business associate? If not, which state health privacy laws and the
    FTC Health Breach Notification Rule apply to the patient app and logs?
11. Does any record fall under 42 CFR Part 2 (substance use disorder records from a federally assisted
    program)?
12. Genetic data: which laws apply (GINA, plus state genetic privacy laws), and what consent and retention
    terms do we need?
13. **Outside AI services.** The optional Claude review panel is configured so that patient-scoped proposals
    are never sent to it (`phi_to_llm_permitted=False`). If we want to enable that, is a signed BAA with the
    provider required, and is one in place? Do any state AI or health-AI laws apply? For example, California
    requires a disclosure when generative AI writes patient communications. Our patient messages are fixed,
    reviewed templates, not generated text, and must stay that way unless counsel approves otherwise.
14. Retention and deletion: what periods apply to each data class in section 2?

### 3.5 Patient communications
15. **TCPA:** what consent must we obtain before sending automated text messages? Our default quiet hours
    (no routine messages 9 pm–8 am) are intended to align with TCPA calling-time limits. Confirm the time zone
    rule we must use.
16. Language access and accessibility: which languages and formats must we offer (e.g. ACA §1557, ADA)?
17. Does any message wording amount to a health claim needing substantiation (FTC, FDA, state advertising
    rules)?

## 4. Known technical gaps (privacy officer)

These are stated plainly so they are fixed, not assumed away. Each must be resolved, or accepted in writing
with compensating controls, before the `privacy_security` sign-off.

| Gap | Status | Required action |
|---|---|---|
| **Encryption at rest.** The record store and vault are plain SQLite files. | Open | Encrypted volume (e.g. LUKS or BitLocker) or an encrypted database; keys in a secrets manager. |
| **Authentication.** Roles are passed as strings; the library does not authenticate users. | Open | Put the library behind an application with SSO/MFA that maps people to roles. |
| **Backups and disaster recovery.** None built in. | Open | Encrypted, tested, off-site backups of the store and vault (kept separately). |
| **Breach response.** No runbook. | Open | Written incident and breach-notification procedure. |
| Vault key handling | **Fixed**: persistent vaults refuse to start without a stable key from the secrets manager, and detect a wrong key. | Store the key in the secrets manager; never commit it. |
| PHI to outside AI | **Fixed**: patient-scoped data is withheld unless a BAA is attested; even then only the minimum necessary. | Keep `phi_to_llm_permitted=False` unless question 13 is answered yes. |
| Pending clinician decisions lost on restart | **Fixed**: the approval queue is persisted in the record store. | None. |
| Access to re-identification | Audited; limited to the privacy officer and treating clinician roles. | Review the audit log monthly. |

## 5. Sign-off

Run after completing the review (records your name, date and a fingerprint of this exact document):

```bash
intelligent-plant signoff regulatory_legal  --name "<counsel name>" --role regulatory_counsel \
    --statement "Answers to every question in REGULATORY_REVIEW.md are on file at <location>."
intelligent-plant signoff privacy_security  --name "<name>" --role privacy_officer \
    --statement "Section 4 gaps resolved or accepted with controls documented at <location>."
```

Any edit to this document makes earlier sign-offs stale, so the reviewers must look again.
