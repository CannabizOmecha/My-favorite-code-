# The Intelligent Plant — autonomous seed-to-patient system

[![tests](https://github.com/CannabizOmecha/My-favorite-code-/actions/workflows/tests.yml/badge.svg)](https://github.com/CannabizOmecha/My-favorite-code-/actions/workflows/tests.yml)

> **North star: improve the experience and outcomes of patients.** Everything upstream exists for the person at the
> end of the supply chain. See [`CLAUDE.md`](CLAUDE.md) for the rules that protect patients.

Software implementation of C. A. Cassinelli, *The Intelligent Plant: AI-Powered Cannabis Science, Therapeutics and
Operations* (Second Revised Edition, 2026). Every formula in Appendix A, the Appendix B data schema, the hard rules
of Table 6.1, the autonomy ordering of Table 8.2, the consultation sequence of Chapter 13 and the governance checklist
of Chapter 16 are code here, and the worked examples in the book are regression tests.

```bash
pip install -e ".[dev]"          # numpy + pytest; add [llm] for Claude reasoning
python -m intelligent_plant demo # one autonomous cycle over a SYNTHETIC facility, clinic and supply chain
python -m intelligent_plant quantum  # superposition, entanglement, VQE, Grover, QAOA benchmarks
pytest                           # 92 tests: Appendix A worked examples + safety invariants
```

## Architecture

```
              ┌──────────────────── Swarm cycle (parallel, every interval) ─────────────────────┐
 sensors ─┐   │ 1. ~25 domain agents propose in parallel (failures isolated → alert, no action) │
 LIMS/COA ┤   │ 2. critics attack every proposal in parallel                                     │
 track &  ┼─▶ │      deterministic: safety officer · evidence auditor · data quality  (may VETO) │
 trace    │   │      Claude jury: skeptic · patient advocate · regulator · economist · equity    │
 patient  ┤   │                   (may only WARN → forces human review)                          │
 app      │   │ 3. lexicographic governance: Safety > Ethics > Accuracy > Efficiency > Profit    │
 reg feeds┘   │ 4. autonomy gate (Table 8.2): execute │ execute+alarm │ execute+notify │ queue   │
              │                               for a named human │ refuse (human-only)        │
              │ 5. every disposition → append-only, hash-chained record store                    │
              └──────────────────────────────────────────────────────────────────────────────────┘
                     ▲ clinical consults are routed through a mixture-of-experts (MoE) gate ▲
```

| Layer | Module | Book |
|---|---|---|
| Formulas | `science/chemistry.py` total THC, first-order decarb, Arrhenius fit, decarb planner | Ch 3, A.1–A.3, A.14 |
| | `science/environment.py` Tetens VPD (air & leaf), DLI, PID, MPC with energy pricing | Ch 8, 10, A.8–A.11 |
| | `science/pharmacology.py` AUC, one-compartment PK, CYP2C9 exposure, feedback stability | Ch 2, 4, 6, A.4–A.6 |
| | `science/quality.py` label-claim uncertainty, ten-point COA audit, Shewhart SPC | Ch 3.4, 11, A.12 |
| | `science/breeding.py` GBLUP vs top-k MAS chosen by cross-validation, patient-first index, Ne guard | Ch 9 |
| | `science/stats.py` partial pooling, calibration, N-of-1, factorial synergy test, PSI drift | Ch 4.4, 5.4, 7, 13.3 |
| | `science/economics.py` §280E, Table 10.1 sustainability metrics | Ch 1.4, 10.4, A.13 |
| Knowledge | `knowledge/evidence.py` Table P.1 grades, NASEM evidence map, entourage & harms grades | Ch 5–7 |
| | `knowledge/interactions.py` Table 6.1 as hard rules, Appendix C.1 screen | Ch 6, 13 |
| | `knowledge/regulation.py` + `config/jurisdictions.json` — rules as **configuration**, hemp Dec-11-2026 switch, regulatory diffing | Ch 12.3, 14 |
| Data | `data/store.py` Appendix B schema, validation at entry, never-overwrite corrections, hash chain, outcome→genotype trace | App B, Ch 12.2, 16 |
| | `data/privacy.py` pseudonymisation at collection, separate audited key vault, role redaction | App B, Ch 4.5 |
| Governance | `governance/ethics.py` lexicographic ordering with pre-declared tolerances | Ch 1.5 |
| | `governance/autonomy.py` per-task autonomy level + named accountable owner, approval queue, override log | Table 8.2, Ch 16.3 |
| | `governance/registry.py` model cards, deployment gate, drift, rollback, maturity model | App C.5, Table 16.1 |
| Agents | `agents/cultivation.py` calibration, climate MPC + fail-safe, irrigation, nutrients, vision triage, harvest, lighting | Part III |
| | `agents/breeding.py`, `processing.py`, `compliance.py`, `clinical.py`, `enterprise.py` | Parts III–V |
| | `agents/moe.py` gated, reliability-weighted experts; safety flags are a union, blocking verdicts can't be outvoted | |
| | `agents/swarm.py`, `agents/llm.py` (Claude, optional) | |
| Loop | `orchestrator.py` `IntelligentEnterprise.run_forever(feed)` + actuators + human `decide()` | Ch 16.1 |

## What the patient experiences (`patient/communication.py`)

- **Plain-language messages** (reading grade ≤ 8, no internal IDs or jargon) that say what happened, what to do now,
  and when to get help right away (911 / Poison Control 1-800-222-1222). Clinicians get the technical version.
- **Respect for attention.** Urgent safety messages always go through. Routine ones are de-duplicated, limited to
  2 per day and held during quiet hours.
- **Preferences shape care.** Wanting to avoid feeling high filters products to CBD-dominant ones, and a route
  preference is honoured when it is clinically equivalent. Otherwise the disagreement is flagged for discussion.
  Patient goals open every consultation report.
- **Patients see their own progress** in a weekly summary, including when there is still too little data or little
  change.

## Self-iterating learning (`learning/continual.py`)

Every cycle, `ContinualLearningAgent` runs one self-iteration of each registered model:

- **Recursive Bayesian inference.** A conjugate Normal–Inverse-Gamma linear model is updated one observation at a
  time. Its Student-t predictive intervals are wide when data are few and tighten as data accumulate, and they are
  measured to be calibrated (90% intervals cover ~90% of new outcomes).
- **Champion/challenger.** Challengers are retrained over a ladder of recent windows and scored *prospectively*,
  on outcomes that arrived after their training cutoff. A challenger is promoted only if it is ≥2% better (a
  threshold fixed in advance) **and** passes the Appendix C.5 gate. Clinical models wait for their named owner.
- **Drift and fail-safe.** Input drift (PSI) and prospective degradation are monitored. A degraded champion is
  **suspended and abstains** until a validated replacement exists. In tests, a regime change is caught within one
  cycle, with no false alarms while data are stable, and the model recovers to the noise floor within 100–200
  outcomes.
- **Never learns from itself.** Only measured outcomes may train a model. Feeding back its own predictions raises
  an error, because that is how feedback loops amplify error.

## Quantum algorithms (`quantum/`)

An exact state-vector simulator implements superposition, interference, entanglement (with von Neumann entropy)
and Born-rule measurement. On it run **QAOA** (QUBO optimisation), **Grover** search and **VQE**. VQE reproduces
the textbook H₂ ground state, −1.137306 Ha, within chemical accuracy. `QuantumPortfolioAgent` uses QAOA to pick
which cultivars to advance, trading patient-first value against kinship so that near-clones aren't both selected
(Ch 9.4).

Following Chapter 15, **every quantum result is benchmarked against the classical answer**, and the decision gate
uses quantum only when it is at least as good. On a simulator it never is faster: the cost grows as 2ⁿ, so the gate
keeps the classical answer and logs both. A real QPU plugs in through the `QuantumBackend` protocol and is held to
the same test.

## What runs on its own, and what waits for a person

The system is autonomous wherever the manuscript says to automate, and it routes everything else to a **named
role**. These boundaries come from the book, not from the code's caution:

| Runs autonomously | Executes, owner notified (reversible) | Needs a named human |
|---|---|---|
| data logging, alerting, calibration work orders, COA audits, SPC, inventory reconciliation, regulatory monitoring, patient log capture & reminders, **hard-rule interaction alerts**, **patient safety alerts** | climate setpoints (with alarms), sensor fail-safe, drying environment, irrigation, lighting/energy, batch **holds**, hemp quarantine, model rollback | nutrient recipes, disease confirmation, harvest timing, breeding selection, decarb plans, batch **release**, anomaly investigation, regulatory interpretation (counsel), consultation reports, product choice, every titration step, continue/stop review, N-of-1 enrolment, model deployment |

Hard refusals (the swarm will not act even if an agent asks): **crop destruction** and **dose selection**. Agents
cannot originate dose numbers. Clinicians enter their own titration steps from a published protocol (ref 140), and
the system enforces the structure around them (Table 13.1): one variable at a time, hold intervals, no oral re-dose
inside the onset window, stop/slow on adverse effects, pre-specified review thresholds.

## Claude integration (optional)

With `pip install ".[llm]"` and Anthropic credentials, `ClaudeReasoner` (`claude-opus-5-5`, adaptive thinking,
server-side refusal fallback enabled) adds:
- a five-persona **jury** that deliberates on every human-gated proposal in parallel. Jurors can only *warn*, never
  approve or veto.
- regulatory-change summaries, labelled **"lead, verify against primary source"** and routed to counsel.

Claude is never in the path of hard interaction rules, dose decisions or autonomy gates. Without credentials,
everything falls back to deterministic logic.

## Connecting a real facility

`run_forever(feed)` pulls an observations dict each cycle. Feed adapters for the BMS/climate controller, fertigation,
LIMS (COAs), your state track-and-trace system, the patient app and regulatory feeds produce that dict. The
actuators in `orchestrator.py` are where writes go back out. Observation keys are documented at the top of each
`agents/*.py` module.

## Before production use

- **Validate on your own data** (Appendix C.4/C.5): the MPC room model, the volatile-loss parameters, the EC/pH
  ranges and the vision models are placeholders until identified from your facility.
- **Clinical decision support** may be regulated as a medical device, and patient data falls under HIPAA and state
  law. Get regulatory and legal review before clinical use.
- **Jurisdiction config** reflects the manuscript's mid-November-2026 record. Verify it against primary sources and
  add your state's testing panels and limits.
- `simulation.py` is entirely synthetic. None of its numbers is an empirical result.

## A finding from implementing Chapter 3.3

The book states that a lower-temperature, longer decarboxylation preserves terpenes better at equal conversion.
Under first-order kinetics that holds only if terpene loss has a **higher** activation energy than decarboxylation
(87 kJ/mol). For evaporative losses (typically lower Eₐ), a hotter, shorter run retains more. The planner reports
what the parameters imply, and `test_decarb_planner_reports_physics_not_assumption` demonstrates both regimes.
Measuring Eₐ for terpene loss in your matrix settles it.
