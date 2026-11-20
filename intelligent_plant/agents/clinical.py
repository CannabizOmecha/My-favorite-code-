"""Clinical swarm (Chapters 4-7, 13; Appendix C.1-C.3).

What runs autonomously: log capture and validation, reminders, hard-rule
interaction alerts, safety alerts, outcome modelling, review scheduling.
What a clinician decides: whether to trial at all, the product, every dose,
every titration step, continuation. The software never originates a dose
number: TitrationPlan steps are entered by the clinician, and the system
enforces the structure around them (Table 13.1).

ctx.observations["patients"][pid] = {
  "profile": PatientRiskProfile, "indication": str, "standard_care_tried": bool,
  "onset_need": "sustained"|"rapid", "status": "new_consult"|"titrating"|...,
  "plan": TitrationPlan|None, "logs": [Table 13.2 dicts], "meds_changed": bool,
  "preferences": {"avoid_psychoactivity": bool, "route_preference": str, "goals": [str]},
  "last_summary_at": datetime|None}

The patient's own priorities (Chapter 13.1: "their preferences about psychoactivity") shape every
recommendation, inside safety limits: a preference can narrow the options, never widen them past a
safety rule.
ctx.observations["product_catalog"] = [{product_id, form, route, unit_mg_thc,
  unit_mg_cbd, coa: COA, batch_cv, available_months}]
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..knowledge.evidence import HARMS, SPECIALIST_ONLY, Grade, evidence_for
from ..knowledge.interactions import check_interactions, screen_contraindications
from ..science.pharmacology import ROUTES, oral_redose_safe, relative_thc_exposure
from ..science.quality import audit_coa
from ..science.stats import n_of_1_schedule, partial_pool, population_hyperparameters
from .base import Agent, Context, Proposal
from ..patient.communication import progress_summary
from .moe import Expert, ExpertOpinion, MixtureOfExperts

SCOPE = ("Adults (>=18) with an indication in the curated evidence map, under a treating clinician. "
         "OUT OF SCOPE: pediatric patients, pregnancy/lactation, specialist-managed epilepsies (Dravet, LGS, TSC: "
         "FDA-approved product under a specialist), acute emergencies, and any dose selection.")

LOG_FIELDS = ("timestamp", "product_id", "batch_lot", "total_thc_unit", "total_cbd_unit", "dose_units",
              "dose_mg_thc", "dose_mg_cbd", "route", "score_pre", "score_post", "interval_min")
SEVERE = {"moderate", "severe"}


@dataclass
class TitrationPlan:
    """Entered by the treating clinician. `steps` are the clinician's own
    instructions (from a published protocol, ref 140); the system only tracks
    which step is current and when the next review is due."""
    patient_id: str
    clinician: str
    steps: list[str]
    hold_hours: float
    review_after_days: int
    benefit_threshold: float                 # pre-specified improvement (0-10 scale) to continue
    start: datetime
    current_step: int = 0
    step_started: datetime | None = None
    phase: str = "initiation"                # baseline | initiation | titration | maintenance | review | stopped
    baseline_score: float | None = None
    n_of_1_enrolled: bool = False
    history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- experts
class EvidenceExpert(Expert):
    name, keywords = "evidence", ("indication", "pain", "nausea", "spasticity", "ptsd", "seizure")

    async def opine(self, q: dict, ctx: Context) -> ExpertOpinion | None:
        ind = q["indication"]
        ev = evidence_for(ind)
        if ind in SPECIALIST_ONLY:
            return ExpertOpinion(self.name, "out_of_scope_refer_specialist", 1.0,
                                 [f"OUT OF SCOPE: {ev.claim}. {ev.note}"], {"grade": ev.grade.name}, [ev.cite()],
                                 question="trial", blocking=True)
        if ev.grade < Grade.MODERATE:
            rec = "evidence_insufficient_discuss_n_of_1"
        elif not q.get("standard_care_tried"):
            rec = "standard_care_first"
        else:
            rec = "trial_reasonable"
        return ExpertOpinion(self.name, rec, 0.9, [], {"grade": ev.grade.name, "note": ev.note}, [ev.cite()],
                             question="trial")


class ContraindicationExpert(Expert):
    name, keywords = "contraindications", ("age", "psychosis", "pregnan", "cardio")

    async def opine(self, q: dict, ctx: Context) -> ExpertOpinion | None:
        flags = screen_contraindications(q["profile"])
        stop = [f for f in flags if f.level == "stop"]
        return ExpertOpinion(self.name, "do_not_initiate" if stop else None, 1.0,
                             [f"[{f.level.upper()}] {f.message}" for f in flags if f.level in ("stop", "caution")],
                             {"screen": [f.__dict__ for f in flags]}, ["Appendix C.1"],
                             question="trial", blocking=bool(stop))


class InteractionExpert(Expert):
    name, keywords = "interactions", ("medications", "clobazam", "warfarin")

    async def opine(self, q: dict, ctx: Context) -> ExpertOpinion | None:
        alerts = check_interactions(q["profile"].medications, {"CBD", "THC"})
        return ExpertOpinion(self.name, None, 1.0,
                             [f"[{a.severity.upper()} DDI] {a.message} Action: {a.action}" for a in alerts
                              if a.severity == "major"],
                             {"alerts": [a.__dict__ for a in alerts]}, ["Table 6.1 (hard rules)"])


class PharmacogenomicExpert(Expert):
    name, keywords = "pharmacogenomics", ("cyp2c9", "genotype")

    async def opine(self, q: dict, ctx: Context) -> ExpertOpinion | None:
        g = q["profile"].cyp2c9_genotype
        rel, note = relative_thc_exposure(g)
        flags = [f"CYP2C9 {g}: ~{rel:.1f}x relative THC exposure at equal dose. {note}"] if rel > 1.5 else []
        return ExpertOpinion(self.name, None, 0.7, flags, {"relative_thc_exposure": rel, "note": note},
                             ["Table 4.1 (ref 26)", "Genetic data: harm reduction, never gatekeeping (Ch 4.5)"])


class RouteExpert(Expert):
    name, keywords = "route", ("onset", "sustained", "rapid", "route")

    async def opine(self, q: dict, ctx: Context) -> ExpertOpinion | None:
        need = q.get("onset_need", "sustained")
        if need == "rapid":
            rec, why = "inhaled", "Onset within minutes; respiratory harms and dose control must be discussed."
        else:
            rec, why = "oral", ("Sustained effect; non-inhaled is the guideline-recommended form for chronic pain "
                                "(ref 97). Onset 60-120 min: never re-dose before the full effect.")
        flags = []
        pref = (q.get("preferences") or {}).get("route_preference")
        if pref and pref != rec:
            if need == "sustained" and pref in ("oral", "oromucosal"):
                rec, why = pref, why + f" Patient prefers {pref}; clinically equivalent for sustained effect."
            else:
                flags.append(f"Patient prefers {pref}; recommended {rec} for clinical reasons. Discuss with the "
                             f"patient rather than override their preference silently.")
        if q.get("food_relation") == "variable":
            flags.append("CBD absorption varies ~3.8x with high-fat food (ref 94): take consistently relative to meals.")
        return ExpertOpinion(self.name, rec, 0.8, flags, {"route": rec, "why": why, "pk": ROUTES[rec]},
                             ["Ch 6.1", "Ch 13.1 step 4"], question="route")


class HarmsExpert(Expert):
    name, keywords = "harms", ("age", "psychosis", "potency", "substance")

    async def opine(self, q: dict, ctx: Context) -> ExpertOpinion | None:
        p = q["profile"]
        flags = []
        if p.psychosis_history_family_strong or p.age < 25:
            flags.append("Avoid high-potency THC products; " + HARMS["psychosis_daily_high_potency"].cite())
        if p.substance_use_disorder_history:
            flags.append(HARMS["cud"].cite())
        flags.append("Child-resistant packaging and locked storage are not optional (ref 110).")
        return ExpertOpinion(self.name, None, 0.8, flags, {}, ["Ch 6.3"])


class ProductMatchExpert(Expert):
    """Choose a specific, repeatable product with an audited COA (Ch 13.1 step 5,
    Appendix C.2). CBD-first (ref 140); consistency is a safety trait (Ch 9.3)."""
    name, keywords = "product_match", ("product", "chemotype", "indication")

    async def opine(self, q: dict, ctx: Context) -> ExpertOpinion | None:
        route = q.get("route_hint", "oral")
        catalog = ctx.observations.get("product_catalog", [])
        ranked, rejected = [], []
        for prod in catalog:
            audit = audit_coa(prod["coa"], ctx.observations.get("market_total_thc"))
            reasons = []
            if not audit.passed:
                reasons.append("COA failed audit")
            if prod["form"] in {"edible", "capsule", "tincture"} and not prod["coa"].finished_product:
                reasons.append("no finished-product COA (label is an estimate, Ch 11.2)")
            if prod.get("available_months", 0) < 3:
                reasons.append("not reliably available for the trial")
            if prod["route"] != route:
                reasons.append(f"route {prod['route']} != {route}")
            if prod.get("batch_cv", 1.0) > 0.10:
                reasons.append(f"batch CV {prod['batch_cv']:.0%} > 10%")
            if (q.get("preferences") or {}).get("avoid_psychoactivity") and \
                    prod["unit_mg_cbd"] < 10 * max(prod["unit_mg_thc"], 0.01):
                reasons.append("patient wants to avoid feeling high; product is not CBD-dominant")
            if reasons:
                rejected.append({"product_id": prod["product_id"], "reasons": reasons})
                continue
            cbd_dominant = prod["unit_mg_cbd"] >= 10 * max(prod["unit_mg_thc"], 0.01)
            ranked.append({"product_id": prod["product_id"], "cbd_dominant": cbd_dominant,
                           "unit_mg_thc": prod["unit_mg_thc"], "unit_mg_cbd": prod["unit_mg_cbd"],
                           "coa_id": prod["coa"].coa_id, "coa_trust": audit.trust_score, "batch_cv": prod["batch_cv"]})
        # Lexicographic: safety (CBD-dominant, lower THC per unit) > accuracy (COA trust, consistency)
        ranked.sort(key=lambda r: (not r["cbd_dominant"], r["unit_mg_thc"], -r["coa_trust"], r["batch_cv"]))
        rec = ranked[0]["product_id"] if ranked else None
        return ExpertOpinion(self.name, rec, 0.7, [] if ranked else ["No catalog product meets the C.2 criteria."],
                             {"ranked": ranked[:5], "rejected": rejected}, ["Appendix C.2", "ref 140 CBD-first"],
                             question="product")


def clinical_moe() -> MixtureOfExperts:
    experts = [EvidenceExpert(), ContraindicationExpert(), InteractionExpert(), PharmacogenomicExpert(),
               RouteExpert(), HarmsExpert(), ProductMatchExpert()]
    # Safety-relevant experts are consulted on EVERY case regardless of the gate.
    return MixtureOfExperts(experts, top_k=3, always_consult=("evidence", "contraindications", "interactions",
                                                              "harms", "route"))


# ---------------------------------------------------------------- agents
class ConsultationAgent(Agent):
    """Runs the seven-step consultation sequence (Chapter 13.1) as decision support."""
    name, domain = "consultation", "clinical"

    def __init__(self, moe: MixtureOfExperts | None = None):
        self.moe = moe or clinical_moe()

    async def propose(self, ctx: Context) -> list[Proposal]:
        out = []
        for pid, pt in ctx.observations.get("patients", {}).items():
            if pt.get("status") != "new_consult":
                continue
            q = {"patient_id": pid, "indication": pt["indication"], "profile": pt["profile"],
                 "standard_care_tried": pt.get("standard_care_tried", False),
                 "onset_need": pt.get("onset_need", "sustained"), "food_relation": pt.get("food_relation"),
                 "medications": pt["profile"].medications, "preferences": pt.get("preferences", {})}
            first = await self.moe.consult(q, ctx)
            route = self.moe.answer(first, "route") or "oral"
            gate = self.moe.answer(first, "trial")
            prod = await self.moe.experts["product_match"].opine({**q, "route_hint": route}, ctx)
            stops = [f for f in first.flags if f.startswith("[STOP]") or f.startswith("OUT OF SCOPE")]
            ev = next(o for o in first.opinions if o.expert == "evidence")
            for o in first.opinions:
                if o.expert == "interactions":
                    for a in o.findings["alerts"]:
                        out.append(Proposal(self.name, "interaction_alert",
                                            f"Patient {pid}: {a['message']} Action: {a['action']}",
                                            {"patient_id": pid, **a}, {"safety": 1.0},
                                            [f"refs {a['refs']}"], target=f"patient:{pid}:ddi:{a['rule_id']}"))
            sequence = {
                "0_patient_priorities": {"goals": pt.get("preferences", {}).get("goals", []),
                                         "preferences": pt.get("preferences", {}),
                                         "note": "Start the conversation here: what matters most to this patient."},
                "1_indication_and_evidence": ev.findings | {"recommendation": gate},
                "2_contraindications": [f for f in first.flags if "]" in f and "DDI" not in f],
                "3_interactions": [f for f in first.flags if "DDI" in f],
                "4_route": route,
                "5_product": prod.findings if prod else {},
                "6_titration": "Clinician sets starting dose, increments and ceiling from a published protocol "
                               "(ref 140); start low, go slow, CBD first.",
                "7_outcomes_review_stopping_rule": "Define validated instrument, review at 4-6 weeks and the "
                                                   "benefit threshold BEFORE starting.",
            }
            out.append(Proposal(self.name, "consultation_report",
                                f"Patient {pid} ({pt['indication']}): {gate or 'n/a'}; "
                                f"{len(stops)} stop/out-of-scope flag(s); route {route}.",
                                {"patient_id": pid, "sequence": sequence, "flags": first.flags,
                                 "experts_consulted": first.consulted, "abstained": first.abstained,
                                 "gate_weights": first.gate_weights, "scope": SCOPE},
                                {"safety": 0.8, "accuracy": 0.7}, [e for o in first.opinions for e in o.evidence],
                                uncertainty="Decision support only. Evidence grade: " + ev.findings["grade"] +
                                            ". Most cannabinoid combinations have never been studied.",
                                target=f"patient:{pid}:consult"))
            if not stops and gate == "trial_reasonable" and prod and prod.recommendation:
                out.append(Proposal(self.name, "product_recommendation",
                                    f"Patient {pid}: candidate product {prod.recommendation} (route {route}); "
                                    f"clinician selects and sets dose.",
                                    {"patient_id": pid, "product_id": prod.recommendation,
                                     "alternatives": prod.findings["ranked"][1:3]},
                                    {"safety": 0.6, "accuracy": 0.6}, prod.evidence + ev.evidence,
                                    uncertainty="Terpene profile is a consistency fingerprint, not a predictor of "
                                                "effect (Ch 7.3).", target=f"patient:{pid}:product"))
        return out


class TitrationMonitorAgent(Agent):
    """Watches patient logs between visits and enforces the titration rules."""
    name, domain = "titration_monitor", "clinical"

    def __init__(self, oral_wait_min: float = 120.0, min_records: int = 5):
        self.oral_wait_min, self.min_records = oral_wait_min, min_records

    @staticmethod
    def _improvements(logs: list[dict]) -> list[float]:
        return [l["score_pre"] - l["score_post"] for l in logs
                if l.get("score_pre") is not None and l.get("score_post") is not None]

    async def propose(self, ctx: Context) -> list[Proposal]:
        patients = ctx.observations.get("patients", {})
        all_impr = [self._improvements(p.get("logs", [])) for p in patients.values()]
        mu, tau, sigma = population_hyperparameters(all_impr) if any(all_impr) else (0.0, 2.0, 2.0)
        out = []
        for pid, pt in patients.items():
            plan: TitrationPlan | None = pt.get("plan")
            logs = sorted(pt.get("logs", []), key=lambda l: l["timestamp"])
            if not plan or plan.phase == "stopped":
                continue
            out += self._log_quality(pid, logs)
            out += self._rule_checks(pid, logs)
            if pt.get("meds_changed"):
                for a in check_interactions(pt["profile"].medications, {"CBD", "THC"}):
                    out.append(Proposal(self.name, "interaction_alert",
                                        f"Patient {pid} medication change: {a.message} Action: {a.action}",
                                        {"patient_id": pid, **a.__dict__, "patient_event": "medication_check",
                                         "notify": ["treating_clinician", "pharmacist", "patient"]},
                                        {"safety": 1.0}, [f"refs {a.refs}"],
                                        target=f"patient:{pid}:ddi:{a.rule_id}"))
            last = pt.get("last_summary_at")
            if logs and (last is None or ctx.now - last >= timedelta(days=7)):
                goals = pt.get("preferences", {}).get("goals") or [None]
                out.append(Proposal(self.name, "patient_progress_summary",
                                    f"Patient {pid}: weekly progress summary sent to the patient.",
                                    {"patient_id": pid, "notify": ["patient"], "patient_event": "progress_summary",
                                     "summary": progress_summary(logs, goals[0])},
                                    {"safety": 0.2, "ethics": 0.6, "accuracy": 0.5}, ["Ch 13.3 patient log"],
                                    target=f"patient:{pid}:summary"))
            impr = self._improvements(logs)
            est = partial_pool(impr, mu, tau, sigma, min_n=self.min_records)
            out += self._step_logic(pid, plan, logs, est, ctx.now)
            if len(logs) >= 30 and not plan.n_of_1_enrolled and plan.phase == "maintenance":
                out.append(Proposal(self.name, "n_of_1_enrolment",
                                    f"Patient {pid}: {len(logs)} records on a stable regimen. Offer a randomised, "
                                    f"blinded N-of-1 trial to test the benefit within this patient.",
                                    {"patient_id": pid, "schedule": n_of_1_schedule(4)},
                                    {"accuracy": 1.0, "safety": 0.2}, ["Ch 5.4 (ref 98)", "Ch 13.3"],
                                    uncertainty="Requires a stable condition and a rapidly reversible treatment."))
        return out

    def _log_quality(self, pid: str, logs: list[dict]) -> list[Proposal]:
        missing = {f for l in logs[-5:] for f in LOG_FIELDS if l.get(f) in (None, "")}
        if not missing:
            return []
        return [Proposal(self.name, "patient_reminder",
                         f"Patient {pid}: recent log entries are missing {sorted(missing)}. Reminder sent: write the "
                         f"numbers down before dosing.", {"patient_id": pid, "missing": sorted(missing),
                                                            "patient_event": "missing_log_fields"},
                         {"safety": 0.3, "accuracy": 0.8}, ["Table 13.2"], target=f"patient:{pid}:reminder")]

    def _rule_checks(self, pid: str, logs: list[dict]) -> list[Proposal]:
        out = []
        for prev, cur in zip(logs, logs[1:]):
            t0, t1 = datetime.fromisoformat(prev["timestamp"]), datetime.fromisoformat(cur["timestamp"])
            if prev.get("route") == "oral" and cur.get("route") == "oral" and \
                    not oral_redose_safe((t1 - t0).total_seconds() / 60, self.oral_wait_min):
                out.append(Proposal(self.name, "safety_alert",
                                    f"Patient {pid}: oral re-dose {(t1 - t0).total_seconds() / 60:.0f} min after the "
                                    f"previous oral dose. Oral effects can take hours: wait before any further dose.",
                                    {"patient_id": pid, "at": cur["timestamp"], "notify": ["patient", "clinician"],
                                     "patient_event": "oral_redose"},
                                    {"safety": 1.0}, ["Ch 6.1 edible overdose script"],
                                    target=f"patient:{pid}:redose:{cur['timestamp']}"))
            changed = [k for k in ("product_id", "dose_units", "route") if prev.get(k) != cur.get(k)]
            if len(changed) >= 2:
                out.append(Proposal(self.name, "safety_alert",
                                    f"Patient {pid}: {changed} changed at once; effects can no longer be attributed. "
                                    f"Change one variable at a time.", {"patient_id": pid, "changed": changed,
                                                                         "notify": ["patient", "clinician"],
                                                                         "patient_event": "multiple_changes"},
                                    {"safety": 0.6, "accuracy": 1.0}, ["Ch 13.2"],
                                    target=f"patient:{pid}:multivar:{cur['timestamp']}"))
        for l in logs[-10:]:
            sev = [a for a in l.get("adverse_effects", []) if a.get("severity") in SEVERE]
            if sev:
                out.append(Proposal(self.name, "safety_alert",
                                    f"Patient {pid}: adverse effect(s) {[a['name'] for a in sev]} at {l['timestamp']}. "
                                    f"Clinician notified: stop or slow per protocol.",
                                    {"patient_id": pid, "adverse_effects": sev, "notify": ["patient", "clinician"],
                                     "patient_event": "adverse_effect"},
                                    {"safety": 1.0}, ["Table 13.1 initiation rule"],
                                    target=f"patient:{pid}:ae:{l['timestamp']}"))
        return out

    def _step_logic(self, pid: str, plan: TitrationPlan, logs: list[dict], est, now: datetime) -> list[Proposal]:
        out = []
        recent_ae = any(a.get("severity") in SEVERE for l in logs[-5:] for a in l.get("adverse_effects", []))
        unc = (f"Pooled improvement {est.mean:.1f} (90% {est.lo90:.1f} to {est.hi90:.1f}) points; {est.note}")
        if now - plan.start >= timedelta(days=plan.review_after_days) and plan.phase != "review":
            meets = est.lo90 >= plan.benefit_threshold
            out.append(Proposal(self.name, "continue_or_stop_review",
                                f"Patient {pid}: scheduled review. Pre-specified threshold {plan.benefit_threshold}: "
                                f"{'met with confidence' if meets else 'not demonstrated'}.",
                                {"patient_id": pid, "estimate": est.__dict__, "threshold": plan.benefit_threshold},
                                {"safety": 0.5, "accuracy": 1.0}, ["Table 13.1 review"], uncertainty=unc,
                                target=f"patient:{pid}:step"))
            return out
        if recent_ae:
            out.append(Proposal(self.name, "titration_step",
                                f"Patient {pid}: hold or step back (adverse effects); clinician to decide.",
                                {"patient_id": pid, "action": "hold_or_step_back", "current_step": plan.current_step},
                                {"safety": 1.0}, ["Table 13.1"], uncertainty=unc, target=f"patient:{pid}:step"))
            return out
        started = plan.step_started or plan.start
        held = (now - started).total_seconds() / 3600 >= plan.hold_hours
        if not est.abstain and est.mean >= plan.benefit_threshold:
            if plan.phase != "maintenance":
                out.append(Proposal(self.name, "titration_step",
                                    f"Patient {pid}: benefit threshold reached at step {plan.current_step + 1}; "
                                    f"recommend maintenance (do not escalate to chase tolerance).",
                                    {"patient_id": pid, "action": "enter_maintenance",
                                     "current_step": plan.current_step},
                                    {"safety": 0.8, "accuracy": 0.7}, ["Table 13.1 maintenance"], uncertainty=unc,
                                    target=f"patient:{pid}:step"))
        elif held and plan.current_step + 1 < len(plan.steps) and plan.phase in ("initiation", "titration"):
            out.append(Proposal(self.name, "titration_step",
                                f"Patient {pid}: hold interval complete, no significant adverse effects, threshold not "
                                f"reached. Clinician may advance to their step {plan.current_step + 2}: "
                                f"'{plan.steps[plan.current_step + 1]}'.",
                                {"patient_id": pid, "action": "advance", "to_step": plan.current_step + 1},
                                {"safety": 0.4, "accuracy": 0.6}, ["Table 13.1 titration", "ref 140"],
                                uncertainty=unc, target=f"patient:{pid}:step"))
        if len(logs) >= 20:
            first, last = self._improvements(logs[:10]), self._improvements(logs[-10:])
            if first and last and statistics.mean(first) - statistics.mean(last) > 1.5:
                out.append(Proposal(self.name, "continue_or_stop_review",
                                    f"Patient {pid}: response has drifted down by "
                                    f"{statistics.mean(first) - statistics.mean(last):.1f} points (possible tolerance, "
                                    f"falling b in Ch 2.5). Review rather than escalate.",
                                    {"patient_id": pid}, {"safety": 0.6, "accuracy": 0.7}, ["Ch 2.5", "Ch 13.3"],
                                    uncertainty=unc, target=f"patient:{pid}:drift"))
        return out


def clinical_agents() -> list[Agent]:
    return [ConsultationAgent(), TitrationMonitorAgent()]
