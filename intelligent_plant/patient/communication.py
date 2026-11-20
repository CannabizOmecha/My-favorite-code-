"""Patient-facing communication: the part of the system a patient actually experiences.

Principles ("the person at the end of the supply chain is the reason the chain exists", Epilogue):
  * Plain language (target reading grade <= 8), kind, and actionable: what happened, what to do now,
    when to get help right away.
  * Never show internal identifiers, rule codes or jargon to the patient. Clinicians get the
    technical version separately.
  * Respect attention: urgent safety messages always go through; routine messages are de-duplicated,
    capped per day and held during quiet hours. Alert fatigue makes the important message invisible.
  * Honest about uncertainty, in words a patient can use ("a few records is too few to be sure").
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

URGENT_HELP = ("Get help right away (call 911) for chest pain, fainting, trouble breathing, or severe "
               "confusion. If a child may have eaten a cannabis product, call 911 or Poison Control at "
               "1-800-222-1222 now.")

# event key -> (urgency, patient-facing text). {fields} are filled from the payload.
TEMPLATES: dict[str, tuple[str, str]] = {
    "oral_redose": ("urgent",
                    "We noticed a second dose by mouth soon after the first. Edibles, oils and capsules can take "
                    "2 hours or more to work, so taking more too soon can make you feel very unwell later. "
                    "Please do not take any more today. Rest somewhere safe and do not drive. Your care team has "
                    "been told."),
    "adverse_effect": ("urgent",
                       "You logged {effects}. Thank you for writing it down. Please do not take your next dose until "
                       "your care team contacts you. They have been told and will help you decide what to do next."),
    "multiple_changes": ("routine",
                         "It looks like more than one thing changed at once (for example the product, the amount or "
                         "how you take it). Changing one thing at a time helps you and your care team see what is "
                         "really helping."),
    "missing_log_fields": ("routine",
                           "A few details were missing from your recent log ({fields}). Writing them down before each "
                           "dose takes a minute and helps your care team tailor your plan to you."),
    "medication_check": ("routine",
                         "One of your other medicines can interact with cannabis products. Your pharmacist and care "
                         "team have been told and may adjust your plan. Please keep taking your medicines as prescribed "
                         "unless they tell you otherwise."),
    "progress_summary": ("routine", "{summary}"),
}

PLAIN_FIELD_NAMES = {"timestamp": "time", "product_id": "product", "batch_lot": "batch number",
                     "total_thc_unit": "THC per unit", "total_cbd_unit": "CBD per unit", "dose_units": "amount taken",
                     "dose_mg_thc": "THC amount", "dose_mg_cbd": "CBD amount", "route": "how you took it",
                     "score_pre": "symptom score before", "score_post": "symptom score after",
                     "interval_min": "time between scores"}


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def reading_grade(text: str) -> float:
    """Flesch-Kincaid grade level (heuristic syllable count)."""
    sentences = max(1, len(re.findall(r"[.!?]+(?=\s|$)", text)))
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return 0.39 * len(words) / sentences + 11.8 * syl / len(words) - 15.59


@dataclass
class PatientMessage:
    patient_id: str
    urgency: str
    text: str
    created_at: datetime
    delivered: bool = False
    held_reason: str | None = None


@dataclass
class PatientMessenger:
    max_routine_per_day: int = 2
    quiet_start: time = time(21, 0)
    quiet_end: time = time(8, 0)
    dedupe_window: timedelta = timedelta(hours=24)
    sent: list[PatientMessage] = field(default_factory=list)
    held: list[PatientMessage] = field(default_factory=list)
    _last_by_key: dict = field(default_factory=dict)

    def _quiet(self, t: datetime) -> bool:
        tt = t.time()
        return tt >= self.quiet_start or tt < self.quiet_end

    def compose(self, event: str, payload: dict) -> tuple[str, str]:
        urgency, tpl = TEMPLATES[event]
        fields = {
            "effects": ", ".join(e.get("name", "a side effect") for e in payload.get("adverse_effects", []))
                       or "a side effect",
            "fields": ", ".join(PLAIN_FIELD_NAMES.get(f, f) for f in payload.get("missing", [])),
            "summary": payload.get("summary", ""),
        }
        text = tpl.format(**fields)
        if urgency == "urgent":
            text += " " + URGENT_HELP
        return urgency, text

    def send(self, patient_id: str, event: str, payload: dict, now: datetime) -> PatientMessage:
        urgency, text = self.compose(event, payload)
        msg = PatientMessage(patient_id, urgency, text, now)
        key = (patient_id, event)
        last = self._last_by_key.get(key)
        if urgency != "urgent":
            today = [m for m in self.sent if m.patient_id == patient_id and m.urgency == "routine"
                     and m.created_at.date() == now.date()]
            if last and now - last < self.dedupe_window:
                msg.held_reason = "duplicate within 24 h"
            elif len(today) >= self.max_routine_per_day:
                msg.held_reason = "daily limit for routine messages reached"
            elif self._quiet(now):
                msg.held_reason = "quiet hours; will send in the morning"
        if msg.held_reason:
            self.held.append(msg)
        else:
            msg.delivered = True
            self.sent.append(msg)
            self._last_by_key[key] = now
        return msg

    def stats(self) -> dict:
        per_patient = defaultdict(int)
        for m in self.sent:
            per_patient[m.patient_id] += 1
        return {"delivered": len(self.sent), "held": len(self.held),
                "urgent_delivered": sum(m.urgency == "urgent" for m in self.sent),
                "max_messages_one_patient": max(per_patient.values(), default=0)}


def progress_summary(logs: list[dict], goal: str | None = None, min_records: int = 7) -> str:
    """A patient's own results, shown back in plain words, with honest uncertainty."""
    pairs = [(l["score_pre"], l["score_post"]) for l in logs
             if l.get("score_pre") is not None and l.get("score_post") is not None]
    if not pairs:
        return "Once you have logged a few doses, we will show you how you are doing here."
    pre = statistics.mean(p for p, _ in pairs)
    post = statistics.mean(q for _, q in pairs)
    change = pre - post
    side = sum(1 for l in logs if l.get("adverse_effects"))
    what = f" for {goal}" if goal else ""
    s = (f"Here is how you are doing{what}. Across {len(pairs)} logged doses, your symptom score went from "
         f"about {pre:.1f} before to {post:.1f} after, out of 10.")
    if len(pairs) < min_records:
        s += " That is still too few records to be sure, so keep logging and we will know more soon."
    elif change >= 2:
        s += " That is a clear change."
    elif change >= 1:
        s += " That is a small change. Your care team will talk with you about whether it is enough."
    else:
        s += " So far there is little change. That is useful to know, and your care team will review it with you."
    if side:
        s += f" You also logged side effects {side} time{'s' if side > 1 else ''}; your care team can see these."
    return s
