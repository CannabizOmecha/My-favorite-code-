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
from datetime import datetime, time, timedelta, timezone, tzinfo

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
    tz: tzinfo = timezone.utc                 # default patient-local time zone for quiet hours and daily caps
    patient_tz: dict[str, tzinfo] = field(default_factory=dict)
    sent: list[PatientMessage] = field(default_factory=list)
    held: list[PatientMessage] = field(default_factory=list)
    suppressed: list[PatientMessage] = field(default_factory=list)   # duplicates, kept for audit
    _last_by_key: dict = field(default_factory=dict)
    _delivered_events: set = field(default_factory=set)
    _held_meta: dict = field(default_factory=dict)   # id(msg) -> (event, payload)

    @staticmethod
    def _aware(t: datetime) -> datetime:
        return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)

    def _local(self, patient_id: str, t: datetime) -> datetime:
        return self._aware(t).astimezone(self.patient_tz.get(patient_id, self.tz))

    def _quiet(self, t: datetime, patient_id: str | None = None) -> bool:
        tt = self._local(patient_id, t).time() if patient_id is not None else self._aware(t).astimezone(self.tz).time()
        s, e = self.quiet_start, self.quiet_end
        return (s <= tt < e) if s <= e else (tt >= s or tt < e)

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

    def send(self, patient_id: str, event: str, payload: dict, now: datetime,
             event_id: str | None = None) -> PatientMessage:
        """Urgent messages are always delivered, once per distinct event (`event_id`, e.g. the
        log timestamp): a new urgent event always gets through; the same one is not re-paged
        every cycle. Routine messages are de-duplicated, capped per local day and held during
        the patient's local quiet hours; held messages are released by `release_held`."""
        now = self._aware(now)
        urgency, text = self.compose(event, payload)
        msg = PatientMessage(patient_id, urgency, text, now)
        if not text.strip():
            msg.held_reason = "nothing to say"
            return msg
        if urgency == "urgent" and event_id is not None:
            if (patient_id, event, event_id) in self._delivered_events:
                msg.held_reason = "already delivered for this event"
                self.suppressed.append(msg)
                return msg
        if urgency != "urgent":
            msg.held_reason = self._routine_block(patient_id, event, now)
        if msg.held_reason:
            if msg.held_reason == "duplicate within 24 h":   # already said; never re-sent
                self.suppressed.append(msg)
            else:
                self.held.append(msg)
                self._held_meta[id(msg)] = event
        else:
            self._deliver(msg, event, now)
            if urgency == "urgent" and event_id is not None:
                self._delivered_events.add((patient_id, event, event_id))
        return msg

    def _routine_block(self, patient_id: str, event: str, now: datetime) -> str | None:
        last = self._last_by_key.get((patient_id, event))
        today_local = self._local(patient_id, now).date()
        today = [m for m in self.sent if m.patient_id == patient_id and m.urgency == "routine"
                 and self._local(patient_id, m.created_at).date() == today_local]
        if last and now - last < self.dedupe_window:
            return "duplicate within 24 h"
        if len(today) >= self.max_routine_per_day:
            return "daily limit for routine messages reached"
        if self._quiet(now, patient_id):
            return "quiet hours; will send in the morning"
        return None

    def _deliver(self, msg: PatientMessage, event: str, now: datetime) -> None:
        msg.delivered, msg.held_reason = True, None
        self.sent.append(msg)
        self._last_by_key[(msg.patient_id, event)] = now

    def release_held(self, now: datetime) -> list[PatientMessage]:
        """Deliver held routine messages once quiet hours / daily caps allow. A held message
        superseded by a delivered message of the same kind is dropped rather than repeated."""
        now = self._aware(now)
        released, still = [], []
        for msg in self.held:
            event = self._held_meta.get(id(msg))
            block = self._routine_block(msg.patient_id, event, now) if event else "unknown event"
            if block is None:
                self._held_meta.pop(id(msg), None)
                self._deliver(msg, event, now)
                released.append(msg)
            elif block == "duplicate within 24 h":
                self._held_meta.pop(id(msg), None)        # same message already reached the patient
                msg.held_reason = block
                self.suppressed.append(msg)
            else:
                msg.held_reason = block
                still.append(msg)
        self.held = still
        return released

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
