"""Append-only, hash-chained record store implementing Appendix B and the
Chapter 16.2 governance checklist.

Rules enforced here:
  * Never overwrite a record; corrections append a new version with a reason code.
  * Every record carries provenance: source, operator, instrument, timestamp.
  * Validation (required fields, ranges, units, cross-field) runs BEFORE acceptance.
  * The hash chain makes retrospective alteration detectable. It does NOT make
    an entry true (the oracle problem, Chapter 12.2).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

Validator = Callable[[dict], list[str]]


def _rng(field: str, lo: float, hi: float) -> Validator:
    def v(d: dict) -> list[str]:
        x = d.get(field)
        if x is None:
            return []
        try:
            fx = float(x)
        except (TypeError, ValueError):
            return [f"{field}={x!r} is not numeric"]
        return [] if lo <= fx <= hi else [f"{field}={x} outside [{lo}, {hi}]"]   # NaN fails the range
    return v


def _vpd_consistent(d: dict) -> list[str]:
    from ..science.environment import vpd_air
    if {"temp_air", "rh", "vpd_air"} <= d.keys() and d["vpd_air"] is not None:
        expected = vpd_air(d["temp_air"], d["rh"])
        if abs(expected - d["vpd_air"]) > 0.05:
            return [f"vpd_air {d['vpd_air']} inconsistent with T/RH (expected {expected:.2f})"]
    return []


def _total_thc_bound(d: dict) -> list[str]:
    if d.get("unit_mg_thc") is not None and d["unit_mg_thc"] < 0:
        return ["unit_mg_thc negative"]
    return []


# Appendix B, Table B.1 - required key fields and entry validators.
SCHEMA: dict[str, dict[str, Any]] = {
    "genotype": {"required": ["genotype_id", "cultivar_name", "synthase_genotype", "marker_panel_version", "source"],
                 "validators": []},
    "environment_log": {"required": ["room_id", "timestamp", "temp_air", "rh", "sensor_cal_date"],
                        "id_fields": ["room_id", "timestamp"],
                        "validators": [_rng("temp_air", -5, 50), _rng("temp_leaf", -5, 50), _rng("rh", 0, 100),
                                       _rng("co2", 0, 5000), _rng("ppfd", 0, 3000), _vpd_consistent]},
    "batch": {"required": ["batch_id", "genotype_id", "room_id", "start_date"],
              "validators": [_rng("yield_g", 0, 1e7), _rng("dli_cumulative", 0, 1e5)]},
    "coa": {"required": ["coa_id", "batch_id", "lab_id", "method", "thc", "thca", "cbd", "cbda", "test_date"],
            "validators": [_rng("thc", 0, 100), _rng("thca", 0, 100), _rng("cbd", 0, 100), _rng("cbda", 0, 100)]},
    "product": {"required": ["product_id", "batch_id", "coa_id", "form", "route", "unit_mg_thc", "unit_mg_cbd", "lot"],
                "validators": [_total_thc_bound, _rng("unit_mg_cbd", 0, 1e5)]},
    "patient": {"required": ["patient_id", "age_band", "indication", "consent_version"], "validators": []},
    "administration": {"required": ["admin_id", "patient_id", "product_id", "coa_id", "timestamp", "dose_units",
                                    "dose_mg_thc", "dose_mg_cbd", "route"],
                       "validators": [_rng("dose_mg_thc", 0, 1e4), _rng("dose_mg_cbd", 0, 1e5)]},
    "outcome": {"required": ["outcome_id", "admin_id", "instrument", "score_pre", "score_post", "interval_min"],
                "validators": [_rng("score_pre", 0, 10), _rng("score_post", 0, 10), _rng("interval_min", 0, 10080)]},
    "model_registry": {"required": ["model_id", "version", "training_snapshot_id", "validation_metrics",
                                    "approver", "owner"], "validators": []},
    "calibration": {"required": ["instrument_id", "cal_date", "next_due", "owner"],
                    "id_fields": ["instrument_id", "cal_date"], "validators": []},
    "decision": {"required": ["proposal_id", "agent", "action", "disposition"], "validators": []},
    "approval": {"required": ["proposal_id", "task", "owner_role", "status"], "validators": []},
    "inventory_event": {"required": ["event_id", "item_id", "event", "qty_g"], "validators": []},
}


class ValidationError(ValueError):
    def __init__(self, entity: str, errors: list[str]):
        super().__init__(f"{entity}: " + "; ".join(errors))
        self.errors = errors


@dataclass
class Provenance:
    source: str
    operator: str
    instrument: str | None = None
    calibrated: bool | None = None


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


class RecordStore:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute("""CREATE TABLE IF NOT EXISTS records(
            seq INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT, entity_id TEXT, version INTEGER,
            data TEXT, source TEXT, operator TEXT, instrument TEXT, calibrated INTEGER,
            recorded_at TEXT, reason_code TEXT, prev_hash TEXT, hash TEXT)""")
        self._db.execute("CREATE INDEX IF NOT EXISTS ix_ent ON records(entity, entity_id)")

    # -- write path -------------------------------------------------------
    def validate(self, entity: str, data: dict) -> list[str]:
        spec = SCHEMA.get(entity)
        if spec is None:
            return [f"unknown entity {entity!r}"]
        errs = [f"missing {f}" for f in spec["required"] if data.get(f) in (None, "")]
        for v in spec["validators"]:
            errs.extend(v(data))
        return errs

    def _entity_id(self, entity: str, data: dict) -> str:
        fields = SCHEMA[entity].get("id_fields") or [SCHEMA[entity]["required"][0]]
        # Escape the separator so ("A|2026", "x") and ("A", "2026|x") cannot collide.
        return "|".join(str(data[f]).replace("\\", "\\\\").replace("|", "\\|") for f in fields)

    def append(self, entity: str, data: dict, prov: Provenance, reason_code: str | None = None) -> int:
        errs = self.validate(entity, data)
        if errs:
            raise ValidationError(entity, errs)
        eid = self._entity_id(entity, data)
        with self._lock:
            cur = self._db.execute("SELECT MAX(version) FROM records WHERE entity=? AND entity_id=?", (entity, eid))
            prev_v = cur.fetchone()[0]
            if prev_v is not None and not reason_code:
                raise ValidationError(entity, [f"{eid} exists; corrections require a reason_code"])
            version = (prev_v or 0) + 1
            row = self._db.execute("SELECT hash FROM records ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = row[0] if row else "GENESIS"
            ts = datetime.now(timezone.utc).isoformat()
            cal = None if prov.calibrated is None else bool(prov.calibrated)   # hashed as stored
            payload = _canon([entity, eid, version, data, prov.source, prov.operator, prov.instrument,
                              cal, ts, reason_code, prev_hash])
            h = hashlib.sha256(payload.encode()).hexdigest()
            cur = self._db.execute(
                "INSERT INTO records(entity,entity_id,version,data,source,operator,instrument,calibrated,"
                "recorded_at,reason_code,prev_hash,hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (entity, eid, version, _canon(data), prov.source, prov.operator, prov.instrument,
                 None if prov.calibrated is None else int(prov.calibrated), ts, reason_code, prev_hash, h))
            self._db.commit()
            return cur.lastrowid

    def correct(self, entity: str, data: dict, prov: Provenance, reason_code: str) -> int:
        return self.append(entity, data, prov, reason_code=reason_code)

    # -- read path --------------------------------------------------------
    def get(self, entity: str, entity_id: str) -> dict | None:
        row = self._db.execute("SELECT data FROM records WHERE entity=? AND entity_id=? ORDER BY version DESC LIMIT 1",
                               (entity, str(entity_id))).fetchone()
        return json.loads(row[0]) if row else None

    def history(self, entity: str, entity_id: str) -> list[dict]:
        rows = self._db.execute("SELECT version, data, reason_code, operator, recorded_at FROM records "
                                "WHERE entity=? AND entity_id=? ORDER BY version", (entity, str(entity_id)))
        return [{"version": v, "data": json.loads(d), "reason_code": r, "operator": o, "recorded_at": t}
                for v, d, r, o, t in rows]

    def all(self, entity: str, where: Callable[[dict], bool] | None = None) -> list[dict]:
        rows = self._db.execute(
            "SELECT r.data FROM records r JOIN (SELECT entity_id, MAX(version) v FROM records WHERE entity=? "
            "GROUP BY entity_id) m ON r.entity_id=m.entity_id AND r.version=m.v WHERE r.entity=? ORDER BY r.seq",
            (entity, entity))
        out = [json.loads(d) for (d,) in rows]
        return [o for o in out if where is None or where(o)]

    def head(self) -> tuple[str, int]:
        """(latest hash, record count): publish this to an external anchor so that truncating
        the tail of the chain - which leaves the remaining links valid - is detectable."""
        row = self._db.execute("SELECT hash FROM records ORDER BY seq DESC LIMIT 1").fetchone()
        return (row[0] if row else "GENESIS"), self.count()

    def verify_chain(self, expected_head: str | None = None,
                     expected_count: int | None = None) -> tuple[bool, int | None]:
        """Recompute every hash. Returns (ok, first_bad_seq). Pass a previously anchored head
        and/or count to also detect deletion of the most recent records."""
        prev = "GENESIS"
        n, head_seen = 0, expected_head in (None, "GENESIS")
        for row in self._db.execute("SELECT seq,entity,entity_id,version,data,source,operator,instrument,calibrated,"
                                    "recorded_at,reason_code,prev_hash,hash FROM records ORDER BY seq"):
            seq, ent, eid, ver, data, src, op, ins, cal, ts, rc, ph, h = row
            if ph != prev:
                return False, seq
            payload = _canon([ent, eid, ver, json.loads(data), src, op, ins,
                              None if cal is None else bool(cal), ts, rc, ph])
            if hashlib.sha256(payload.encode()).hexdigest() != h:
                return False, seq
            prev = h
            n += 1
            head_seen = head_seen or h == expected_head
        if (expected_count is not None and n < expected_count) or not head_seen:
            return False, n + 1      # records after this point are missing
        return True, None

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    # -- traceability (Chapter 16.1) -------------------------------------
    def trace_outcome(self, outcome_id: str) -> dict:
        """outcome -> administration -> product + COA actually consumed -> batch ->
        room/environment + genotype. Each break is reported, not papered over."""
        chain: dict[str, Any] = {"breaks": []}
        o = self.get("outcome", outcome_id)
        chain["outcome"] = o
        a = o and self.get("administration", o["admin_id"])
        chain["administration"] = a
        if not a:
            chain["breaks"].append("outcome->administration")
            return chain
        chain["product"] = p = self.get("product", a["product_id"])
        if not p:
            chain["breaks"].append("administration->product")
        chain["coa"] = c = self.get("coa", a["coa_id"])      # COA on the admin record (Appendix B rule 3)
        if not c:
            chain["breaks"].append("administration->coa")
        b = self.get("batch", (c or p or {}).get("batch_id", ""))
        chain["batch"] = b
        if not b:
            chain["breaks"].append("coa->batch")
            return chain
        chain["genotype"] = self.get("genotype", b["genotype_id"]) or chain["breaks"].append("batch->genotype")
        env = self.all("environment_log", lambda e: e["room_id"] == b["room_id"])
        chain["environment_records"] = len(env)
        if not env:
            chain["breaks"].append("batch->environment")
        return chain


def bulk_append(store: RecordStore, entity: str, rows: Iterable[dict], prov: Provenance) -> tuple[int, list[str]]:
    ok, rejected = 0, []
    for r in rows:
        try:
            store.append(entity, r, prov)
            ok += 1
        except ValidationError as e:
            rejected.append(str(e))
    return ok, rejected
