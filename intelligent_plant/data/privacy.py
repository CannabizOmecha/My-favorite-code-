"""Pseudonymisation at the point of collection (Appendix B, rule 1). The
re-identification key lives in a SEPARATE, access-controlled store. Genetic
data is the most sensitive category (Chapter 4.5, 16.2)."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timezone


class PseudonymVault:
    """Separate store holding identity <-> pseudonym. Only roles in
    `authorized_roles` may re-identify; every access is audit-logged."""

    def __init__(self, path: str = ":memory:", key: bytes | None = None,
                 authorized_roles: frozenset[str] = frozenset({"privacy_officer", "treating_clinician"})):
        if key is None and path != ":memory:":
            # A random key on a persistent vault would give the same patient a new pseudonym after
            # every restart, silently splitting their history. Require the key to be supplied.
            raise ValueError("a persistent PseudonymVault requires an explicit key")
        self._key = key or secrets.token_bytes(32)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS map(pseudonym TEXT PRIMARY KEY, identity TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS audit(ts TEXT, actor TEXT, role TEXT, action TEXT, pseudonym TEXT)")
        self.authorized_roles = authorized_roles

    def pseudonymize(self, identity: str) -> str:
        p = "P-" + hmac.new(self._key, identity.encode(), hashlib.sha256).hexdigest()[:16]
        self._db.execute("INSERT OR IGNORE INTO map VALUES(?,?)", (p, identity))
        self._db.commit()
        return p

    def reidentify(self, pseudonym: str, actor: str, role: str, purpose: str) -> str:
        allowed = role in self.authorized_roles
        self._db.execute("INSERT INTO audit VALUES(?,?,?,?,?)",
                         (datetime.now(timezone.utc).isoformat(), actor, role,
                          f"reidentify:{'granted' if allowed else 'denied'}:{purpose}", pseudonym))
        self._db.commit()
        if not allowed:
            raise PermissionError(f"role {role!r} may not re-identify patients")
        row = self._db.execute("SELECT identity FROM map WHERE pseudonym=?", (pseudonym,)).fetchone()
        if not row:
            raise KeyError(pseudonym)
        return row[0]

    def audit_log(self) -> list[tuple]:
        return list(self._db.execute("SELECT * FROM audit ORDER BY ts"))


SENSITIVE_FIELDS = {"cyp2c9_genotype", "genotype", "medications", "medications_json", "psychosis_history",
                    "psychosis_history_personal", "psychosis_history_family_strong",
                    "substance_use_disorder_history", "pregnant_or_lactating", "unstable_cardiovascular"}
# Any key containing one of these is sensitive too (e.g. "synthase_genotype", "psychosis_notes").
SENSITIVE_SUBSTRINGS = ("genotype", "medication", "psychosis", "substance_use", "pregnan")


def _sensitive(key) -> bool:
    k = str(key).lower()
    return k in SENSITIVE_FIELDS or any(s in k for s in SENSITIVE_SUBSTRINGS)


def _redact(v):
    if isinstance(v, dict):
        return {k: ("[REDACTED]" if _sensitive(k) else _redact(x)) for k, x in v.items()}
    if isinstance(v, list):
        return [_redact(x) for x in v]
    return v


def redact_for_role(record: dict, role: str) -> dict:
    """Least-privilege view: operations roles never see genetic, medication or mental-health data,
    at any nesting depth."""
    if role in {"treating_clinician", "pharmacist", "privacy_officer"}:
        return dict(record)
    return _redact(record)
