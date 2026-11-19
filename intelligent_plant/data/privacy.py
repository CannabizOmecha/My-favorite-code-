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


SENSITIVE_FIELDS = {"cyp2c9_genotype", "genotype", "medications_json", "psychosis_history"}


def redact_for_role(record: dict, role: str) -> dict:
    """Least-privilege view: operations roles never see genetic or medication data."""
    if role in {"treating_clinician", "pharmacist", "privacy_officer"}:
        return dict(record)
    return {k: ("[REDACTED]" if k in SENSITIVE_FIELDS else v) for k, v in record.items()}
