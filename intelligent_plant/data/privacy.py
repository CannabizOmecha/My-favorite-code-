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
        # A pseudonym must stay the same patient forever. A random key per start-up would give the same
        # person a new ID after every restart, splitting their history so interaction checks and
        # titration monitoring silently miss earlier records. So a persistent vault REQUIRES a stable key
        # (from a secrets manager), and the key is checked against the one the vault was created with.
        if key is None:
            if path != ":memory:":
                raise ValueError("A persistent vault needs a stable key from your secrets manager "
                                 "(see key_from_env); refusing to generate a throwaway key.")
            key = secrets.token_bytes(32)
        if len(key) < 32:
            raise ValueError("Vault key must be at least 32 bytes.")
        self._key = key
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS map(pseudonym TEXT PRIMARY KEY, identity TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS audit(ts TEXT, actor TEXT, role TEXT, action TEXT, pseudonym TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        fp = hashlib.sha256(b"vault-key-fingerprint:" + key).hexdigest()
        row = self._db.execute("SELECT v FROM meta WHERE k='key_fingerprint'").fetchone()
        if row is None:
            self._db.execute("INSERT INTO meta VALUES('key_fingerprint', ?)", (fp,))
            self._db.commit()
        elif row[0] != fp:
            raise ValueError("Vault key does not match the key this vault was created with; patient IDs would "
                             "not line up with existing records. Use the original key.")
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


def key_from_env(var: str = "INTELLIGENT_PLANT_VAULT_KEY") -> bytes:
    """Read the vault key (hex, >= 64 hex chars) injected by your secrets manager. Never commit it."""
    import os
    raw = os.environ.get(var)
    if not raw:
        raise ValueError(f"{var} is not set; inject the vault key from your secrets manager.")
    key = bytes.fromhex(raw)
    if len(key) < 32:
        raise ValueError(f"{var} must be at least 32 bytes (64 hex characters).")
    return key


_PSEUDONYM = None


def scrub_identifiers(text: str) -> str:
    """Replace pseudonymous patient IDs before text leaves the system (minimum necessary)."""
    import re
    global _PSEUDONYM
    _PSEUDONYM = _PSEUDONYM or re.compile(r"P-[0-9a-f]{16}")
    return _PSEUDONYM.sub("[patient]", text)


SENSITIVE_FIELDS = {"cyp2c9_genotype", "genotype", "medications_json", "psychosis_history"}


def redact_for_role(record: dict, role: str) -> dict:
    """Least-privilege view: operations roles never see genetic or medication data."""
    if role in {"treating_clinician", "pharmacist", "privacy_officer"}:
        return dict(record)
    return {k: ("[REDACTED]" if k in SENSITIVE_FIELDS else v) for k, v in record.items()}
