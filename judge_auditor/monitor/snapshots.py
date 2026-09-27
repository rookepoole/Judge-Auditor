"""SQLite snapshot store for drift monitoring.

A snapshot is one full battery run (all probes, one adapter, one seed)
stored as a row, linked into a tamper-evident hash chain:

    chain_hash = sha256(prev_chain_hash + canonical_json(core fields))

with prev = "GENESIS" for the first row. Editing any stored field (or
deleting a row) breaks the chain at that id (or the next surviving id),
which verify_chain() reports.

Deviation notes: see monitor/__init__.py (D1: run seed is a snapshot
field, not part of the protocol identity; D2: the drift protocol is
frozen once per (probes, n) and cached next to the db).
"""

import datetime
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

from ..calibration.protocols import freeze as freeze_protocol
from ..calibration.protocols import PROTOCOL_VERSION as CAL_PROTOCOL_VERSION
from ..probes import load_probe
from ..report import grade
from ..runner import run_all

DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "audit" / "drift.db"
GENESIS = "GENESIS"
DRIFT_PROTOCOL_SEED = 0  # pinned: run seed lives on the snapshot, not the protocol
# v2 (2026-09-26): the inner calibration hash is the protocol-v2 content
# hash (item-generating content only, no created_at), so the identity is
# stable across re-freezes of the identical battery. v1 identities remain
# valid history but are not interchangeable with v2 — compare() refuses
# across schemes, by design.
DRIFT_IDENTITY_VERSION = "drift-identity-v2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    adapter_name TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    seed INTEGER NOT NULL,
    n INTEGER NOT NULL,
    grade TEXT NOT NULL,
    label TEXT NOT NULL,
    probes_json TEXT NOT NULL,
    chain_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baseline_log (
    ts TEXT NOT NULL,
    old_id INTEGER,
    new_id INTEGER,
    reason TEXT NOT NULL
);
"""

PROBES_DIR = Path(__file__).resolve().parent.parent.parent / "probes"


def _canonical(obj) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DEFAULT_DB):
    """Create the drift tables if they do not exist."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def discover_probes():
    """All 9 battery probes, sorted by filename."""
    return [load_probe(p) for p in sorted(PROBES_DIR.glob("*.yaml"))]


def _probe_content_manifest(probes) -> dict:
    """{probe name: sha256 of its YAML file bytes} for the probes in play.

    Binds the drift protocol identity to probe *content*, not just
    name+version: editing a probe bank without bumping the version changes
    the identity, so compare() refuses instead of silently comparing
    across different instruments (red-team attack 4b, 2026-09-26).
    """
    by_name = {}
    for path in sorted(PROBES_DIR.glob("*.yaml")):
        by_name[load_probe(path)["name"]] = path
    manifest = {}
    for p in probes:
        path = by_name.get(p["name"])
        if path is None:
            raise ValueError(f"no probe file found for probe {p['name']!r}")
        manifest[p["name"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _drift_identity(calibration_sha256: str, manifest: dict) -> str:
    """Drift protocol identity: calibration hash + probe content manifest.

    Kept separate from calibration's own sha256 so the calibration
    protocol format is untouched; the monitor's identity is strictly
    stronger (a calibration-hash collision across different probe
    *contents* is impossible here).
    """
    return hashlib.sha256(
        (DRIFT_IDENTITY_VERSION + calibration_sha256
         + _canonical(manifest)).encode()
    ).hexdigest()


def _protocol_matches(body: dict, probes, n: int, manifest: dict) -> bool:
    """Does a cached protocol body describe this (probes, n) battery?"""
    # A stale scheme version never "matches": the cache is re-frozen under
    # the current protocol version, minting a new (content-defined) identity.
    if body.get("protocol_version") != CAL_PROTOCOL_VERSION:
        return False
    if body.get("n") != n:
        return False
    wanted_probes = [{"name": p["name"], "version": p["version"]}
                     for p in probes]
    if body.get("probes") != wanted_probes:
        return False
    if body.get("probe_files") != manifest:
        return False
    wanted_bars = {p["name"]: {"alpha": p["frozen_bar"]["alpha"],
                               "min_n": p["frozen_bar"]["min_n"]}
                   for p in probes}
    return body.get("frozen_bars") == wanted_bars


def _drift_protocol_path(db_path) -> Path:
    return Path(db_path).parent / "drift_protocol.json"


def ensure_drift_protocol(probes, n: int, db_path=DEFAULT_DB):
    """Return the stable drift-protocol identity for this (probes, n).

    Freezes via calibration.freeze() (seed pinned to DRIFT_PROTOCOL_SEED)
    into <dbdir>/drift_protocol.json on first use and reuses it while the
    battery definition matches. The identity stored on snapshots is NOT
    the raw calibration sha256 but a derived drift identity binding the
    calibration hash to a manifest of probe-file content hashes, so a
    probe bank edit without a version bump changes the identity and
    compare() refuses instead of comparing silently across different
    instruments. A battery change re-freezes, producing a new identity —
    which correctly makes compare() refuse against older snapshots.
    """
    init_db(db_path)
    manifest = _probe_content_manifest(probes)
    path = _drift_protocol_path(db_path)
    if path.exists():
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            body = None
        if body and _protocol_matches(body, probes, n, manifest):
            identity = body.get("drift_identity")
            if identity:
                return identity
    # Freeze to a temp file first so a crash cannot leave a half-written
    # protocol in the stable path.
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False,
                                     dir=str(path.parent)) as tmp:
        tmp_path = tmp.name
    try:
        protocol = freeze_protocol(probes, n, DRIFT_PROTOCOL_SEED, tmp_path)
        identity = _drift_identity(protocol["sha256"], manifest)
        protocol["probe_files"] = manifest
        protocol["drift_identity"] = identity
        Path(tmp_path).write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        Path(tmp_path).replace(path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return identity


def drift_protocol_hash(probes, n: int, db_path=DEFAULT_DB) -> str:
    """Stable drift protocol identity for (probes, n). Convenience wrapper."""
    return ensure_drift_protocol(probes, n, db_path)


def _probe_entry(r) -> dict:
    return {
        "probe_name": r.probe_name,
        "version": r.version,
        "expected_direction": r.expected_direction,
        "n_items": r.n_items,
        "n_valid": r.n_valid,
        "n_invalid": r.n_invalid,
        "events": r.events,
        "rate": r.rate,
        "p_value": r.p_value,
        "alpha": r.alpha,
        "two_sided": r.two_sided,
        "tripped": r.tripped,
        "inconclusive": r.inconclusive,
        "invalid_rate": r.invalid_rate,
        "seed": r.seed,
        "min_n": r.min_n,
        "reason": r.reason,
    }


def chain_entry_core(*, ts, adapter_name, protocol_sha256, seed, n, grade_,
                     label, probes) -> dict:
    """The exact dict that is chain-hashed for a snapshot row."""
    return {
        "ts": ts,
        "adapter_name": adapter_name,
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "n": n,
        "grade": grade_,
        "label": label,
        "probes": probes,
    }


def take_snapshot(results, adapter_name: str, protocol_sha256: str,
                  seed: int, n: int, label: str = "",
                  db_path=DEFAULT_DB) -> int:
    """Store one battery run as a chained snapshot row. Returns the id."""
    init_db(db_path)
    probes = sorted((_probe_entry(r) for r in results),
                    key=lambda e: e["probe_name"])
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    grade_ = grade(results)
    core = chain_entry_core(ts=ts, adapter_name=adapter_name,
                            protocol_sha256=protocol_sha256, seed=seed,
                            n=n, grade_=grade_, label=label, probes=probes)
    with _connect(db_path) as conn:
        prev = conn.execute(
            "SELECT chain_hash FROM snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = prev["chain_hash"] if prev else GENESIS
        chain_hash = hashlib.sha256(
            (prev_hash + _canonical(core)).encode()).hexdigest()
        cur = conn.execute(
            """INSERT INTO snapshots
               (ts, adapter_name, protocol_sha256, seed, n, grade, label,
                probes_json, chain_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, adapter_name, protocol_sha256, seed, n, grade_, label,
             _canonical(probes), chain_hash),
        )
        return cur.lastrowid


def snapshot_battery(adapter, adapter_name: str, n: int = 30, seed: int = 0,
                     label: str = "", db_path=DEFAULT_DB,
                     protocol_path=None, probes=None, **gen_kwargs) -> int:
    """Run the full battery via run_all and store it as a snapshot.

    The protocol hash comes from the stable drift protocol (cached next
    to the db) unless protocol_path is given, in which case that frozen
    protocol's embedded sha256 is used after a body sanity check.
    `probes` optionally restricts the battery to a subset of probe dicts
    (used for legacy/subset chat imports).
    """
    if probes is None:
        probes = discover_probes()
    if protocol_path is not None:
        body = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        if not _protocol_matches(body, probes, n):
            raise ValueError(
                f"protocol {protocol_path} does not describe the current "
                f"battery (probes/n mismatch)")
        protocol_sha256 = body["sha256"]
    else:
        protocol_sha256 = ensure_drift_protocol(probes, n, db_path)
    results = run_all(probes, adapter, n=n, seed=seed, **gen_kwargs)
    return take_snapshot(results, adapter_name, protocol_sha256, seed, n,
                         label=label, db_path=db_path)


def _row_to_snapshot(row) -> dict:
    return {
        "id": row["id"],
        "ts": row["ts"],
        "adapter_name": row["adapter_name"],
        "protocol_sha256": row["protocol_sha256"],
        "seed": row["seed"],
        "n": row["n"],
        "grade": row["grade"],
        "label": row["label"],
        "probes": json.loads(row["probes_json"]),
        "chain_hash": row["chain_hash"],
    }


def get_snapshot(db_path, snapshot_id: int) -> dict:
    """Load one snapshot (raises KeyError if absent)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM snapshots WHERE id = ?",
                           (snapshot_id,)).fetchone()
    if row is None:
        raise KeyError(f"no snapshot with id {snapshot_id}")
    return _row_to_snapshot(row)


def list_snapshots(db_path=DEFAULT_DB, since: str | None = None) -> list:
    """All snapshots, oldest first. `since` is an ISO date/datetime prefix."""
    init_db(db_path)
    with _connect(db_path) as conn:
        if since:
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE ts >= ? ORDER BY id",
                (since,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM snapshots ORDER BY id").fetchall()
    return [_row_to_snapshot(r) for r in rows]


def verify_chain(db_path=DEFAULT_DB):
    """Recompute the hash chain. Returns the first broken id, or None."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT id, ts, adapter_name, protocol_sha256, seed, n,
                      grade, label, probes_json, chain_hash
               FROM snapshots ORDER BY id""").fetchall()
    prev_hash = GENESIS
    for row in rows:
        core = chain_entry_core(
            ts=row["ts"], adapter_name=row["adapter_name"],
            protocol_sha256=row["protocol_sha256"], seed=row["seed"],
            n=row["n"], grade_=row["grade"], label=row["label"],
            probes=json.loads(row["probes_json"]))
        expected = hashlib.sha256(
            (prev_hash + _canonical(core)).encode()).hexdigest()
        if expected != row["chain_hash"]:
            return row["id"]
        prev_hash = row["chain_hash"]
    return None


def get_baseline(db_path=DEFAULT_DB):
    """The pinned baseline snapshot id, or None."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'baseline_snapshot_id'"
        ).fetchone()
    return int(row["value"]) if row else None


def pin_baseline(db_path, snapshot_id: int, reason: str = "") -> int:
    """Pin a baseline and append to the audit log. Returns the id."""
    get_snapshot(db_path, snapshot_id)  # KeyError if absent
    old_id = get_baseline(db_path)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("baseline_snapshot_id", str(snapshot_id)))
        conn.execute(
            "INSERT INTO baseline_log (ts, old_id, new_id, reason) "
            "VALUES (?, ?, ?, ?)",
            (ts, old_id, snapshot_id, reason))
    return snapshot_id
