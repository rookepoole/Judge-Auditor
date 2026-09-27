"""Drift monitoring for the Judge Auditor.

Snapshot the full 9-probe battery over time into SQLite, pin baselines,
and compare snapshots with multiplicity-corrected drift tests.

Deviation notes (documented, deliberate — see Builder A report 2026-09-26):
- D4 (2026-09-26): protocol-v2 / drift-identity-v2. The calibration hash
  now covers item-generating content only (``created_at`` excluded), so a
  re-freeze of the identical battery mints the identical calibration hash
  and the identical drift identity. This removes the ``created_at`` hazard
  at the root: D1's workaround note and D2's cache-everything rationale
  are superseded (kept as history below). The cache at
  ``<dbdir>/drift_protocol.json`` is still reused when the battery matches,
  but a cache miss no longer forks the identity. v1 identities remain valid
  ledger history; compare() refuses across v1/v2 schemes by design, and
  the baseline was re-registered under v2 via new snapshot rows (never
  edited in place) with the re-pinning recorded in baseline_log.
- D1: the drift protocol identity covers (probes, n) with the protocol
  seed pinned to 0; the run seed is recorded per snapshot and surfaced as
  ``seed_mismatch``, not as a protocol change. Rationale (pre-v2): the
  calibration ``freeze()`` hash covered (created_at, probes, n, seed);
  hashing the run seed would make the specified "seed mismatch -> proceed
  with flag" branch unreachable, and ``created_at`` (microsecond precision)
  would make every cross-time comparison hard-refuse on protocol mismatch.
  [Superseded by D4: v2 hashes content only; the seed note still stands.]
- D2: the drift protocol is frozen once per (probes, n) and cached at
  ``<dbdir>/drift_protocol.json``; later snapshots reuse it. A fresh freeze
  per snapshot would change ``created_at`` every time, so ``compare()``
  would refuse every real weekly comparison — defeating the purpose of
  drift monitoring. The battery definition is re-checked on every
  snapshot; any change re-freezes and the new hash correctly refuses
  comparison against older snapshots.
  [Superseded by D4: under v2 a fresh freeze of the same battery yields
  the same identity, so the cache is an optimization, not a correctness
  crutch. The re-check on every snapshot is unchanged.]
- D3 (2026-09-26, red-team repair): the snapshot protocol identity is a
  derived drift identity = sha256("drift-identity-v1" + calibration sha256
  + canonical({probe name: sha256 of its YAML file bytes})), not the raw
  calibration sha256. The calibration hash covers probe name+version
  only, so a bank edit without a version bump previously left the
  identity unchanged — compare() would then run silently across
  different instruments. The calibration protocol format itself is
  untouched; the extra binding lives in the monitor.
"""

from .drift import (
    DRIFT_ALPHA,
    MIN_ABS_SHIFT,
    compare,
    holm_reject,
)
from .snapshots import (
    DEFAULT_DB,
    chain_entry_core,
    drift_protocol_hash,
    ensure_drift_protocol,
    get_baseline,
    get_snapshot,
    init_db,
    list_snapshots,
    pin_baseline,
    snapshot_battery,
    take_snapshot,
    verify_chain,
)

__all__ = [
    "DRIFT_ALPHA",
    "MIN_ABS_SHIFT",
    "DEFAULT_DB",
    "chain_entry_core",
    "compare",
    "drift_protocol_hash",
    "ensure_drift_protocol",
    "get_baseline",
    "get_snapshot",
    "holm_reject",
    "init_db",
    "list_snapshots",
    "pin_baseline",
    "snapshot_battery",
    "take_snapshot",
    "verify_chain",
]
