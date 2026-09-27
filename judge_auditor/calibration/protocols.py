"""Pre-registered calibration protocols (protocol-v2).

freeze() builds a canonical protocol document describing exactly which
probes (name + version), item count n, generation seed, frozen decision
bars, label schema, tie rule, and metrics a calibration run must use.
The sha256 hash of the *content form* (see _content_form) is embedded in
the document as "sha256" and also written to a .sha256 sidecar file.

verify() recomputes the hash and checks the run report (and optionally the
label file) against the protocol. Any mismatch raises ProtocolMismatch
naming the field, the expected value, and the actual value.

v2 change (2026-09-26): the hash covers item-generating content only.
v1 hashed the full body including `created_at`, so two freezes of the
byte-identical battery minted different identities and the drift
monitor's compare() hard-refused legitimate cross-time comparisons on a
timestamp-only difference. v1 files still verify via the legacy path in
_check_hash; they are never silently re-hashed.
"""

import datetime
import hashlib
import json
from pathlib import Path

PROTOCOL_VERSION = "protocol-v2"
LEGACY_PROTOCOL_VERSION = "protocol-v1"

from .labels import LABEL_SCHEMA_VERSION, load_labels


class ProtocolMismatch(Exception):
    """A protocol, run report, or label file failed verification."""

    def __init__(self, field, expected, actual):
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"protocol mismatch on {field!r}: expected {expected!r}, "
            f"got {actual!r}")


def _canonical(obj):
    """Canonical JSON bytes: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _hash(obj):
    return hashlib.sha256(_canonical(obj)).hexdigest()


# Fields that define the battery and its frozen decision procedure: everything
# that can change which items are generated or how a run is graded.
# `created_at` is deliberately excluded — it is wrapper metadata, and hashing
# it meant two freezes of the identical battery produced different identities.
_HASHED_FIELDS = ("protocol_version", "probes", "n", "seed",
                  "frozen_bars", "label_schema", "tie_rule", "metrics")


def _content_form(protocol_body):
    """The item-generating content of a protocol: exactly what the v2 hash covers."""
    try:
        return {k: protocol_body[k] for k in _HASHED_FIELDS}
    except KeyError as e:
        raise ProtocolMismatch("protocol",
                               "all v2 content fields present",
                               f"missing {e}")


def build_protocol(probes, n, seed):
    """Build the canonical (hashless) protocol dict.

    probes: list of loaded probe dicts (name, version, frozen_bar).
    """
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "probes": [{"name": p["name"], "version": p["version"]}
                   for p in probes],
        "n": n,
        "seed": seed,
        "frozen_bars": {
            p["name"]: {"alpha": p["frozen_bar"]["alpha"],
                        "min_n": p["frozen_bar"]["min_n"]}
            for p in probes
        },
        "label_schema": LABEL_SCHEMA_VERSION,
        "tie_rule": "tie-v1",
        "metrics": ["tpr", "tnr", "kappa"],
    }


def freeze(probes, n, seed, out_path):
    """Freeze a protocol to out_path (+ out_path.sha256 sidecar).

    Returns the protocol dict (with the embedded "sha256" field).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(probes, n, seed)
    digest = _hash(_content_form(protocol))
    protocol_with_hash = dict(protocol)
    protocol_with_hash["sha256"] = digest
    out_path.write_text(json.dumps(protocol_with_hash, indent=2,
                                   sort_keys=True) + "\n")
    Path(str(out_path) + ".sha256").write_text(digest + "\n")
    return protocol_with_hash


def _check_hash(protocol_with_hash):
    expected = protocol_with_hash.get("sha256")
    version = protocol_with_hash.get("protocol_version")
    if version == LEGACY_PROTOCOL_VERSION:
        # v1 hashed the full body including created_at; keep verifying it
        # exactly as frozen, never re-hash it under v2 rules.
        actual = _hash({k: v for k, v in protocol_with_hash.items()
                        if k != "sha256"})
    elif version == PROTOCOL_VERSION:
        actual = _hash(_content_form(protocol_with_hash))
    else:
        raise ProtocolMismatch("protocol_version",
                               f"{LEGACY_PROTOCOL_VERSION}/{PROTOCOL_VERSION}",
                               version)
    if expected != actual:
        raise ProtocolMismatch("sha256", expected, actual)
    return {k: v for k, v in protocol_with_hash.items() if k != "sha256"}


def verify(protocol_path, run_report_path, labels_path=None):
    """Verify a run report (and optional label file) against a protocol.

    Checks: sha256 integrity; probe names/versions; n_items == protocol n;
    report seed == protocol seed; per-probe alpha/min_n == frozen bars.
    If labels_path is given, every label file must parse under labels-v1
    rules. Returns the verified protocol dict (including "sha256") on
    success.
    """
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol_with_hash = json.load(f)
    protocol = _check_hash(protocol_with_hash)

    with open(run_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    report_probes = {p["probe"]: p for p in report.get("probes", [])}

    for p in protocol["probes"]:
        name = p["name"]
        if name not in report_probes:
            raise ProtocolMismatch(f"probes.{name}", "present", "absent")
        entry = report_probes[name]
        if entry.get("version") != p["version"]:
            raise ProtocolMismatch(f"probes.{name}.version",
                                   p["version"], entry.get("version"))
        if entry.get("n_items") != protocol["n"]:
            raise ProtocolMismatch(f"probes.{name}.n_items",
                                   protocol["n"], entry.get("n_items"))
        if entry.get("seed") != protocol["seed"]:
            raise ProtocolMismatch(f"probes.{name}.seed",
                                   protocol["seed"], entry.get("seed"))
        bar = protocol["frozen_bars"][name]
        if entry.get("alpha") != bar["alpha"]:
            raise ProtocolMismatch(f"probes.{name}.alpha",
                                   bar["alpha"], entry.get("alpha"))
        if entry.get("min_n") != bar["min_n"]:
            raise ProtocolMismatch(f"probes.{name}.min_n",
                                   bar["min_n"], entry.get("min_n"))

    if labels_path is not None:
        load_labels(labels_path)  # raises ValueError naming line/field

    return protocol_with_hash
