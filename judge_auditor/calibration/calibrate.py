"""Calibration join: labels x report verdicts, with the iron rule enforced.

calibrate() joins human labels to the judge verdicts stored in a run
report by (probe, item_id), enforces that the labeler is a different party
than the judge under audit (case-insensitive), and returns per-probe
calibration metrics via metrics.calibrate_probe().

IRON RULE: labeler_id == judge_id (case-insensitive) -> CalibrationRefusal,
a ValueError subclass. A judge grading its own labels is not calibration;
it is agreement with itself. The calibration is void, not "passing with a
warning".
"""

from pathlib import Path

from .labels import LABEL_SCHEMA_VERSION, load_labels
from .metrics import calibrate_probe


class CalibrationRefusal(ValueError):
    """The calibration was refused (e.g. judge-labeler collapse)."""


def _infer_probe(item_id):
    """Fallback probe name from an item id like "verbosity-000"."""
    if "-" in item_id:
        return item_id.rsplit("-", 1)[0]
    return item_id


_KNOWN_PROBES = None


def _known_probe_names():
    """Probe names from the probes/*.yaml registry (cached)."""
    global _KNOWN_PROBES
    if _KNOWN_PROBES is None:
        probes_dir = Path(__file__).resolve().parent.parent.parent / "probes"
        _KNOWN_PROBES = frozenset(p.stem for p in probes_dir.glob("*.yaml"))
    return _KNOWN_PROBES


def _resolve_probe(lab):
    """Probe for a label record, fail-closed.

    Uses the explicit "probe" field when present; otherwise infers from an
    item_id of the form "<probe>-<nnn>" and validates the inferred name
    against the probe registry. A record with no probe and no inferable
    probe name raises CalibrationRefusal instead of silently mis-joining:
    blinded ids like "item-015" infer the garbage probe "item", which used
    to split the (probe, item_id) index without any error. Such records
    must have the probe pinned at import (sealed map), never inferred.
    """
    probe = lab.get("probe")
    if probe:
        return probe
    inferred = _infer_probe(lab["item_id"])
    if inferred in _known_probe_names():
        return inferred
    raise CalibrationRefusal(
        f"label for item {lab['item_id']!r} carries no probe and "
        f"{inferred!r} is not a known probe: pin the probe at import "
        f"(sealed map) instead of relying on inference.")


def report_verdicts(report):
    """Index a run report's per-probe verdicts by (probe, item_id).

    Requires the report entries to carry "verdicts": [{item_id, verdict}]
    (written by report.build_report). Raises ValueError if any entry lacks
    them.
    """
    index = {}
    for entry in report.get("probes", []):
        name = entry["probe"]
        if "verdicts" not in entry:
            raise ValueError(
                f"report probe {name!r} has no 'verdicts' list: "
                "re-run with a report.py that records per-probe verdicts")
        for v in entry["verdicts"]:
            index[(name, v["item_id"])] = v["verdict"]
    return index


def calibrate(report, labels, judge_id):
    """Calibrate judge verdicts in `report` against human `labels`.

    report: run-report dict (as loaded from report.json).
    labels: list of label records (as from load_labels).
    judge_id: identity of the judge under audit.

    Returns a dict:
      {protocol_sha256 (None unless attached by the caller),
       labeler_id, label_schema_version, tie_rule,
       n_labels, n_unmatched,
       per_probe: {probe_name: calibrate_probe(...) result}}
    """
    verdicts = report_verdicts(report)

    labeler_ids = {str(l["labeler_id"]) for l in labels}
    for lid in labeler_ids:
        if lid.lower() == str(judge_id).lower():
            raise CalibrationRefusal(
                f"iron rule violated: labeler_id {lid!r} == judge_id "
                f"{judge_id!r} (case-insensitive). Judge-labeler collapse "
                "voids calibration; a different party must label.")

    seen = set()
    per_probe_pairs = {}
    n_unmatched = 0
    for lab in labels:
        probe = _resolve_probe(lab)
        key = (str(lab["labeler_id"]), probe, lab["item_id"])
        if key in seen:
            raise CalibrationRefusal(
                f"duplicate label for labeler {lab['labeler_id']!r}, "
                f"probe {probe!r}, item {lab['item_id']!r}: ambiguous "
                "calibration input")
        seen.add(key)
        verdict = verdicts.get((probe, lab["item_id"]))
        if verdict is None:
            n_unmatched += 1
            continue
        per_probe_pairs.setdefault(probe, []).append(
            (lab["label"], verdict))

    per_probe = {name: calibrate_probe(pairs)
                 for name, pairs in sorted(per_probe_pairs.items())}
    labeler_id = ",".join(sorted(labeler_ids)) if labeler_ids else ""
    return {
        "protocol_sha256": None,
        "labeler_id": labeler_id,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "tie_rule": "tie-v1",
        "n_labels": len(labels),
        "n_unmatched": n_unmatched,
        "per_probe": per_probe,
    }


def calibrate_files(report_path, labels_path, judge_id):
    """Load report.json + one or more labels.jsonl files and calibrate."""
    import json
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if isinstance(labels_path, str):
        labels_path = [labels_path]
    labels = []
    for p in labels_path:
        labels.extend(load_labels(p))
    return calibrate(report, labels, judge_id)
