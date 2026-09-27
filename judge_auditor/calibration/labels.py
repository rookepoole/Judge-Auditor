"""Human-label ingestion for calibration (schema labels-v2, accepts v1).

JSONL schema v2 ("labels-v2"): each non-blank line is one JSON object::

    {"item_id": str, "label": "A"|"B"|"TIE",
     "labeler_id": str, "timestamp": ISO-8601 str,
     "confidence": 1|2|3 (optional)}

"confidence" is the labeler's self-reported confidence in this label:
1 = guessing ("I don't trust myself on this one"), 2 = leaning,
3 = confident. It feeds adjudication-v1 (tie-breaks) and lets low-trust
labels be down-weighted instead of treated as equal gold.

BACKWARD COMPATIBILITY (frozen): v1 records (no "confidence") are
accepted; missing confidence normalizes to 2 ("unrated", the midpoint —
the least presumptuous default). A v1 file therefore loads identically
under v2, with every record at confidence 2.

An optional extra "probe" field is accepted and validated as a string when
present (it pins the probe for the (probe, item_id) join; import-labels writes
it). Other extra fields are ignored.

IRON RULE (see SPEC.md): the labeler must be a different party than the judge
under audit. Judge-labeler collapse voids calibration. Enforced in
calibrate.calibrate() / adjudicate.calibrate_multi(), not here: this module
only guarantees the labels parse.

Validation errors raise ValueError naming the physical line number (1-based)
and the offending field. Blank lines are skipped.
"""

import datetime
import json

LABEL_SCHEMA_VERSION = "labels-v2"
VALID_LABELS = ("A", "B", "TIE")
VALID_CONFIDENCE = (1, 2, 3)
DEFAULT_CONFIDENCE = 2  # v1 records: unrated -> midpoint (frozen)


def _check(cond, line_no, field, msg):
    if not cond:
        raise ValueError(
            f"labels line {line_no}: invalid field {field!r}: {msg}")


def validate_label_record(rec, line_no):
    """Schema-validate one decoded label record. Returns the record.

    Raises ValueError naming the line number and field on any violation.
    """
    _check(isinstance(rec, dict), line_no, "<record>",
           f"expected a JSON object, got {type(rec).__name__}")
    _check(isinstance(rec.get("item_id"), str) and rec["item_id"].strip(),
           line_no, "item_id", "expected a non-empty string")
    _check(rec.get("label") in VALID_LABELS, line_no, "label",
           f"expected one of {VALID_LABELS}, got {rec.get('label')!r}")
    _check(isinstance(rec.get("labeler_id"), str)
           and rec["labeler_id"].strip(),
           line_no, "labeler_id", "expected a non-empty string")
    ts = rec.get("timestamp")
    _check(isinstance(ts, str) and ts.strip(), line_no, "timestamp",
           "expected a non-empty ISO-8601 string")
    try:
        datetime.datetime.fromisoformat(ts)
    except ValueError:
        raise ValueError(
            f"labels line {line_no}: invalid field 'timestamp': "
            f"not ISO-8601: {ts!r}")
    if "probe" in rec:
        _check(isinstance(rec["probe"], str) and rec["probe"].strip(),
               line_no, "probe",
               "expected a non-empty string when present")
    if "confidence" in rec:
        _check(type(rec["confidence"]) is int
               and rec["confidence"] in VALID_CONFIDENCE, line_no,
               "confidence",
               f"expected one of {VALID_CONFIDENCE} (int), got "
               f"{rec['confidence']!r} (1=guessing, 2=leaning, 3=confident)")
    else:
        # v1 record: normalize to the frozen default; the loader always
        # emits confidence so downstream code never branches on presence.
        rec["confidence"] = DEFAULT_CONFIDENCE
    return rec


def load_labels(path):
    """Load and schema-validate a labels.jsonl file.

    Returns a list of record dicts in file order. Raises ValueError naming
    the line number and field for the first malformed record.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"labels line {line_no}: not valid JSON: {e}") from e
            records.append(validate_label_record(rec, line_no))
    return records
