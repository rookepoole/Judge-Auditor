"""Blinded re-scoring: measure a judge's self-consistency.

make_rescore_set() regenerates the exact items of a run report (same
probes, same seeds, same n), then per item applies an A/B side-swap decided
by random.Random(rescore_seed) and strips judge-identity cues ONLY
(self_preference probe: the "Note: Answer X was written by ..." context is
replaced with ""). All other probes' bias attributes are untouched. The
blinded set is written as JSONL [{rescore_id, system, user}] with NO probe
names; the mapping (probe, item_id, swapped) goes to a sealed JSON file the
judge must not read.

compare_rescores() maps each re-score verdict back through its swap
(A<->B; TIE stays TIE) and computes the fraction that matches the original
verdict from the report.json probe entries' "verdicts" lists, per probe and
overall, with exact binomial confidence intervals.
"""

import json
import random
from pathlib import Path

from ..probes import generate_items, load_probe, render_prompt
from ..runner import parse_verdict
from .metrics import clopper_pearson_ci

PROBES_DIR = Path(__file__).resolve().parent.parent.parent / "probes"


def _load_probe_by_name(name):
    return load_probe(PROBES_DIR / f"{name}.yaml")


def make_rescore_set(report_path, rescore_seed, out_path, sealed_path):
    """Build the blinded re-score set from a run report.

    Returns the number of items written.
    """
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    rng = random.Random(rescore_seed)
    entries = []
    for entry in report.get("probes", []):
        probe = _load_probe_by_name(entry["probe"])
        for item in generate_items(
                probe, n=entry["n_items"], seed=entry["seed"]):
            swapped = rng.random() < 0.5
            if entry["probe"] == "self_preference":
                # Strip judge-identity cues ONLY; answers and everything
                # else are untouched.
                item.context = ""
            if swapped:
                item.answer_a, item.answer_b = item.answer_b, item.answer_a
            system, user = render_prompt(item)
            entries.append({
                "probe": entry["probe"],
                "item_id": item.item_id,
                "swapped": swapped,
                "system": system,
                "user": user,
            })

    rng.shuffle(entries)

    sealed = {}
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for k, e in enumerate(entries):
            rid = f"rescore-{k:04d}"
            sealed[rid] = {"probe": e["probe"],
                           "item_id": e["item_id"],
                           "swapped": e["swapped"]}
            f.write(json.dumps({"rescore_id": rid,
                                "system": e["system"],
                                "user": e["user"]}) + "\n")
    with open(sealed_path, "w", encoding="utf-8") as f:
        json.dump(sealed, f, indent=1)
    return len(entries)


def _unswap(verdict, swapped):
    """Map a re-score verdict back to the original (unswapped) frame."""
    if swapped and verdict in ("A", "B"):
        return "B" if verdict == "A" else "A"
    return verdict  # TIE and INVALID stay


def compare_rescores(report_path, rescore_verdicts_jsonl, sealed_path):
    """Compare re-score verdicts against the original report verdicts.

    rescore_verdicts_jsonl: JSONL of {"rescore_id": str, "verdict": str}
    (verdict text is parsed with runner.parse_verdict for robustness).

    Returns {"per_probe": {name: {n, consistent, consistency, ci}},
             "overall": {n, consistent, consistency, ci}}.
    """
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    original = {}
    for entry in report.get("probes", []):
        for v in entry.get("verdicts", []):
            original[(entry["probe"], v["item_id"])] = v["verdict"]

    with open(sealed_path, "r", encoding="utf-8") as f:
        sealed = json.load(f)

    verdicts = {}
    with open(rescore_verdicts_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            d = json.loads(line)
            verdicts[d["rescore_id"]] = parse_verdict(d.get("verdict", ""))

    per_probe = {}
    total_n = total_match = 0
    for rid, v in verdicts.items():
        if rid not in sealed:
            raise ValueError(
                f"rescore verdicts line for {rid!r}: no sealed mapping")
        s = sealed[rid]
        probe, item_id, swapped = s["probe"], s["item_id"], s["swapped"]
        orig = original.get((probe, item_id))
        if orig is None:
            raise ValueError(
                f"{rid!r}: no original verdict for "
                f"({probe!r}, {item_id!r})")
        if v == "INVALID":
            continue  # unparseable re-score: excluded, not a mismatch
        back = _unswap(v, swapped)
        match = 1 if back == orig else 0
        slot = per_probe.setdefault(probe, {"n": 0, "consistent": 0})
        slot["n"] += 1
        slot["consistent"] += match
        total_n += 1
        total_match += match

    def _with_ci(slot):
        n, c = slot["n"], slot["consistent"]
        slot["consistency"] = c / n if n else 0.0
        slot["ci"] = list(clopper_pearson_ci(c, n))
        return slot

    return {
        "per_probe": {k: _with_ci(v)
                       for k, v in sorted(per_probe.items())},
        "overall": _with_ci({"n": total_n, "consistent": total_match}),
    }
