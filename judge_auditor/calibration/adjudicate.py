"""Multi-labeler adjudication (rule "adjudication-v1", schema labels-v2).

The calibrator needs calibrating: a single labeler's first-instinct labels
bake that labeler's own biases into "truth" (observed 2026-09-26: a human
labeler ran always-A on 7 content-identical halo items while self-reporting
low confidence). This module turns N labelers' labels into one adjudicated
gold standard, measures the labelers against each other and against the
gold, and screens for labeler-side position runs.

FROZEN RULES (adjudication-v1; see SPEC.md section 11):

- Votes are grouped by (probe, item_id). Duplicate (labeler, probe, item)
  labels raise CalibrationRefusal — same as calibrate().
- Gold = plurality of votes over {A, B, TIE}. TIE is a votable outcome:
  labelers who honestly can't tell push the gold toward TIE, which
  tie-v1 then excludes from TPR/TNR (no decisive signal claimed).
- Vote ties are broken by summed confidence per side (higher wins).
  Unbreakable ties -> gold TIE with unresolved=True. No decisive signal
  is ever manufactured from a dead heat.
- Single labeler: adjudication is passthrough (gold = their label).
  calibrate_multi() with one labeler reproduces calibrate() per-probe.
- Iron rule: judge_id must not appear among labeler_ids (case-insensitive)
  -> CalibrationRefusal. Unchanged from v1.
- Inter-rater metrics: Fleiss' kappa over items with >= 2 raters;
  pairwise Cohen's kappa (3-category, same table as metrics) over
  co-labeled items; per-labeler kappa vs the adjudicated gold.
- Position screen (a SCREEN, not a verdict): per (labeler, probe), exact
  two-sided binomial p of the observed A-count under p=0.5; flagged when
  p < 0.05 and n >= 5. A skewed A-rate is not proof of bias (the correct
  side varies by item) — it is a flag for review. KNOWN LIMITATION
  (red-team 2026-09-26): the screen tests against 0.5, not against truth,
  so a labeler in perfect agreement with skewed truth flags by
  construction — cross-check flags against labeler-vs-gold.
  With exactly 2 raters, every disagreement is a vote tie, so the
  confidence tie-break decides 100% of splits (red-team 2026-09-26);
  prefer >= 3 raters.
- Zero-denominator conventions mirror tie-v1: rates with no denominator
  report 0.0 with CI (0.0, 1.0); Fleiss with Pe == 1.0 reports 1.0 iff
  P_bar == 1.0 else 0.0.
"""

from ..scoring import binomial_cdf, binomial_sf
from .calibrate import CalibrationRefusal, _resolve_probe, report_verdicts
from .labels import load_labels
from .metrics import _CATEGORIES, _kappa_from_table, calibrate_probe

ADJUDICATION_RULE = "adjudication-v1"
POSITION_SCREEN_ALPHA = 0.05
POSITION_SCREEN_MIN_N = 5


def index_labels(labels):
    """Group label records by (probe, item_id).

    Returns {(probe, item_id): {labeler_id: (label, confidence)}}.
    Duplicate (labeler, probe, item_id) -> CalibrationRefusal.
    """
    index = {}
    seen = set()
    for lab in labels:
        probe = _resolve_probe(lab)
        key = (str(lab["labeler_id"]), probe, lab["item_id"])
        if key in seen:
            raise CalibrationRefusal(
                f"duplicate label for labeler {lab['labeler_id']!r}, "
                f"probe {probe!r}, item {lab['item_id']!r}: ambiguous "
                "calibration input")
        seen.add(key)
        item_key = (probe, lab["item_id"])
        index.setdefault(item_key, {})[str(lab["labeler_id"])] = (
            lab["label"], lab["confidence"])
    return index


def adjudicate_item(votes):
    """Adjudicate one item's votes into a gold label.

    votes: {labeler_id: (label, confidence)} with label in {A, B, TIE}.
    Returns {gold, votes, n_labelers, agreement, tie_broken, unresolved}.
    """
    if not votes:
        raise ValueError("adjudicate_item: no votes")
    counts = {c: 0 for c in _CATEGORIES}
    conf_sums = {c: 0 for c in _CATEGORIES}
    for label, conf in votes.values():
        if label not in _CATEGORIES:
            raise ValueError(f"adjudicate_item: invalid label {label!r}")
        counts[label] += 1
        conf_sums[label] += conf

    top = max(counts.values())
    leaders = [c for c in _CATEGORIES if counts[c] == top]
    tie_broken = False
    unresolved = False
    if len(leaders) == 1:
        gold = leaders[0]
    else:
        best_conf = max(conf_sums[c] for c in leaders)
        conf_leaders = [c for c in leaders if conf_sums[c] == best_conf]
        if len(conf_leaders) == 1:
            gold = conf_leaders[0]
            tie_broken = True
        else:
            gold = "TIE"  # dead heat: claim no decisive signal
            unresolved = True

    agreement = counts[gold] / len(votes)
    return {
        "gold": gold,
        "votes": {lid: {"label": lab, "confidence": conf}
                  for lid, (lab, conf) in votes.items()},
        "n_labelers": len(votes),
        "agreement": agreement,
        "tie_broken": tie_broken,
        "unresolved": unresolved,
    }


def adjudicate_all(index):
    """Adjudicate every indexed item. Returns {(probe, item_id): result}."""
    return {key: adjudicate_item(votes) for key, votes in index.items()}


def fleiss_kappa(index):
    """Fleiss' kappa over items with >= 2 raters, categories {A, B, TIE}.

    Returns (kappa, n_items). Items with < 2 raters are excluded.
    Degenerate convention mirrors tie-v1: Pe == 1.0 -> 1.0 iff P_bar == 1.0
    else 0.0; no ratable items -> (0.0, 0).
    """
    n_ij = []  # per item: {category: count}
    for votes in index.values():
        if len(votes) < 2:
            continue
        counts = {c: 0 for c in _CATEGORIES}
        for label, _ in votes.values():
            counts[label] += 1
        n_ij.append(counts)

    n_items = len(n_ij)
    if n_items == 0:
        return (0.0, 0)

    p_bar_sum = 0.0
    cat_totals = {c: 0 for c in _CATEGORIES}
    assign_total = 0
    for counts in n_ij:
        n_i = sum(counts.values())
        p_i = sum(n * (n - 1) for n in counts.values()) / (n_i * (n_i - 1))
        p_bar_sum += p_i
        for c in _CATEGORIES:
            cat_totals[c] += counts[c]
        assign_total += n_i

    p_bar = p_bar_sum / n_items
    p_e = sum((cat_totals[c] / assign_total) ** 2 for c in _CATEGORIES)
    if p_e == 1.0:
        kappa = 1.0 if p_bar == 1.0 else 0.0
    else:
        kappa = (p_bar - p_e) / (1.0 - p_e)
    return (kappa, n_items)


def pairwise_cohen(index):
    """Pairwise Cohen's kappa for every labeler pair over co-labeled items.

    Returns {labeler_a|labeler_b (sorted, "|" joined):
             {kappa, po, pe, n}}. Pairs with no shared items are omitted.
    """
    labelers = sorted({lid for votes in index.values() for lid in votes})
    out = {}
    for i, a in enumerate(labelers):
        for b in labelers[i + 1:]:
            table = {j: {h: 0 for h in _CATEGORIES} for j in _CATEGORIES}
            n = 0
            for votes in index.values():
                if a in votes and b in votes:
                    la = votes[a][0]
                    lb = votes[b][0]
                    table[la][lb] += 1
                    n += 1
            if n == 0:
                continue
            kappa, po, pe = _kappa_from_table(table)
            out[f"{a}|{b}"] = {"kappa": kappa, "po": po, "pe": pe, "n": n}
    return out


def labeler_vs_gold(index, adjudicated):
    """Each labeler's agreement with the adjudicated gold.

    Returns {labeler_id: {kappa, po, pe, n, agreement}}. This is the
    noisy-labeler finder: a labeler far below the rest on kappa vs gold
    is the one baking bias into "truth".
    """
    per_labeler = {}
    for key, votes in index.items():
        gold = adjudicated[key]["gold"]
        for lid, (label, _conf) in votes.items():
            per_labeler.setdefault(lid, []).append((label, gold))
    out = {}
    for lid, pairs in sorted(per_labeler.items()):
        table = {j: {h: 0 for h in _CATEGORIES} for j in _CATEGORIES}
        for label, gold in pairs:
            table[label][gold] += 1
        kappa, po, pe = _kappa_from_table(table)
        out[lid] = {"kappa": kappa, "po": po, "pe": pe,
                    "n": len(pairs), "agreement": po}
    return out


def _two_sided_binomial_p(k, n, p=0.5):
    """Exact two-sided p-value for observing k successes in n trials."""
    if n <= 0:
        return 1.0
    return min(1.0, 2.0 * min(binomial_sf(k, n, p), binomial_cdf(k, n, p)))


def position_screen(index):
    """Screen for labeler-side position runs: per (labeler, probe).

    Returns a list of {labeler, probe, n, n_a, a_rate, p_value, flagged},
    sorted with flagged first. flagged = p < 0.05 and n >= 5. This is a
    screen for review, NOT a bias verdict: the correct side varies by
    item, so a skewed A-rate alone proves nothing.
    """
    groups = {}
    for (probe, _item), votes in index.items():
        for lid, (label, _conf) in votes.items():
            groups.setdefault((lid, probe), []).append(label)
    out = []
    for (lid, probe), labels in sorted(groups.items()):
        n = len(labels)
        n_a = sum(1 for lab in labels if lab == "A")
        p_value = _two_sided_binomial_p(n_a, n)
        flagged = p_value < POSITION_SCREEN_ALPHA and n >= POSITION_SCREEN_MIN_N
        out.append({"labeler": lid, "probe": probe, "n": n, "n_a": n_a,
                    "a_rate": n_a / n if n else 0.0,
                    "p_value": p_value, "flagged": flagged})
    out.sort(key=lambda r: (not r["flagged"], r["p_value"]))
    return out


def calibrate_multi(report, labels, judge_id):
    """Calibrate judge verdicts against adjudicated multi-labeler gold.

    report: run-report dict (as loaded from report.json).
    labels: label records from >= 1 labelers (as from load_labels).
    judge_id: identity of the judge under audit.

    Adjudicates every item (adjudication-v1), joins the gold to the report
    verdicts by (probe, item_id), and runs calibrate_probe per probe with
    the gold as the human label. Unresolved items (gold TIE) are excluded
    from TPR/TNR by the frozen tie-v1 rule and kept in kappa.

    Returns a dict with per_probe metrics plus:
      labeler_ids, n_labels, n_unmatched, n_items, n_unresolved,
      inter_rater {fleiss_kappa, fleiss_n_items, pairwise},
      labeler_vs_gold, position_flags, adjudication {(probe, item)}.
    """
    labeler_ids = {str(l["labeler_id"]) for l in labels}
    for lid in labeler_ids:
        if lid.lower() == str(judge_id).lower():
            raise CalibrationRefusal(
                f"iron rule violated: labeler_id {lid!r} == judge_id "
                f"{judge_id!r} (case-insensitive). Judge-labeler collapse "
                "voids calibration; a different party must label.")

    verdicts = report_verdicts(report)
    index = index_labels(labels)
    adjudicated = adjudicate_all(index)

    per_probe_pairs = {}
    adjudication = {}
    n_unmatched = 0
    n_unresolved = 0
    for (probe, item_id), adj in sorted(adjudicated.items()):
        verdict = verdicts.get((probe, item_id))
        if verdict is None:
            n_unmatched += len(adj["votes"])
            continue
        per_probe_pairs.setdefault(probe, []).append((adj["gold"], verdict))
        if adj["unresolved"]:
            n_unresolved += 1
        adjudication[f"{probe}:{item_id}"] = adj

    per_probe = {name: calibrate_probe(pairs)
                 for name, pairs in sorted(per_probe_pairs.items())}
    fleiss, fleiss_n = fleiss_kappa(index)
    return {
        "protocol_sha256": None,
        "labeler_ids": sorted(labeler_ids),
        "label_schema_version": "labels-v2",
        "tie_rule": "tie-v1",
        "adjudication_rule": ADJUDICATION_RULE,
        "n_labels": len(labels),
        "n_unmatched": n_unmatched,
        "n_items": sum(len(v) for v in per_probe_pairs.values()),
        "n_unresolved": n_unresolved,
        "per_probe": per_probe,
        "inter_rater": {
            "fleiss_kappa": fleiss,
            "fleiss_n_items": fleiss_n,
            "pairwise": pairwise_cohen(index),
        },
        "labeler_vs_gold": labeler_vs_gold(index, adjudicated),
        "position_flags": position_screen(index),
        "adjudication": adjudication,
    }


def calibrate_multi_files(report_path, labels_path, judge_id):
    """Load report.json + one or more labels.jsonl files, then calibrate."""
    import json
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if isinstance(labels_path, str):
        labels_path = [labels_path]
    labels = []
    for p in labels_path:
        labels.extend(load_labels(p))
    return calibrate_multi(report, labels, judge_id)
