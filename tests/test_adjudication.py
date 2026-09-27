"""Verify-gated tests for labels-v2 + adjudication-v1.

The load-bearing test is the planted-bias scenario: three synthetic
labelers (one clean, one always-A position run like the observed human
halo run, one random) label items with known truth. The machinery must:
(a) adjudicate the clean signal back, (b) flag the position-biased
labeler in position_screen, (c) rank it lowest in labeler_vs_gold.

Also: v1 backward compatibility, confidence validation, tie-break rules,
Fleiss hand-checks, iron-rule refusal, duplicate refusal, and the
consistency invariant — single-labeler calibrate_multi reproduces
calibrate() per-probe exactly on synthetic fixtures.
"""

import json
import os
import tempfile
import unittest

from judge_auditor.calibration import (
    ADJUDICATION_RULE,
    CalibrationRefusal,
    adjudicate_item,
    calibrate,
    calibrate_multi,
    fleiss_kappa,
    index_labels,
    labeler_vs_gold,
    load_labels,
    pairwise_cohen,
    position_screen,
)

def _demo_report():
    """Minimal synthetic run report: 2 probes x 4 items with verdicts."""
    return {
        "judge_id": "demo-judge",
        "probes": [
            {"probe": "halo",
             "verdicts": [{"item_id": f"halo-{i:03d}", "verdict": v}
                          for i, v in enumerate(["A", "B", "A", "TIE"])]},
            {"probe": "bandwagon",
             "verdicts": [{"item_id": f"bandwagon-{i:03d}", "verdict": v}
                          for i, v in enumerate(["B", "B", "A", "A"])]},
        ],
    }


def _demo_labels(labeler_id="demo-rater"):
    """Synthetic labels that join _demo_report() on (probe, item_id)."""
    rows = []
    for probe, verdicts in (("halo", ["A", "B", "A", "TIE"]),
                            ("bandwagon", ["B", "B", "A", "A"])):
        for i, lab in enumerate(verdicts):
            rows.append(_rec(f"{probe}-{i:03d}", lab, labeler_id,
                             probe=probe))
    return rows


def _write_labels(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _rec(item_id, label, labeler_id, confidence=None, probe=None):
    r = {"item_id": item_id, "label": label, "labeler_id": labeler_id,
         "timestamp": "2026-09-26T12:00:00+00:00"}
    if confidence is not None:
        r["confidence"] = confidence
    if probe is not None:
        r["probe"] = probe
    return r


class TestLabelsV2(unittest.TestCase):
    def test_v1_backward_compat_confidence_defaults_to_2(self):
        # v1 records carry no confidence field: all must load with the
        # frozen midpoint default of 2.
        rows = [_rec(f"halo-{i:03d}", "A", "demo-rater", probe="halo")
                for i in range(49)]
        recs = load_labels(_write_labels(rows))
        self.assertEqual(len(recs), 49)
        self.assertTrue(all(r["confidence"] == 2 for r in recs))

    def test_confidence_accepted_1_to_3(self):
        p = _write_labels([
            _rec("i1", "A", "ann", confidence=1, probe="halo"),
            _rec("i2", "B", "ann", confidence=3, probe="halo"),
        ])
        try:
            recs = load_labels(p)
            self.assertEqual([r["confidence"] for r in recs], [1, 3])
        finally:
            os.unlink(p)

    def test_confidence_rejected_out_of_range(self):
        for bad in (0, 4, "high", 2.0, True):
            p = _write_labels([_rec("i1", "A", "ann", confidence=bad,
                                    probe="halo")])
            try:
                with self.assertRaises(ValueError,
                                      msg=f"confidence={bad!r} accepted"):
                    load_labels(p)
            finally:
                os.unlink(p)

    def test_confidence_null_rejected(self):
        # explicit JSON null is not "missing" — it is malformed.
        fd, p = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write('{"item_id": "i1", "label": "A", "labeler_id": "ann",'
                    ' "timestamp": "2026-09-26T12:00:00+00:00",'
                    ' "confidence": null, "probe": "halo"}\n')
        try:
            with self.assertRaises(ValueError):
                load_labels(p)
        finally:
            os.unlink(p)

    def test_confidence_error_names_line_and_field(self):
        p = _write_labels([
            _rec("i1", "A", "ann", probe="halo"),
            _rec("i2", "B", "ann", confidence=9, probe="halo"),
        ])
        try:
            with self.assertRaises(ValueError) as ctx:
                load_labels(p)
            msg = str(ctx.exception)
            self.assertIn("line 2", msg)
            self.assertIn("confidence", msg)
        finally:
            os.unlink(p)


class TestAdjudicateItem(unittest.TestCase):
    def test_plurality_wins(self):
        r = adjudicate_item({"a": ("A", 3), "b": ("A", 1), "c": ("B", 3)})
        self.assertEqual(r["gold"], "A")
        self.assertFalse(r["tie_broken"])
        self.assertFalse(r["unresolved"])
        self.assertAlmostEqual(r["agreement"], 2 / 3)

    def test_vote_tie_broken_by_confidence(self):
        r = adjudicate_item({"a": ("A", 1), "b": ("B", 3)})
        self.assertEqual(r["gold"], "B")
        self.assertTrue(r["tie_broken"])
        self.assertFalse(r["unresolved"])

    def test_dead_heat_goes_tie_unresolved(self):
        r = adjudicate_item({"a": ("A", 3), "b": ("B", 3)})
        self.assertEqual(r["gold"], "TIE")
        self.assertTrue(r["unresolved"])

    def test_tie_is_votable(self):
        r = adjudicate_item({"a": ("TIE", 3), "b": ("TIE", 2),
                             "c": ("A", 3)})
        self.assertEqual(r["gold"], "TIE")
        self.assertFalse(r["unresolved"])  # clean plurality, not a dead heat

    def test_single_labeler_passthrough(self):
        r = adjudicate_item({"a": ("B", 1)})
        self.assertEqual(r["gold"], "B")
        self.assertAlmostEqual(r["agreement"], 1.0)

    def test_provenance_records_votes(self):
        r = adjudicate_item({"a": ("A", 3), "b": ("B", 1)})
        self.assertEqual(r["votes"]["a"], {"label": "A", "confidence": 3})
        self.assertEqual(r["n_labelers"], 2)


class TestFleiss(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        idx = {("p", f"i{i}"): {"a": ("A", 3), "b": ("A", 2), "c": ("A", 1)}
               for i in range(5)}
        kappa, n = fleiss_kappa(idx)
        self.assertAlmostEqual(kappa, 1.0)
        self.assertEqual(n, 5)

    def test_hand_computed_case(self):
        # 2 items, 3 raters each.
        # item1: A,A,B ; item2: B,B,B
        idx = {("p", "i1"): {"a": ("A", 2), "b": ("A", 2), "c": ("B", 2)},
               ("p", "i2"): {"a": ("B", 2), "b": ("B", 2), "c": ("B", 2)}}
        kappa, n = fleiss_kappa(idx)
        # P1 = (2*1 + 1*0)/(3*2) = 1/3 ; P2 = (3*2)/(3*2) = 1
        # P_bar = (1/3 + 1)/2 = 2/3
        # p_A = 2/6 = 1/3, p_B = 4/6 = 2/3 ; P_e = 1/9 + 4/9 = 5/9
        # kappa = (2/3 - 5/9)/(1 - 5/9) = (1/9)/(4/9) = 0.25
        self.assertAlmostEqual(kappa, 0.25, places=9)
        self.assertEqual(n, 2)

    def test_single_rater_items_excluded(self):
        idx = {("p", "i1"): {"a": ("A", 2)},
               ("p", "i2"): {"a": ("A", 2), "b": ("A", 2)}}
        kappa, n = fleiss_kappa(idx)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(kappa, 1.0)

    def test_no_ratable_items(self):
        kappa, n = fleiss_kappa({("p", "i1"): {"a": ("A", 2)}})
        self.assertEqual((kappa, n), (0.0, 0))


class TestPlantedBiasScenario(unittest.TestCase):
    """Three synthetic labelers, known truth; the machinery must find the
    planted position-biased labeler and still adjudicate the clean signal."""

    def setUp(self):
        # 10 items, truth alternating A/B/A/B...; probe "halo".
        # clean: truth always (conf 3). biased: always A (conf 2).
        # noisy: truth except items 0 and 4 (conf 1) — wrong only on
        # truth-A items, so clean+biased still outvote it 2-to-1 and the
        # gold recovers truth on all 10 despite the plant.
        rows = []
        self.truth = {}
        noisy_wrong = {0, 4}
        for i in range(10):
            item = f"item-{i:03d}"
            truth = "A" if i % 2 == 0 else "B"
            self.truth[item] = truth
            noisy_label = ("B" if truth == "A" else "A") \
                if i in noisy_wrong else truth
            rows.append(_rec(item, truth, "clean", confidence=3,
                             probe="halo"))
            rows.append(_rec(item, "A", "biased", confidence=2,
                             probe="halo"))
            rows.append(_rec(item, noisy_label, "noisy", confidence=1,
                             probe="halo"))
        self.path = _write_labels(rows)
        self.labels = load_labels(self.path)
        self.index = index_labels(self.labels)

    def tearDown(self):
        os.unlink(self.path)

    def test_adjudication_recovers_truth_despite_plant(self):
        from judge_auditor.calibration import adjudicate_all
        adj = adjudicate_all(self.index)
        for item, truth in self.truth.items():
            self.assertEqual(adj[("halo", item)]["gold"], truth,
                             f"gold wrong on {item}")

    def test_position_screen_flags_biased_labeler(self):
        flags = position_screen(self.index)
        flagged = {(f["labeler"], f["probe"]) for f in flags if f["flagged"]}
        self.assertIn(("biased", "halo"), flagged)
        # clean labeler: 5 A / 10 -> p = 1.0, not flagged
        clean_flags = [f for f in flags
                       if f["labeler"] == "clean" and f["flagged"]]
        self.assertEqual(clean_flags, [])

    def test_position_screen_flags_perfect_labeler_on_skewed_truth(self):
        # RED-TEAM attack 1 (SUBSTANTIATED, 2026-09-26): the screen tests
        # the A-rate against 0.5, not against truth. A labeler in perfect
        # agreement with all-A truth flags by construction. Documented in
        # SPEC.md 11c as a known limitation; locked here so the behavior
        # cannot drift silently.
        rows = [_rec(f"i{i}", "A", "good", confidence=3, probe="halo")
                for i in range(7)]
        p = _write_labels(rows)
        try:
            flags = position_screen(index_labels(load_labels(p)))
        finally:
            os.unlink(p)
        self.assertTrue(flags[0]["flagged"])

    def test_two_rater_splits_resolved_by_confidence(self):
        # RED-TEAM attack 4 (SUBSTANTIATED, 2026-09-26): with exactly 2
        # raters every disagreement is a vote tie, so the confidence
        # tie-break decides 100% of splits. Documented in SPEC.md 11d.
        from judge_auditor.calibration import adjudicate_all
        rows = []
        for i in range(10):
            rows.append(_rec(f"i{i}", "A" if i % 2 == 0 else "B",
                             "confident", confidence=3, probe="halo"))
            rows.append(_rec(f"i{i}", "B" if i % 2 == 0 else "A",
                             "meek", confidence=1, probe="halo"))
        p = _write_labels(rows)
        try:
            adj = adjudicate_all(index_labels(load_labels(p)))
        finally:
            os.unlink(p)
        for i in range(10):
            self.assertEqual(adj[("halo", f"i{i}")]["gold"],
                             "A" if i % 2 == 0 else "B")

    def test_labeler_vs_gold_ranks_biased_lowest(self):
        from judge_auditor.calibration import adjudicate_all
        adj = adjudicate_all(self.index)
        vg = labeler_vs_gold(self.index, adj)
        self.assertLess(vg["biased"]["kappa"], vg["clean"]["kappa"])
        self.assertLess(vg["biased"]["kappa"], vg["noisy"]["kappa"])
        self.assertAlmostEqual(vg["clean"]["kappa"], 1.0)

    def test_pairwise_cohen_clean_vs_biased(self):
        pw = pairwise_cohen(self.index)
        self.assertIn("biased|clean", pw)
        # clean=truth alternating, biased=always A: agreement 0.5, kappa 0
        self.assertAlmostEqual(pw["biased|clean"]["po"], 0.5)
        self.assertAlmostEqual(pw["biased|clean"]["kappa"], 0.0)


class TestCalibrateMulti(unittest.TestCase):
    def test_single_labeler_matches_calibrate(self):
        report = _demo_report()
        labels = load_labels(_write_labels(_demo_labels()))
        single = calibrate(report, labels, judge_id="demo-judge")
        multi = calibrate_multi(report, labels, judge_id="demo-judge")
        self.assertEqual(multi["adjudication_rule"], ADJUDICATION_RULE)
        self.assertEqual(multi["label_schema_version"], "labels-v2")
        # per-probe metrics must be identical: adjudication is passthrough
        self.assertEqual(multi["per_probe"], single["per_probe"])
        self.assertEqual(multi["n_unmatched"], single["n_unmatched"])

    def test_iron_rule_refusal(self):
        report = _demo_report()
        labels = load_labels(
            _write_labels(_demo_labels(labeler_id="demo-rater")))
        with self.assertRaises(CalibrationRefusal):
            calibrate_multi(report, labels, judge_id="demo-rater")
        with self.assertRaises(CalibrationRefusal):
            calibrate_multi(report, labels, judge_id="DEMO-RATER")

    def test_duplicate_label_refusal(self):
        report = _demo_report()
        labels = load_labels(_write_labels(_demo_labels()))
        dup = labels + [dict(labels[0])]
        with self.assertRaises(CalibrationRefusal):
            calibrate_multi(report, dup, judge_id="demo-judge")

    def test_unresolved_items_excluded_from_tpr_tnr(self):
        # Two labelers split A/B with equal confidence -> gold TIE
        # (unresolved) -> contributes to kappa only, not TPR/TNR.
        report = {"probes": [
            {"probe": "halo",
             "verdicts": [{"item_id": "item-001", "verdict": "A"}]}]}
        rows = [_rec("item-001", "A", "ann", confidence=3, probe="halo"),
                _rec("item-001", "B", "bob", confidence=3, probe="halo")]
        p = _write_labels(rows)
        try:
            labels = load_labels(p)
            res = calibrate_multi(report, labels, judge_id="judge")
            self.assertEqual(res["n_unresolved"], 1)
            m = res["per_probe"]["halo"]
            self.assertEqual(m["tp"] + m["fn"], 0)
            self.assertEqual(m["tn"] + m["fp"], 0)
            self.assertEqual(m["tpr"], 0.0)  # frozen zero-denom convention
            self.assertEqual(m["n_scored"], 1)  # still in kappa
        finally:
            os.unlink(p)


def _write_report(report):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return path


class TestProbeResolutionFailClosed(unittest.TestCase):
    """Probe-less labels must resolve via the registry or refuse loudly.

    Regression: blinded ids like "item-015" used to infer the garbage probe
    "item" and silently split the (probe, item_id) join instead of joining
    the rater's labels to the pinned ones.
    """

    def test_infers_from_probe_embedded_id(self):
        idx = index_labels([_rec("verbosity-000", "A", "ann", confidence=2)])
        self.assertEqual(set(idx), {("verbosity", "verbosity-000")})

    def test_refuses_blinded_id_without_probe(self):
        with self.assertRaises(CalibrationRefusal):
            index_labels([_rec("item-015", "A", "ann", confidence=2)])

    def test_refuses_mixed_probe_presence(self):
        rows = [_rec("item-001", "A", "ann", confidence=2, probe="halo"),
                _rec("item-002", "B", "ann", confidence=2)]
        with self.assertRaises(CalibrationRefusal):
            index_labels(rows)

    def test_explicit_probe_still_wins(self):
        idx = index_labels([_rec("item-015", "A", "ann", confidence=2, probe="halo")])
        self.assertEqual(set(idx), {("halo", "item-015")})

    def test_calibrate_refuses_probeless_blinded_id(self):
        report = {"probes": [{"probe": "halo",
                              "verdicts": [{"item_id": "item-015",
                                            "verdict": "A"}]}]}
        with self.assertRaises(CalibrationRefusal):
            calibrate(report, [_rec("item-015", "A", "ann", confidence=2)], "some-judge")

    def test_calibrate_joins_inferred_probe(self):
        report = {"probes": [{"probe": "halo",
                              "verdicts": [{"item_id": "halo-000",
                                            "verdict": "A"}]}]}
        cal = calibrate(report, [_rec("halo-000", "A", "ann", confidence=2)], "some-judge")
        self.assertEqual(cal["n_unmatched"], 0)
        self.assertIn("halo", cal["per_probe"])

    def test_calibrate_multi_files_concats_label_files(self):
        from judge_auditor.calibration.adjudicate import calibrate_multi_files
        report = {"probes": [{"probe": "halo",
                              "verdicts": [{"item_id": "item-001",
                                            "verdict": "A"},
                                           {"item_id": "item-002",
                                            "verdict": "B"}]}]}
        rp = _write_report(report)
        p1 = _write_labels([_rec("item-001", "A", "ann", confidence=2, probe="halo"),
                            _rec("item-002", "B", "ann", probe="halo")])
        p2 = _write_labels([_rec("item-001", "A", "bob", probe="halo"),
                            _rec("item-002", "B", "bob", probe="halo")])
        try:
            cal = calibrate_multi_files(rp, [p1, p2], "some-judge")
        finally:
            for p in (rp, p1, p2):
                os.unlink(p)
        self.assertEqual(cal["labeler_ids"], ["ann", "bob"])
        self.assertEqual(cal["n_labels"], 4)
        self.assertEqual(cal["n_items"], 2)


if __name__ == "__main__":
    unittest.main()
