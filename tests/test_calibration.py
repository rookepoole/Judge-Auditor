"""Tests for the calibration harness (all offline, no API keys).

Covers: frozen tie rule metrics (TPR/TNR/kappa/CI), labeler!=judge iron
rule, protocol freeze/verify round-trip + tamper detection, rescore
determinism + self-consistency, replay determinism, and label schema
validation errors.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from judge_auditor import calibration as cal
from judge_auditor.calibration import (CalibrationRefusal, ProtocolMismatch,
                                       calibrate, calibrate_probe,
                                       clopper_pearson_ci, compare_rescores,
                                       freeze, load_labels, make_rescore_set,
                                       verify)
from judge_auditor.probes import load_probe
from judge_auditor.report import build_report
from judge_auditor.runner import run_probe

REPO = Path(__file__).resolve().parent.parent
PROBES_DIR = REPO / "probes"

# The 5 Phase-1 probes are frozen and stable. The 4 Phase-2 probes
# (authority, bandwagon, complexity, halo) are built concurrently by another
# builder and their YAML banks may be mid-edit; the calibration machinery
# is probe-agnostic (it works off report.json probe entries), so the test
# fixture pins the stable set for hermeticity.
STABLE_PROBES = ("verbosity", "position", "self_preference", "sycophancy",
                 "format")


class AlwaysA:
    name = "test-always-a"

    def judge(self, system, user):
        return "A"


def _all_probes():
    return [load_probe(PROBES_DIR / f"{name}.yaml")
            for name in STABLE_PROBES]


def _fake_report(tmp):
    """Build a real-shaped report.json offline (always-A judge, n=30)."""
    probes = _all_probes()
    results = [run_probe(p, AlwaysA(), n=30, seed=0) for p in probes]
    report = build_report(results, "test-always-a")
    path = Path(tmp) / "report.json"
    path.write_text(json.dumps(report, indent=2))
    return path, probes


class TestMetrics(unittest.TestCase):
    def test_perfect_judge(self):
        pairs = ([("A", "A")] * 40 + [("B", "B")] * 40
                 + [("TIE", "TIE")] * 10)
        m = calibrate_probe(pairs)
        self.assertEqual(m["tpr"], 1.0)
        self.assertEqual(m["tnr"], 1.0)
        self.assertEqual(m["kappa"], 1.0)
        self.assertEqual(m["po"], 1.0)
        self.assertEqual(m["tp"], 40)
        self.assertEqual(m["tn"], 40)

    def test_coin_flip_judge(self):
        rng = random.Random(12345)
        labels = ["A"] * 90 + ["B"] * 90 + ["TIE"] * 20
        rng.shuffle(labels)
        jrng = random.Random(999)
        pairs = [(lab, "A" if jrng.random() < 0.5 else "B")
                 for lab in labels]
        m = calibrate_probe(pairs)
        self.assertLess(abs(m["kappa"]), 0.25)
        self.assertGreaterEqual(m["tpr"], 0.35)
        self.assertLessEqual(m["tpr"], 0.65)
        self.assertGreaterEqual(m["tnr"], 0.35)
        self.assertLessEqual(m["tnr"], 0.65)
        # CIs bracket the point estimate and stay in [0, 1].
        self.assertLessEqual(m["tpr_ci"][0], m["tpr"])
        self.assertLessEqual(m["tpr"], m["tpr_ci"][1])
        self.assertGreaterEqual(m["tpr_ci"][0], 0.0)
        self.assertLessEqual(m["tpr_ci"][1], 1.0)

    def test_always_opposite_judge(self):
        pairs = ([("A", "B")] * 40 + [("B", "A")] * 40
                 + [("TIE", "A")] * 10)
        m = calibrate_probe(pairs)
        self.assertEqual(m["tpr"], 0.0)
        self.assertEqual(m["tnr"], 0.0)
        self.assertFalse(m["kappa"] != m["kappa"])  # not NaN
        self.assertLess(m["kappa"], 0.0)

    def test_always_tie_judge(self):
        pairs = [("A", "TIE")] * 40 + [("B", "TIE")] * 40
        m = calibrate_probe(pairs)
        self.assertEqual(m["tpr"], 0.0)
        self.assertEqual(m["tnr"], 0.0)
        self.assertFalse(m["kappa"] != m["kappa"])  # not NaN

    def test_all_human_tie_degenerate(self):
        # Zero decisive labels: rates are 0.0 (never NaN), CI is (0, 1).
        # Fully degenerate case (all judge TIE, all human TIE): Po=Pe=1
        # -> kappa 1.0 by the frozen convention.
        pairs = [("TIE", "TIE")] * 25
        m = calibrate_probe(pairs)
        self.assertEqual(m["tpr"], 0.0)
        self.assertEqual(m["tnr"], 0.0)
        self.assertEqual(m["tpr_ci"], [0.0, 1.0])
        self.assertEqual(m["tnr_ci"], [0.0, 1.0])
        self.assertFalse(m["kappa"] != m["kappa"])
        self.assertEqual(m["po"], 1.0)
        self.assertEqual(m["pe"], 1.0)
        self.assertEqual(m["kappa"], 1.0)

    def test_all_human_tie_mixed_judge(self):
        # Human TIEs with a decisive judge: kappa is defined (0.0 here),
        # never NaN.
        pairs = [("TIE", "TIE")] * 20 + [("TIE", "A")] * 5
        m = calibrate_probe(pairs)
        self.assertEqual(m["tpr"], 0.0)
        self.assertEqual(m["tnr"], 0.0)
        self.assertFalse(m["kappa"] != m["kappa"])
        self.assertEqual(m["kappa"], 0.0)

    def test_judge_invalid_excluded_pairwise(self):
        pairs = [("A", "A")] * 10 + [("A", "INVALID")] * 5
        m = calibrate_probe(pairs)
        self.assertEqual(m["n_labeled"], 15)
        self.assertEqual(m["n_scored"], 10)
        self.assertEqual(m["tpr"], 1.0)

    def test_clopper_pearson_known_values(self):
        # 95% CI for 0/10 must start at 0 and end near 0.3085.
        lo, hi = clopper_pearson_ci(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.3085, places=3)
        # Symmetry: CI(3/10) mirrors CI(7/10).
        lo3, hi3 = clopper_pearson_ci(3, 10)
        lo7, hi7 = clopper_pearson_ci(7, 10)
        self.assertAlmostEqual(lo3, 1.0 - hi7, places=9)
        self.assertAlmostEqual(hi3, 1.0 - lo7, places=9)


class TestLabels(unittest.TestCase):
    def _write(self, tmp, lines):
        p = Path(tmp) / "labels.jsonl"
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_valid_labels_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [
                '{"item_id": "verbosity-000", "label": "A", '
                '"labeler_id": "ann", "timestamp": "2026-09-26T10:00:00+00:00",'
                ' "probe": "verbosity"}',
                "",
                '{"item_id": "position-001", "label": "TIE", '
                '"labeler_id": "ann", "timestamp": "2026-09-26T10:01:00Z"}',
            ])
            recs = load_labels(p)
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[0]["probe"], "verbosity")

    def test_malformed_line_names_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [
                '{"item_id": "verbosity-000", "label": "A", '
                '"labeler_id": "ann", "timestamp": "2026-09-26T10:00:00+00:00"}',
                '{"item_id": "verbosity-001", "label": "A", '
                '"labeler_id": "ann", "timestamp": "2026-09-26T10:00:00+00:00"}',
                '{"item_id": "verbosity-002", "label": "C", '
                '"labeler_id": "ann", "timestamp": "2026-09-26T10:00:00+00:00"}',
            ])
            with self.assertRaises(ValueError) as ctx:
                load_labels(p)
            self.assertIn("line 3", str(ctx.exception))
            self.assertIn("label", str(ctx.exception))

    def test_bad_timestamp_names_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [
                '{"item_id": "x-0", "label": "A", "labeler_id": "ann", '
                '"timestamp": "not-a-time"}',
            ])
            with self.assertRaises(ValueError) as ctx:
                load_labels(p)
            self.assertIn("timestamp", str(ctx.exception))

    def test_not_json_names_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, ["{oops"])
            with self.assertRaises(ValueError) as ctx:
                load_labels(p)
            self.assertIn("line 1", str(ctx.exception))


class TestIronRule(unittest.TestCase):
    def _report_labels(self):
        report = {"probes": [{
            "probe": "verbosity",
            "verdicts": [{"item_id": "verbosity-000", "verdict": "A"},
                         {"item_id": "verbosity-001", "verdict": "B"}],
        }]}
        labels = [
            {"item_id": "verbosity-000", "label": "A",
             "labeler_id": "ann", "timestamp": "2026-09-26T10:00:00+00:00"},
            {"item_id": "verbosity-001", "label": "B",
             "labeler_id": "ann", "timestamp": "2026-09-26T10:00:00+00:00"},
        ]
        return report, labels

    def test_labeler_equals_judge_refused(self):
        report, labels = self._report_labels()
        with self.assertRaises(ValueError):
            calibrate(report, labels, "ann")

    def test_labeler_equals_judge_case_insensitive(self):
        report, labels = self._report_labels()
        with self.assertRaises(CalibrationRefusal):
            calibrate(report, labels, "ANN")

    def test_different_labeler_passes(self):
        report, labels = self._report_labels()
        out = calibrate(report, labels, "judge-model-x")
        self.assertEqual(out["per_probe"]["verbosity"]["tpr"], 1.0)
        self.assertEqual(out["per_probe"]["verbosity"]["tnr"], 1.0)
        self.assertEqual(out["n_unmatched"], 0)

    def test_duplicate_label_refused(self):
        report, labels = self._report_labels()
        labels = labels + [dict(labels[0])]
        with self.assertRaises(CalibrationRefusal):
            calibrate(report, labels, "judge-model-x")

    def test_unmatched_labels_counted(self):
        report, labels = self._report_labels()
        labels = labels + [
            {"item_id": "nope-999", "label": "A", "probe": "nope",
             "labeler_id": "ann", "timestamp": "2026-09-26T10:00:00+00:00"}]
        out = calibrate(report, labels, "judge-model-x")
        self.assertEqual(out["n_unmatched"], 1)
        self.assertEqual(out["n_labels"], 3)

    def test_unmatched_labels_counted_probe_embedded_id(self):
        # A probe-less label whose item_id embeds a REAL probe name still
        # joins the probe namespace: if the item isn't in the report it is
        # counted as unmatched, not refused.
        report, labels = self._report_labels()
        labels = labels + [
            {"item_id": "verbosity-999", "label": "A",
             "labeler_id": "ann", "timestamp": "2026-09-26T10:00:00+00:00"}]
        out = calibrate(report, labels, "judge-model-x")
        self.assertEqual(out["n_unmatched"], 1)
        self.assertEqual(out["n_labels"], 3)


class TestProtocol(unittest.TestCase):
    def test_freeze_verify_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, probes = _fake_report(tmp)
            ppath = str(Path(tmp) / "protocol.json")
            protocol = freeze(probes, 30, 0, ppath)
            self.assertTrue(Path(ppath + ".sha256").exists())
            verified = verify(ppath, str(report_path))
            self.assertEqual(verified["protocol_version"], "protocol-v2")
            self.assertEqual(len(verified["probes"]), 5)
            self.assertEqual(verified["n"], 30)
            self.assertEqual(verified["seed"], 0)

    def test_hash_covers_content_only_not_created_at(self):
        # Regression (2026-09-26): v1 hashed created_at, so two freezes of
        # the byte-identical battery minted different identities.
        from judge_auditor.calibration import protocols as P
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmp:
            _, probes = _fake_report(tmp)
            p1 = P.build_protocol(probes, 30, 0)
            p2 = P.build_protocol(probes, 30, 0)
            p1["created_at"] = "2026-01-01T00:00:00+00:00"
            p2["created_at"] = "2026-09-26T12:34:56+00:00"
            self.assertNotEqual(p1["created_at"], p2["created_at"])
            self.assertEqual(P._hash(P._content_form(p1)),
                             P._hash(P._content_form(p2)))

    def test_content_change_changes_hash(self):
        from judge_auditor.calibration import protocols as P
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmp:
            _, probes = _fake_report(tmp)
            p1 = P.build_protocol(probes, 30, 0)
            p2 = P.build_protocol(probes, 30, 0)
            p2["n"] = 31
            self.assertNotEqual(P._hash(P._content_form(p1)),
                                P._hash(P._content_form(p2)))

    def test_legacy_v1_still_verifies(self):
        # A v1-frozen file (full-body hash incl. created_at) verifies under
        # the legacy path; it is never silently re-hashed as v2.
        from judge_auditor.calibration import protocols as P
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmp:
            report_path, probes = _fake_report(tmp)
            p = P.build_protocol(probes, 30, 0)
            p["protocol_version"] = "protocol-v1"
            body = {k: v for k, v in p.items()}
            digest = P._hash(body)
            p["sha256"] = digest
            ppath = Path(tmp) / "proto_v1.json"
            ppath.write_text(json.dumps(p, indent=2, sort_keys=True))
            verified = verify(str(ppath), str(report_path))
            self.assertEqual(verified["protocol_version"], "protocol-v1")
            self.assertEqual(verified["sha256"], digest)

    def test_legacy_v1_tamper_detected(self):
        from judge_auditor.calibration import protocols as P
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmp:
            report_path, probes = _fake_report(tmp)
            p = P.build_protocol(probes, 30, 0)
            p["protocol_version"] = "protocol-v1"
            p["sha256"] = P._hash({k: v for k, v in p.items()
                                   if k != "sha256"})
            ppath = Path(tmp) / "proto_v1.json"
            ppath.write_text(json.dumps(p, indent=2, sort_keys=True))
            text = ppath.read_text().replace('"n": 30', '"n": 31')
            ppath.write_text(text)
            with self.assertRaises(ProtocolMismatch) as ctx:
                verify(str(ppath), str(report_path))
            self.assertEqual(ctx.exception.field, "sha256")

    def test_unknown_protocol_version_rejected(self):
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmp:
            report_path, probes = _fake_report(tmp)
            ppath = str(Path(tmp) / "protocol.json")
            protocol = freeze(probes, 30, 0, ppath)
            protocol["protocol_version"] = "protocol-v99"
            Path(ppath).write_text(json.dumps(protocol, indent=2,
                                              sort_keys=True))
            with self.assertRaises(ProtocolMismatch) as ctx:
                verify(ppath, str(report_path))
            self.assertEqual(ctx.exception.field, "protocol_version")

    def test_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, probes = _fake_report(tmp)
            ppath = Path(tmp) / "protocol.json"
            freeze(probes, 30, 0, ppath)
            text = ppath.read_text().replace('"n": 30', '"n": 31')
            ppath.write_text(text)
            with self.assertRaises(ProtocolMismatch) as ctx:
                verify(str(ppath), str(report_path))
            self.assertEqual(ctx.exception.field, "sha256")

    def test_version_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, probes = _fake_report(tmp)
            ppath = str(Path(tmp) / "protocol.json")
            protocol = freeze(probes, 30, 0, ppath)
            protocol["probes"][0]["version"] = "9.9.9"
            protocol["sha256"] = cal.protocols._hash(
                cal.protocols._content_form(protocol))
            Path(ppath).write_text(json.dumps(protocol, indent=2,
                                              sort_keys=True))
            with self.assertRaises(ProtocolMismatch) as ctx:
                verify(ppath, str(report_path))
            self.assertIn("version", ctx.exception.field)

    def test_labels_verified_when_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, probes = _fake_report(tmp)
            ppath = str(Path(tmp) / "protocol.json")
            freeze(probes, 30, 0, ppath)
            bad = Path(tmp) / "labels.jsonl"
            bad.write_text('{"item_id": "x", "label": "ZZZ"}\n')
            with self.assertRaises(ValueError):
                verify(ppath, str(report_path), labels_path=str(bad))


class TestRescore(unittest.TestCase):
    def test_rescore_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, _ = _fake_report(tmp)
            o1, s1 = Path(tmp) / "r1.jsonl", Path(tmp) / "s1.json"
            o2, s2 = Path(tmp) / "r2.jsonl", Path(tmp) / "s2.json"
            n1 = make_rescore_set(str(report_path), 42, o1, s1)
            n2 = make_rescore_set(str(report_path), 42, o2, s2)
            self.assertEqual(n1, n2)
            self.assertEqual(o1.read_bytes(), o2.read_bytes())
            self.assertEqual(s1.read_bytes(), s2.read_bytes())

    def test_rescore_blinded_and_stripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, _ = _fake_report(tmp)
            out, sealed = Path(tmp) / "r.jsonl", Path(tmp) / "s.json"
            make_rescore_set(str(report_path), 7, out, sealed)
            text = out.read_text()
            # No probe names leak into the blinded set...
            for name in ("verbosity", "position", "self_preference",
                         "sycophancy", "format"):
                self.assertNotIn(f'"{name}"', text)
            # ...and the self_preference identity cue is stripped: the
            # "Note: Answer ..." context line and the cue names are gone.
            self.assertNotIn("Note: Answer", text)
            self.assertNotIn("JudgeModel", text)
            self.assertNotIn("RivalModel", text)
            # ...while genuine answer content that happens to contain the
            # phrase is preserved (surgical strip, not a blanket delete).
            self.assertIn("was written by J.K. Rowling", text)
            sealed_d = json.loads(sealed.read_text())
            self.assertTrue(any(v["probe"] == "self_preference"
                                for v in sealed_d.values()))

    def test_self_consistency_replay_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, _ = _fake_report(tmp)
            report = json.loads(Path(report_path).read_text())
            original = {(e["probe"], v["item_id"]): v["verdict"]
                        for e in report["probes"] for v in e["verdicts"]}
            out, sealed = Path(tmp) / "r.jsonl", Path(tmp) / "s.json"
            make_rescore_set(str(report_path), 11, out, sealed)
            sealed_d = json.loads(sealed.read_text())
            # Verdict-preserving replay: map the original verdict FORWARD
            # through the swap, as an identical re-judgment would produce.
            with open(Path(tmp) / "verdicts.jsonl", "w") as f:
                for rid, s in sealed_d.items():
                    v = original[(s["probe"], s["item_id"])]
                    if s["swapped"] and v in ("A", "B"):
                        v = "B" if v == "A" else "A"
                    f.write(json.dumps({"rescore_id": rid,
                                        "verdict": v}) + "\n")
            res = compare_rescores(str(report_path),
                                   str(Path(tmp) / "verdicts.jsonl"),
                                   str(sealed))
            self.assertEqual(res["overall"]["consistency"], 1.0)
            for name, slot in res["per_probe"].items():
                self.assertEqual(slot["consistency"], 1.0, name)
            # Exact Clopper-Pearson CI for a perfect rate: upper is 1.0,
            # lower is (alpha/2)^(1/n) < 1.0 — it must NOT collapse to a
            # point.
            n = res["overall"]["n"]
            lo, hi = res["overall"]["ci"]
            self.assertEqual(hi, 1.0)
            self.assertAlmostEqual(lo, clopper_pearson_ci(n, n)[0])
            self.assertLess(lo, 1.0)

    def test_replay_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, _ = _fake_report(tmp)
            report = json.loads(Path(report_path).read_text())
            labels = []
            for e in report["probes"]:
                for v in e["verdicts"][:10]:
                    labels.append({
                        "item_id": v["item_id"], "label": v["verdict"],
                        "labeler_id": "ann",
                        "timestamp": "2026-09-26T10:00:00+00:00",
                        "probe": e["probe"]})
            a = calibrate(report, labels, "judge-x")
            b = calibrate(report, labels, "judge-x")
            self.assertEqual(json.dumps(a, sort_keys=True),
                             json.dumps(b, sort_keys=True))


class TestCalibrateCLI(unittest.TestCase):
    def test_cli_calibrate_refusal_exits_nonzero(self):
        from judge_auditor.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            report_path, _ = _fake_report(tmp)
            labels = Path(tmp) / "labels.jsonl"
            labels.write_text(
                '{"item_id": "verbosity-000", "label": "A", '
                '"labeler_id": "SameJudge", '
                '"timestamp": "2026-09-26T10:00:00+00:00", '
                '"probe": "verbosity"}\n')
            with self.assertRaises(SystemExit) as ctx:
                main(["calibrate", "--labels", str(labels),
                      "--run", str(report_path),
                      "--judge-id", "samejudge",
                      "--out-dir", str(Path(tmp) / "out")])
            self.assertNotEqual(ctx.exception.code, 0)

    def test_cli_calibrate_success_writes_report_copy(self):
        from judge_auditor.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            report_path, _ = _fake_report(tmp)
            labels = Path(tmp) / "labels.jsonl"
            labels.write_text(
                '{"item_id": "verbosity-000", "label": "A", '
                '"labeler_id": "ann", '
                '"timestamp": "2026-09-26T10:00:00+00:00", '
                '"probe": "verbosity"}\n')
            out = Path(tmp) / "out"
            rc = main(["calibrate", "--labels", str(labels),
                       "--run", str(report_path),
                       "--judge-id", "judge-x",
                       "--out-dir", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "calibration.json").exists())
            md = (out / "report.md").read_text()
            self.assertIn("## Calibration", md)
            self.assertIn("verbosity", md)

    def test_cli_protocol_round_trip(self):
        from judge_auditor.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            report_path, _ = _fake_report(tmp)
            ppath = str(Path(tmp) / "protocol.json")
            rc = main(["protocol-freeze", "--probes",
                       ",".join(STABLE_PROBES), "--n", "30",
                       "--seed", "0", "--out", ppath])
            self.assertEqual(rc, 0)
            rc = main(["protocol-verify", "--protocol", ppath,
                       "--run", str(report_path)])
            self.assertEqual(rc, 0)

    def test_cli_labeling_round_trip(self):
        from judge_auditor.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            report_path, _ = _fake_report(tmp)
            lab = str(Path(tmp) / "labeling.jsonl")
            rc = main(["export-labeling-set", "--run", str(report_path),
                       "--probes", "verbosity,position", "--seed", "5",
                       "--out", lab])
            self.assertEqual(rc, 0)
            sealed = Path(tmp) / "labeling.sealed.json"
            self.assertTrue(sealed.exists())
            # Craft replies: label everything "A".
            replies = Path(tmp) / "replies.jsonl"
            with open(lab) as f, open(replies, "w") as g:
                for line in f:
                    d = json.loads(line)
                    g.write(json.dumps({"labeling_id": d["labeling_id"],
                                        "label": "A"}) + "\n")
            out = str(Path(tmp) / "labels.jsonl")
            rc = main(["import-labels", "--in", str(replies),
                       "--map", str(sealed), "--labeler-id", "ann",
                       "--out", out])
            self.assertEqual(rc, 0)
            recs = load_labels(out)
            self.assertEqual(len(recs), 60)  # 2 probes x 30 items
            self.assertTrue(all(r["labeler_id"] == "ann" for r in recs))


if __name__ == "__main__":
    unittest.main()
