"""Tests for report building, grading, and rendering."""

import json
import tempfile
import unittest
from pathlib import Path

from judge_auditor.report import build_report, grade, to_json, to_markdown
from judge_auditor.runner import ProbeResult


def _result(name, tripped=False, inconclusive=False):
    return ProbeResult(
        probe_name=name, version="1.0.0", expected_direction="long",
        n_items=30, n_valid=30, n_invalid=0,
        events=30 if tripped else 15,
        rate=1.0 if tripped else 0.5,
        p_value=1e-9 if tripped else 1.0,
        alpha=0.05, two_sided=False, tripped=tripped,
        inconclusive=inconclusive, invalid_rate=0.0, seed=0,
        reason="test")


class TestGrade(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(grade([_result("a"), _result("b")]), "PASS")

    def test_one_trip_needs_attention(self):
        self.assertEqual(
            grade([_result("a", tripped=True), _result("b")]),
            "NEEDS ATTENTION")

    def test_two_trips_fail(self):
        self.assertEqual(
            grade([_result("a", tripped=True), _result("b", tripped=True)]),
            "FAIL")

    def test_inconclusive(self):
        self.assertEqual(
            grade([_result("a", inconclusive=True), _result("b")]),
            "INCONCLUSIVE")


class TestRendering(unittest.TestCase):
    def test_json_round_trip(self):
        report = build_report([_result("verbosity", tripped=True),
                               _result("position")], "mock")
        with tempfile.TemporaryDirectory() as d:
            path = to_json(report, Path(d) / "report.json")
            loaded = json.loads(Path(path).read_text())
        self.assertEqual(loaded["grade"], "NEEDS ATTENTION")
        self.assertEqual(loaded["n_tripped"], 1)
        self.assertEqual(len(loaded["probes"]), 2)
        self.assertIn("generated_at", loaded)

    def test_markdown_contains_results(self):
        report = build_report([_result("verbosity", tripped=True)], "mock")
        md = to_markdown(report)
        self.assertIn("TRIPPED", md)
        self.assertIn("verbosity", md)
        self.assertIn("NEEDS ATTENTION", md)


if __name__ == "__main__":
    unittest.main()
