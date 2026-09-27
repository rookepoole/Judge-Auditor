"""The Phase 1 falsifier: every probe must trip against a mock judge with
the planted bias, and must NOT trip against neutral or wrong-bias mocks.

If a probe cannot reproduce its known bias direction here, the instrument
is wrong and the probe does not ship.
"""

import unittest
from pathlib import Path

from judge_auditor.adapters import CustomHTTPAdapter
from judge_auditor.probes import generate_items, load_probe
from judge_auditor.runner import parse_verdict, run_probe
from tests.mock_judge import MockJudgeServer

PROBES_DIR = Path(__file__).resolve().parent.parent / "probes"

# (probe, mock behavior that MUST trip it)
MUST_TRIP = [
    ("verbosity", "always_long"),
    ("position", "always_first"),
    ("self_preference", "always_self"),
    ("sycophancy", "sycophantic"),
    ("format", "prefer_bullets"),
    ("authority", "prefer_authority"),
    ("bandwagon", "follow_bandwagon"),
    ("halo", "prefer_endorsed"),
    ("complexity", "prefer_jargon"),
]

# (probe, mock behavior that must NOT trip it)
MUST_NOT_TRIP = [
    # Neutral control: position-varying, content-blind.
    ("verbosity", "neutral_alternating"),
    ("position", "neutral_alternating"),
    ("self_preference", "neutral_alternating"),
    ("sycophancy", "neutral_alternating"),
    ("format", "neutral_alternating"),
    ("authority", "neutral_alternating"),
    ("bandwagon", "neutral_alternating"),
    ("halo", "neutral_alternating"),
    ("complexity", "neutral_alternating"),
    # Discriminability: a bias in the WRONG direction must not trip.
    ("verbosity", "always_first"),
    ("verbosity", "prefer_bullets"),
    ("position", "always_long"),
    ("self_preference", "always_first"),
    ("sycophancy", "always_first"),
    ("format", "always_first"),
    ("authority", "always_first"),
    ("bandwagon", "always_first"),
    ("halo", "always_first"),
    ("complexity", "always_first"),
]

# Seed chosen empirically: bias-attribute balance is 15/15 (verbosity),
# so wrong-direction controls sit at exactly p=1.0.
SEED = 0


class TripMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = MockJudgeServer()
        cls.server.start()
        cls.addClassCleanup(cls.server.stop)
        cls.probes = {p.stem: load_probe(p)
                      for p in PROBES_DIR.glob("*.yaml")}

    def _run(self, probe_name, behavior):
        adapter = CustomHTTPAdapter(
            url=self.server.base_url + "/v1/chat/completions",
            headers={"X-Mock-Behavior": behavior})
        return run_probe(self.probes[probe_name], adapter, seed=SEED)

    def test_planted_bias_trips(self):
        for probe_name, behavior in MUST_TRIP:
            with self.subTest(probe=probe_name, behavior=behavior):
                result = self._run(probe_name, behavior)
                self.assertFalse(result.inconclusive,
                                 f"{probe_name}: {result.reason}")
                self.assertTrue(
                    result.tripped,
                    f"{probe_name} did NOT trip against {behavior}: "
                    f"rate={result.rate:.3f} p={result.p_value:.4g}")

    def test_neutral_and_wrong_bias_do_not_trip(self):
        for probe_name, behavior in MUST_NOT_TRIP:
            with self.subTest(probe=probe_name, behavior=behavior):
                result = self._run(probe_name, behavior)
                self.assertFalse(
                    result.tripped,
                    f"{probe_name} FALSE-TRIPPED against {behavior}: "
                    f"rate={result.rate:.3f} p={result.p_value:.4g}")


class NewProbeInvariantTest(unittest.TestCase):
    """Standalone invariants for the four new probes.

    These mirror the generator's own asserts so that a bank edit that
    breaks decorrelation is caught at the test level too.
    """

    @classmethod
    def setUpClass(cls):
        cls.probes = {p.stem: load_probe(p)
                      for p in PROBES_DIR.glob("*.yaml")}

    def test_complexity_length_balance(self):
        # 10-20 of the 30 items have the jargon answer strictly longer,
        # so always_long stays a clean wrong-direction control.
        items = generate_items(self.probes["complexity"], seed=SEED)
        self.assertEqual(len(items), 30)
        jargon_longer = 0
        for item in items:
            jargon = (item.answer_a if item.meta["jargon_side"] == "A"
                      else item.answer_b)
            plain = (item.answer_b if item.meta["jargon_side"] == "A"
                     else item.answer_a)
            if len(jargon) > len(plain):
                jargon_longer += 1
        self.assertGreaterEqual(jargon_longer, 10,
                               f"only {jargon_longer} jargon-longer items")
        self.assertLessEqual(jargon_longer, 20,
                             f"{jargon_longer} jargon-longer items")

    def test_authority_no_leak(self):
        # The non-authority side must never name the credential.
        probe = self.probes["authority"]
        for seed in range(5):
            for item in generate_items(probe, seed=seed):
                nonauth = (item.answer_b
                           if item.meta["authority_side"] == "A"
                           else item.answer_a)
                low = nonauth.lower()
                self.assertNotIn("vasquez", low,
                                 f"{item.item_id} seed={seed}")
                self.assertNotIn("mit", low, f"{item.item_id} seed={seed}")


class ParseVerdictTest(unittest.TestCase):
    def test_plain_letters(self):
        self.assertEqual(parse_verdict("A"), "A")
        self.assertEqual(parse_verdict("B"), "B")

    def test_prose(self):
        self.assertEqual(parse_verdict("Answer B is better because..."), "B")
        self.assertEqual(parse_verdict("I choose A."), "A")

    def test_tie(self):
        self.assertEqual(parse_verdict("It's a tie."), "TIE")

    def test_invalid(self):
        self.assertEqual(parse_verdict(""), "INVALID")
        self.assertEqual(parse_verdict("The answers are both fine"), "INVALID")


if __name__ == "__main__":
    unittest.main()
