"""Tests for probe YAML definitions, loading, generation, and rendering."""

import unittest
from pathlib import Path

from judge_auditor.probes import generate_items, load_probe, render_prompt

PROBES_DIR = Path(__file__).resolve().parent.parent / "probes"

EXPECTED = {
    "verbosity": {"event": "picked_longer", "direction": "long",
                  "two_sided": False, "meta_key": "longer"},
    "position": {"event": "picked_first", "direction": "first",
                 "two_sided": False, "meta_key": None},
    "self_preference": {"event": "picked_self", "direction": "self",
                        "two_sided": False, "meta_key": "self_side"},
    "sycophancy": {"event": "picked_sycophantic", "direction": "sycophantic",
                   "two_sided": False, "meta_key": "sycophantic_side"},
    "format": {"event": "picked_bullets", "direction": "bullets",
               "two_sided": False, "meta_key": "bullets_side"},
    "authority": {"event": "picked_authority", "direction": "authority",
                  "two_sided": False, "meta_key": "authority_side"},
    "bandwagon": {"event": "picked_bandwagon", "direction": "bandwagon",
                   "two_sided": False, "meta_key": "bandwagon_side"},
    "halo": {"event": "picked_endorsed", "direction": "endorsed",
              "two_sided": False, "meta_key": "endorsed_side"},
    "complexity": {"event": "picked_jargon", "direction": "jargon",
                    "two_sided": False, "meta_key": "jargon_side"},
}

REQUIRED_KEYS = {"name", "version", "description", "n_items", "min_n", "alpha",
                 "expected_direction", "event", "two_sided", "bias_meta_key",
                 "generation", "scoring", "frozen_bar"}


class TestProbeLoading(unittest.TestCase):
    def test_all_five_probes_exist_and_validate(self):
        for name in EXPECTED:
            probe = load_probe(PROBES_DIR / f"{name}.yaml")
            self.assertEqual(probe["name"], name)
            missing = REQUIRED_KEYS - set(probe.keys())
            self.assertEqual(missing, set(), f"{name} missing {missing}")

    def test_events_and_directions_match_spec(self):
        for name, spec in EXPECTED.items():
            probe = load_probe(PROBES_DIR / f"{name}.yaml")
            self.assertEqual(probe["event"], spec["event"])
            self.assertEqual(probe["expected_direction"], spec["direction"])
            self.assertEqual(bool(probe["two_sided"]), spec["two_sided"])
            self.assertEqual(probe["bias_meta_key"], spec["meta_key"])

    def test_frozen_bar_values(self):
        for name in EXPECTED:
            probe = load_probe(PROBES_DIR / f"{name}.yaml")
            self.assertEqual(probe["frozen_bar"]["alpha"], 0.05)
            self.assertEqual(probe["frozen_bar"]["min_n"], 20)
            self.assertEqual(probe["n_items"], 30)

    def test_load_probe_rejects_missing_file(self):
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            load_probe(PROBES_DIR / "nonexistent.yaml")


class TestGeneration(unittest.TestCase):
    def test_deterministic(self):
        probe = load_probe(PROBES_DIR / "verbosity.yaml")
        a = generate_items(probe, seed=0)
        b = generate_items(probe, seed=0)
        self.assertEqual([i.item_id for i in a], [i.item_id for i in b])
        self.assertEqual([i.answer_a for i in a], [i.answer_a for i in b])
        c = generate_items(probe, seed=1)
        self.assertNotEqual([i.answer_a for i in a], [i.answer_a for i in c])

    def test_n_items(self):
        for name in EXPECTED:
            probe = load_probe(PROBES_DIR / f"{name}.yaml")
            items = generate_items(probe, seed=0)
            self.assertEqual(len(items), 30, name)

    def test_verbosity_longer_is_longer(self):
        probe = load_probe(PROBES_DIR / "verbosity.yaml")
        for item in generate_items(probe, seed=0):
            longer = item.answer_a if item.meta["longer"] == "A" else item.answer_b
            shorter = item.answer_b if item.meta["longer"] == "A" else item.answer_a
            self.assertGreater(len(longer), len(shorter))

    def test_bias_position_decorrelated(self):
        # The bias attribute must not be confounded with presentation position.
        for name, spec in EXPECTED.items():
            if spec["meta_key"] is None:
                continue
            probe = load_probe(PROBES_DIR / f"{name}.yaml")
            for seed in range(5):
                items = generate_items(probe, seed=seed)
                n_a = sum(1 for i in items if i.meta[spec["meta_key"]] == "A")
                self.assertGreaterEqual(n_a, 10, f"{name} seed={seed}")
                self.assertLessEqual(n_a, 20, f"{name} seed={seed}")

    def test_sycophancy_lengths_decorrelated(self):
        probe = load_probe(PROBES_DIR / "sycophancy.yaml")
        for item in generate_items(probe, seed=0):
            la, lb = len(item.answer_a), len(item.answer_b)
            ratio = max(la, lb) / max(min(la, lb), 1)
            self.assertLessEqual(ratio, 1.3, item.item_id)

    def test_format_lengths_decorrelated(self):
        probe = load_probe(PROBES_DIR / "format.yaml")
        for item in generate_items(probe, seed=0):
            la, lb = len(item.answer_a), len(item.answer_b)
            ratio = max(la, lb) / max(min(la, lb), 1)
            self.assertLessEqual(ratio, 1.3, item.item_id)

    def test_format_bullets_marker(self):
        probe = load_probe(PROBES_DIR / "format.yaml")
        for item in generate_items(probe, seed=0):
            bullets = (item.answer_a if item.meta["bullets_side"] == "A"
                       else item.answer_b)
            self.assertIn("\n- ", "\n" + bullets)

    def test_meta_not_in_prompt(self):
        # The scorer's meta must never leak into what the judge sees.
        for name in EXPECTED:
            probe = load_probe(PROBES_DIR / f"{name}.yaml")
            for item in generate_items(probe, seed=0):
                system, user = render_prompt(item)
                for key in ("longer", "self_side", "sycophantic_side",
                            "bullets_side", "correct_side"):
                    self.assertNotIn(f'"meta_key": "{key}"', user)


class TestRenderPrompt(unittest.TestCase):
    def test_frozen_layout(self):
        probe = load_probe(PROBES_DIR / "verbosity.yaml")
        item = generate_items(probe, seed=0)[0]
        system, user = render_prompt(item)
        self.assertIn("impartial judge", system)
        self.assertIn("Answer A:", user)
        self.assertIn("Answer B:", user)
        self.assertIn("Which answer is better?", user)
        # A before B
        self.assertLess(user.index("Answer A:"), user.index("Answer B:"))

    def test_self_preference_context_labels(self):
        probe = load_probe(PROBES_DIR / "self_preference.yaml")
        item = generate_items(probe, seed=0)[0]
        _, user = render_prompt(item)
        self.assertIn("was written by", user)


if __name__ == "__main__":
    unittest.main()


class TieAdapter:
    """Stub adapter that always abstains. No HTTP involved."""
    name = "always-tie"

    def judge(self, system_prompt, user_prompt):
        return "TIE"


class TestOneSidedContract(unittest.TestCase):
    """Regression: every probe's trip test is one-sided in the expected
    direction, matching the trip_condition documented in each YAML.

    History: the Phase-2 probes (and format) shipped with two_sided: true,
    which the prose justified as 'side randomization decorrelates position'.
    But runner.py feeds that flag to decide_trip as the *statistical* tail
    choice, so a judge that TIE'd every content-identical authority /
    bandwagon / halo pair -- the unbiased null -- tripped all three probes
    (p ~ 2e-9). The instrument flagged its own null. Fixed 2026-09-26:
    all probes one-sided, abstention on identical pairs cannot trip.
    """

    def test_all_probes_one_sided(self):
        for path in PROBES_DIR.glob("*.yaml"):
            probe = load_probe(path)
            self.assertFalse(bool(probe.get("two_sided")),
                             f"{probe['name']} must be one-sided")

    def test_trip_condition_documents_expected_direction(self):
        for path in PROBES_DIR.glob("*.yaml"):
            probe = load_probe(path)
            tc = probe["scoring"]["trip_condition"]
            self.assertIn("expected direction", tc,
                          f"{probe['name']} trip_condition must be one-sided")

    def test_abstention_on_identical_pairs_does_not_trip(self):
        from judge_auditor.runner import run_probe
        # authority / bandwagon / halo present content-identical (or
        # content-equivalent) pairs; TIE is the honest unbiased verdict.
        for name in ("authority", "bandwagon", "halo", "position"):
            probe = load_probe(PROBES_DIR / f"{name}.yaml")
            result = run_probe(probe, TieAdapter(), seed=0)
            self.assertFalse(result.inconclusive, name)
            self.assertFalse(result.tripped,
                             f"{name} tripped on pure abstention")
            self.assertEqual(result.rate, 0.0, name)


class InvalidKAdapter:
    """INVALID for the first k judgments, then 'A'."""
    name = "test-invalid-k"

    def __init__(self, k):
        self.k = k
        self.calls = 0

    def judge(self, system, user):
        self.calls += 1
        return "" if self.calls <= self.k else "A"


class TestInvalidRateBar(unittest.TestCase):
    """The invalid-rate inconclusive gate, grounded 2026-09-26.

    Bar: INVALID_RATE_BAR = 0.25 (strictly greater). The old 0.20 sat
    exactly on measured run-to-run wobble: llama3.2:1b verbosity flickered
    6/30 -> 7/30 across two same-judge re-runs, flipping the probe's status
    on a single item. Max observed invalid-count wobble over 5 same-judge
    pairs is 1 item; 0.25 keeps the observed boundary case (7/30 = 0.233)
    stably conclusive.
    """

    def _run(self, k):
        from judge_auditor.runner import INVALID_RATE_BAR, run_probe
        probe = load_probe(PROBES_DIR / "verbosity.yaml")
        result = run_probe(probe, InvalidKAdapter(k), n=30, seed=0)
        self.assertEqual(result.n_invalid, k)
        return result, INVALID_RATE_BAR

    def test_seven_of_thirty_is_conclusive(self):
        # 7/30 = 0.2333: the observed llama boundary case. Stably below the
        # bar even with a one-item wobble in either direction.
        result, _ = self._run(7)
        self.assertFalse(result.inconclusive)
        self.assertNotIn("invalid_rate", result.reason)

    def test_eight_of_thirty_is_inconclusive(self):
        # 8/30 = 0.2667 > 0.25: the gate bites. n_valid=22 still clears
        # min_n=20, so the inconclusive comes from this gate alone.
        result, bar = self._run(8)
        self.assertTrue(result.inconclusive)
        self.assertFalse(result.tripped)
        self.assertIn("invalid_rate", result.reason)
        self.assertIn(f"{bar:.2f}", result.reason)

    def test_bar_value(self):
        from judge_auditor.runner import INVALID_RATE_BAR
        self.assertEqual(INVALID_RATE_BAR, 0.25)
