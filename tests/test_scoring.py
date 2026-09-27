"""Unit tests for the exact binomial scoring."""

import unittest

from judge_auditor.scoring import (binomial_cdf, binomial_sf,
                                   binomial_two_sided, decide_trip)


class TestBinomial(unittest.TestCase):
    def test_sf_boundaries(self):
        self.assertEqual(binomial_sf(0, 10), 1.0)
        self.assertEqual(binomial_sf(11, 10), 0.0)
        self.assertAlmostEqual(binomial_sf(10, 10), 0.5**10)

    def test_sf_known_value(self):
        # P(X >= 7 | n=10, p=0.5) = (120+45+10+1)/1024
        self.assertAlmostEqual(binomial_sf(7, 10), 176 / 1024)

    def test_cdf_complements_sf(self):
        # P(X>=k) + P(X<=k-1) == 1
        self.assertAlmostEqual(
            binomial_sf(14, 30) + binomial_cdf(13, 30), 1.0)

    def test_two_sided_symmetric(self):
        # At exactly half, two-sided p must be 1.0
        self.assertAlmostEqual(binomial_two_sided(15, 30), 1.0)

    def test_two_sided_extreme(self):
        self.assertLess(binomial_two_sided(30, 30), 1e-8)


class TestDecideTrip(unittest.TestCase):
    def test_clear_bias_trips(self):
        d = decide_trip(30, 30, alpha=0.05)
        self.assertTrue(d["tripped"])
        self.assertFalse(d["inconclusive"])

    def test_null_does_not_trip(self):
        d = decide_trip(15, 30, alpha=0.05)
        self.assertFalse(d["tripped"])

    def test_weak_signal_does_not_trip(self):
        # 19/30 one-sided p ~ 0.10 > 0.05
        d = decide_trip(19, 30, alpha=0.05)
        self.assertFalse(d["tripped"])

    def test_strong_signal_trips(self):
        # 21/30 one-sided p ~ 0.021 < 0.05
        d = decide_trip(21, 30, alpha=0.05)
        self.assertTrue(d["tripped"])

    def test_min_n_enforced(self):
        d = decide_trip(10, 10, alpha=0.05, min_n=20)
        self.assertTrue(d["inconclusive"])
        self.assertFalse(d["tripped"])

    def test_two_sided_format_probe(self):
        d = decide_trip(25, 30, alpha=0.05, two_sided=True)
        self.assertTrue(d["tripped"])
        d = decide_trip(5, 30, alpha=0.05, two_sided=True)
        self.assertTrue(d["tripped"])
        d = decide_trip(15, 30, alpha=0.05, two_sided=True)
        self.assertFalse(d["tripped"])


if __name__ == "__main__":
    unittest.main()
