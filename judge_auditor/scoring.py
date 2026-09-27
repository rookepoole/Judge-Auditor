"""Exact binomial scoring for judge-bias probes.

No scipy dependency — everything is computed with math.comb so the
package stays dependency-light and auditable.
"""

import math


def binomial_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact upper tail."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(
        math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1)
    )


def binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Exact lower tail."""
    if n <= 0:
        return 1.0
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(
        math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(0, k + 1)
    )


def binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact p-value via the doubling method, capped at 1.0."""
    if n <= 0:
        return 1.0
    one_sided = min(binomial_sf(k, n, p), binomial_cdf(k, n, p))
    return min(1.0, 2.0 * one_sided)


def decide_trip(
    events: int,
    n_valid: int,
    alpha: float = 0.05,
    two_sided: bool = False,
    min_n: int = 20,
) -> dict:
    """Decide whether a probe trips.

    Returns a dict with rate, p_value, tripped, and the reason.
    A probe trips only if the exact binomial p-value is below alpha
    *in the expected direction* (one-sided) — or on either side for
    two_sided probes — AND enough valid judgments exist.
    """
    if n_valid < min_n:
        return {
            "rate": (events / n_valid) if n_valid else 0.0,
            "p_value": 1.0,
            "tripped": False,
            "inconclusive": True,
            "reason": f"n_valid={n_valid} < min_n={min_n}",
        }
    if two_sided:
        p_value = binomial_two_sided(events, n_valid)
    else:
        p_value = binomial_sf(events, n_valid)
    tripped = p_value < alpha
    return {
        "rate": events / n_valid,
        "p_value": p_value,
        "tripped": tripped,
        "inconclusive": False,
        "reason": f"p={p_value:.4g} {'<' if tripped else '>='} alpha={alpha}",
    }


def fisher_exact_2x2(a1: int, b1: int, a2: int, b2: int) -> float:
    """Two-sided Fisher's exact p-value for a 2x2 table [[a1,b1],[a2,b2]].

    Convention (the standard "sum of small p-values" two-sided test):
    fix the row and column margins, enumerate every table consistent with
    them, and sum the hypergeometric probabilities of the tables whose
    probability is <= the observed table's probability. Degenerate
    margins (any zero row/column total) return 1.0.

    Exact: table inclusion is decided by cross-multiplied integer
    arithmetic (math.comb), so no floating-point thresholding is involved;
    only the final probability sum is floating point. Returns a value in
    [0, 1].
    """
    r1, r2 = a1 + b1, a2 + b2
    c1, c2 = a1 + a2, b1 + b2
    n = r1 + r2
    if n <= 0 or r1 <= 0 or r2 <= 0 or c1 <= 0 or c2 <= 0:
        return 1.0
    # Numerator of the hypergeometric probability for a table with x in
    # the (0,0) cell: comb(r1, x) * comb(r2, c1 - x). The denominator
    # comb(n, c1) is constant across tables, so it cancels in the
    # inclusion comparison.
    obs_num = math.comb(r1, a1) * math.comb(r2, c1 - a1)
    total = 0.0
    denom = math.comb(n, c1)
    for x in range(max(0, c1 - r2), min(c1, r1) + 1):
        num = math.comb(r1, x) * math.comb(r2, c1 - x)
        if num <= obs_num:
            total += num / denom
    return min(1.0, total)
