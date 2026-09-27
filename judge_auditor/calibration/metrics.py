"""Judge-vs-human calibration metrics (frozen tie rule "tie-v1").

Positive class is fixed: "the human label says A is better".

For one (human_label, judge_verdict) pair, with judge INVALID excluded
pairwise (contributes to nothing, not even a denominator):

    TP = label A & judge A
    FN = label A & (judge B or TIE)
    TN = label B & judge B
    FP = label B & (judge A or TIE)

A judge TIE against a decisive label is a MISS: it lands in the denominator
(FN/FP) and never the numerator. A human TIE label carries no decisive
signal: it is excluded from the TPR/TNR denominators but kept as a third
category in kappa.

Confidence intervals are exact Clopper-Pearson 95% intervals computed by
bisection on scoring.binomial_sf / scoring.binomial_cdf (no scipy).

FROZEN zero-denominator conventions (see SPEC.md): if a rate's denominator
is zero, the rate is reported as 0.0 (never NaN) and the CI as (0.0, 1.0).
Kappa: if Pe == 1 (all mass on one judge row AND one human column),
kappa = 1.0 iff Po == 1 else 0.0.
"""

from ..scoring import binomial_cdf, binomial_sf

CI_ALPHA = 0.05
_CATEGORIES = ("A", "B", "TIE")


def clopper_pearson_ci(k, n, alpha=CI_ALPHA):
    """Exact two-sided Clopper-Pearson interval for a binomial proportion.

    k successes in n trials; returns (lower, upper). Computed by bisection
    on the exact binomial tail functions: lower solves
    P(X >= k; p) = alpha/2, upper solves P(X <= k; p) = alpha/2.
    """
    if not isinstance(k, int) or not isinstance(n, int):
        raise ValueError(f"k and n must be ints, got {k!r}, {n!r}")
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")

    def _bisect(fn, target):
        lo, hi = 0.0, 1.0
        # fn is monotone in p on [0, 1]; bisection assumes fn(lo) and fn(hi)
        # bracket the target, which holds for the two calls below.
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if (fn(lo) - target) * (fn(mid) - target) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    lower = 0.0 if k == 0 else _bisect(
        lambda p: binomial_sf(k, n, p), alpha / 2.0)
    upper = 1.0 if k == n else _bisect(
        lambda p: binomial_cdf(k, n, p), alpha / 2.0)
    return (lower, upper)


def _kappa_from_table(table):
    """Cohen's kappa from a 3x3 {judge: {human: count}} table.

    Returns (kappa, po, pe).
    """
    n = sum(table[j][h] for j in _CATEGORIES for h in _CATEGORIES)
    if n == 0:
        return (0.0, 0.0, 0.0)
    po = sum(table[c][c] for c in _CATEGORIES) / n
    pe = 0.0
    for c in _CATEGORIES:
        row = sum(table[c][h] for h in _CATEGORIES) / n      # judge marginal
        col = sum(table[j][c] for j in _CATEGORIES) / n      # human marginal
        pe += row * col
    if pe == 1.0:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return (kappa, po, pe)


def calibrate_probe(pairs):
    """Calibrate one probe's judge verdicts against human labels.

    pairs: list of (human_label, judge_verdict) with values in
    {"A", "B", "TIE"} (+ "INVALID" allowed for judge_verdict; those pairs
    are excluded pairwise).

    Returns a dict with:
      n_labeled  - pairs supplied (joined labels)
      n_scored   - pairs with a valid (non-INVALID) judge verdict
      tp, fn, tn, fp (frozen tie rule above)
      tpr, tpr_ci, tnr, tnr_ci
      kappa, po, pe  (3-category judge-vs-human; INVALID excluded pairwise)
      confusion - the 3x3 {judge: {human: count}} table
    """
    table = {j: {h: 0 for h in _CATEGORIES} for j in _CATEGORIES}
    tp = fn = tn = fp = 0
    n_scored = 0
    for human, judge in pairs:
        if human not in _CATEGORIES:
            raise ValueError(f"invalid human label: {human!r}")
        if judge == "INVALID":
            continue
        if judge not in _CATEGORIES:
            raise ValueError(f"invalid judge verdict: {judge!r}")
        n_scored += 1
        table[judge][human] += 1
        if human == "A":
            if judge == "A":
                tp += 1
            else:  # judge B or TIE -> miss
                fn += 1
        elif human == "B":
            if judge == "B":
                tn += 1
            else:  # judge A or TIE -> miss
                fp += 1
        # human TIE: no decisive signal; kappa only.

    tpr_denom = tp + fn
    tnr_denom = tn + fp
    tpr = tp / tpr_denom if tpr_denom else 0.0
    tnr = tn / tnr_denom if tnr_denom else 0.0
    tpr_ci = clopper_pearson_ci(tp, tpr_denom) if tpr_denom else (0.0, 1.0)
    tnr_ci = clopper_pearson_ci(tn, tnr_denom) if tnr_denom else (0.0, 1.0)
    kappa, po, pe = _kappa_from_table(table)

    return {
        "n_labeled": len(pairs),
        "n_scored": n_scored,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "tpr": tpr,
        "tpr_ci": list(tpr_ci),
        "tnr": tnr,
        "tnr_ci": list(tnr_ci),
        "kappa": kappa,
        "po": po,
        "pe": pe,
        "confusion": table,
    }
