# Calibration Harness — FROZEN SPEC

This document freezes the calibration rules for the Judge Auditor
calibration harness (`judge_auditor/calibration/`). Frozen means: changes
require a new tie_rule / label_schema / protocol version id, never a silent
edit. This is the candidate for a later red-team pass; write disputes here,
not in code.

## 1. Purpose

Calibration answers: "when a human says which answer is better, how often
does the judge agree?" It is orthogonal to the bias battery (which asks
"does the judge systematically prefer longer/first/self-flattering answers?").
A judge can be perfectly calibrated and still biased, or unbiased and
wildly miscalibrated.

## 2. Iron rule: labeler ≠ judge

The human (or other independent party) providing labels MUST be a different
party than the judge under audit. A judge labeling its own items is not
calibration; it is self-agreement. Judge-labeler collapse VOIDS the
calibration — the run is refused, not graded with a warning.

- Enforced in `calibrate.calibrate()`: any `labeler_id` equal to `judge_id`
  (case-insensitive) raises `CalibrationRefusal` (a `ValueError`), and the
  CLI exits nonzero.
- Rationale for case-insensitivity: "Muse", "MUSE", "claude" style
  collisions must not slip through string-case games.

## 3. Label schema: labels-v1

JSONL, one object per non-blank line:

    {"item_id": str, "label": "A"|"B"|"TIE",
     "labeler_id": str, "timestamp": ISO-8601 str}

- All four fields required; `label` is exactly "A", "B", or "TIE"
  (lowercase is rejected, not normalized — silent normalization hides
  upstream format drift).
- `timestamp` must parse with `datetime.fromisoformat`; unparseable values
  are rejected.
- An optional extra `"probe"` field (string) pins the probe for the join;
  `import-labels` writes it. Other extra fields are ignored.
- Validation errors raise `ValueError` naming the physical 1-based line
  number and the field.
- The schema version is pinned at the harness level
  (`LABEL_SCHEMA_VERSION = "labels-v1"`); there is no per-file schema
  header, so `protocol.verify()` checks labels by requiring them to parse
  under labels-v1 rules.

## 4. Tie rule: tie-v1 (FROZEN)

Positive class is FIXED: **"the human label says A is better"**. This is
arbitrary and therefore frozen — flipping it silently would invert every
reported TPR/TNR.

Per (human_label, judge_verdict) pair, with judge `INVALID` excluded
pairwise (it contributes to no count, not even a denominator):

| human \ judge | A  | B  | TIE |
|---------------|----|----|-----|
| A             | TP | FN | FN  |
| B             | FP | TN | FP  |
| TIE           | —  | —  | —   |

- A judge TIE against a decisive label is a MISS: it lands in the
  denominator (FN/FP) and never in the numerator. A judge that always
  answers TIE gets TPR = TNR = 0.0, not a free pass.
- A human TIE label carries no decisive signal: excluded from the TPR/TNR
  denominators, but INCLUDED as a third category in kappa (Section 5).
- `tpr = tp / (tp + fn)`; `tnr = tn / (tn + fp)`.

Zero-denominator convention (frozen): if `tp + fn == 0` (resp. `tn + fp`),
` tpr` (resp. `tnr`) is reported as **0.0, never NaN**, with CI (0.0, 1.0).
Rationale: NaN propagates silently into averages and dashboards; 0.0 with
an explicit full-width CI and a visible `n_labeled` says "no evidence of
agreement" without pretending precision. (Note: a zero denominator can
only happen when there are no decisive labels of that class — e.g. all
human labels are TIE — not from judge behavior.)

## 5. Kappa: 3-category, paradox-transparent

- Cohen's kappa over the 3×3 table (judge × human) on categories
  {A, B, TIE}. Judge INVALID verdicts are excluded pairwise; human TIE
  labels are a full third category here (unlike in TPR/TNR).
- Report kappa AND observed agreement Po AND expected agreement Pe.
  Rationale (kappa-paradox transparency): kappa can be near zero while Po
  is high (skewed marginals) or spuriously high while Po is modest;
  reporting Po and Pe makes the paradox visible instead of hiding it
  behind one number.
- Degenerate convention (frozen): if Pe == 1.0 (all mass on a single
  judge row AND a single human column), kappa = 1.0 iff Po == 1.0,
  else 0.0. The fully-degenerate all-TIE-vs-all-TIE case is perfect
  agreement; any other Pe == 1.0 case has no defined disagreement
  structure and is reported as 0.0 rather than NaN.

## 6. Confidence intervals

- TPR/TNR CIs are exact Clopper-Pearson 95% intervals: lower solves
  P(X ≥ k; p) = α/2, upper solves P(X ≤ k; p) = α/2, computed by bisection
  on the exact `scoring.binomial_sf` / `scoring.binomial_cdf` (math.comb;
  no scipy). α = 0.05 fixed.
- Self-consistency CIs (`rescore.compare_rescores`) use the same method.
- Zero-denominator CI: (0.0, 1.0) — full uncertainty, matching the 0.0
  rate convention.

## 7. Join rules

- Labels join to report verdicts by (probe, item_id). The explicit
  `"probe"` field wins; otherwise the probe is inferred from the item id
  prefix (`"verbosity-000"` → `"verbosity"`) and validated against the
  probe registry. A record with no `"probe"` whose id does not embed a
  known probe name (e.g. blinded `"item-015"`) raises CalibrationRefusal
  instead of silently mis-joining — the probe must be pinned at import
  (sealed map), never inferred from a blinded id. (2026-09-26: the
  second-labeler packet caught this as a real silent-split bug.)
- Duplicate labels for the same (labeler_id, probe, item_id) raise —
  ambiguous calibration input must not be silently deduplicated.
- Labels with no matching report verdict are skipped and counted as
  `n_unmatched` (a labeling set that covers a subset of items is the
  normal case).

## 8. Blinded re-scoring

- `rescore.make_rescore_set()` regenerates the report's exact items
  (same probes, seeds, n), then per item applies an A/B side-swap from
  `random.Random(rescore_seed)` and strips judge-identity cues ONLY:
  for the self_preference probe the `"Note: Answer X was written by ..."`
  context is replaced with `""`. No other probe's bias attributes are
  touched.
- Blinded set: JSONL `[{rescore_id, system, user}]`, no probe names,
  shuffled. Sealed map: `{rescore_id: {probe, item_id, swapped}}` — the
  judge must not read it until verdicts are recorded.
- `compare_rescores()` maps each verdict back through the swap
  (A↔B; TIE stays TIE) and reports the fraction matching the original
  report verdict, per probe and overall, with exact binomial CIs.
  Unparseable re-score verdicts are excluded, never counted as mismatches.

## 9. Protocols: protocol-v1 → protocol-v2

- `freeze()` writes the canonical protocol JSON (sorted keys) with the
  sha256 of the canonical hashless form embedded as `"sha256"`, plus a
  `.sha256` sidecar.
- `verify()` recomputes the hash (mismatch → `ProtocolMismatch`), then
  checks probe names/versions, `n_items == n`, report `seed == seed`,
  and per-probe `alpha`/`min_n` against the frozen bars; optionally
  requires the label file to parse under labels-v1.
- `ProtocolMismatch` names the field, the expected value, and the actual
  value. The CLI exits nonzero with a loud message on any mismatch.

### 9a. protocol-v2 (2026-09-26; supersedes the v1 hashing rule, v1 files keep verifying)

- The v2 hash covers **item-generating content only**: `protocol_version`,
  `probes` (name+version), `n`, `seed`, `frozen_bars`, `label_schema`,
  `tie_rule`, `metrics`. `created_at` stays in the document as metadata
  but is excluded from the hash.
- Rationale: v1 hashed `created_at`, so two freezes of the byte-identical
  battery minted different identities; the drift monitor's `compare()`
  hard-refused a legitimate cross-judge comparison on a timestamp-only
  hash difference. The drift identity (`drift-identity-v2`) derives from
  the v2 content hash and is therefore stable across re-freezes.
- `verify()` dispatches on `protocol_version`: v1 files verify against the
  legacy full-body hash, v2 against the content form. Unknown versions are
  rejected. v1 files are never silently re-hashed.

## 10. What calibration does NOT claim

- It does not measure bias (that's the probe battery).
- It does not measure label quality; a sloppy labeler calibrates the
  judge against sloppiness. Inter-labeler agreement is out of scope
  for v1.
- Human TIE labels are honest data ("I can't tell"), not missing data;
  they count in kappa and in `n_labeled`, but not in TPR/TNR.

## 11. Multi-labeler adjudication: labels-v2 + adjudication-v1 (FROZEN 2026-09-26)

Red-team amendments (same day, pre-freeze: no protocol has been frozen
against this section yet — verified 2026-09-26): §11c position-screen
known limitation and §11d two-rater semantics added after the red-team
pass below substantiated both. Behavior unchanged; limitations made
explicit.

Motivation: the first real human calibration (2026-09-26) showed the
calibrator needs calibrating — a single labeler's first-instinct labels
bake that labeler's own biases into "truth" (observed: always-A run on 7
content-identical halo items, with the labeler self-reporting low
confidence exactly on those items). Section 10's "inter-labeler agreement
is out of scope for v1" is superseded by this section; sections 1–9 are
unchanged.

### 11a. Label schema: labels-v2

labels-v1 plus one OPTIONAL field:

    "confidence": 1|2|3 (int)

- 1 = guessing ("I don't trust myself on this one"), 2 = leaning,
  3 = confident.
- Missing `confidence` normalizes to 2 ("unrated", the midpoint — the
  least presumptuous default). A v1 file therefore loads identically
  under v2. This default is frozen; it is not re-estimated from data.
- Out-of-range, non-int (including JSON `true`/`null`), or otherwise
  malformed `confidence` is rejected with a ValueError naming the line
  and field — never silently normalized or coerced.
- `LABEL_SCHEMA_VERSION = "labels-v2"`. `protocol.verify()` with a label
  file still only requires the file to parse (v1 files parse under v2).

### 11b. Adjudication rule: adjudication-v1

- Votes group by (probe, item_id). Duplicate (labeler, probe, item_id)
  raises `CalibrationRefusal`, as in v1.
- Gold = plurality of votes over {A, B, TIE}. TIE is a votable outcome:
  honest "I can't tell" votes push the gold toward TIE, which tie-v1
  then excludes from TPR/TNR denominators (kept in kappa).
- Vote ties break by summed confidence per tied side (higher wins).
  Confidence only ever breaks ties; it never outvotes a plurality.
- Unbreakable ties (equal votes AND equal summed confidence) → gold TIE
  with `unresolved: true`. No decisive signal is manufactured from a
  dead heat. Unresolved items are excluded from TPR/TNR by the frozen
  tie-v1 rule automatically (gold TIE carries no decisive signal).
- Single labeler: adjudication is passthrough (gold = their label), so
  `calibrate_multi()` with one labeler reproduces `calibrate()`
  per-probe exactly. This is a tested invariant, not an aspiration.
- Iron rule unchanged: any `labeler_id` equal to `judge_id`
  (case-insensitive) → `CalibrationRefusal`.
- Zero-denominator conventions mirror tie-v1 everywhere (0.0 with CI
  (0.0, 1.0); Fleiss Pe == 1.0 → 1.0 iff P_bar == 1.0 else 0.0).

### 11c. Inter-rater metrics (all frozen definitions)

- **Fleiss' kappa** over items with ≥ 2 raters, categories {A, B, TIE};
  items with < 2 raters excluded; no ratable items → (0.0, 0).
- **Pairwise Cohen's kappa** (3-category, same table as tie-v1 kappa)
  for every labeler pair over co-labeled items; pairs with no shared
  items omitted.
- **Labeler-vs-gold**: each labeler's kappa/Po/Pe vs the adjudicated
  gold. This is the noisy-labeler finder — the labeler far below the
  rest is the one baking bias into "truth".
- **Position screen** (a SCREEN, not a verdict): per (labeler, probe),
  exact two-sided binomial p of the observed A-count under p = 0.5;
  flagged when p < 0.05 and n ≥ 5. A skewed A-rate is not proof of bias
  (the correct side varies by item); it is a flag for review. The
  n ≥ 5 gate keeps tiny-n noise from flagging. KNOWN LIMITATION
  (red-team 2026-09-26, attack 1 SUBSTANTIATED): the screen tests the
  A-rate against 0.5, not against truth — a labeler in perfect agreement
  with skewed truth flags by construction. Always cross-check a flag
  against labeler-vs-gold before acting on it; a high-vs-gold labeler
  with a position flag is agreeing with a skewed item set, not
  necessarily biased.

### 11d. What adjudication does NOT claim

- It does not make a biased labeler pool unbiased: if every labeler
  shares the same position run, the gold inherits it. Adjudication
  dilutes idiosyncratic labeler noise; it does not remove systematic
  pool-level bias. Labeler diversity is an operational requirement,
  not something the math provides.
- Confidence is self-reported, not calibrated: a labeler who always
  reports 3 only gains influence on vote ties, never on pluralities.
- Majority vote is not truth: on genuinely ambiguous items the gold is
  "what most labelers said", and `unresolved`/`agreement` provenance is
  reported per item so downstream users can see it.
- TWO-RATER SEMANTICS (red-team 2026-09-26, attack 4 SUBSTANTIATED): with
  exactly 2 raters, every disagreement is a vote tie by definition, so
  the confidence tie-break decides 100% of splits — self-reported,
  uncalibrated confidence then dictates the gold wherever raters
  disagree. The "confidence never outvotes a plurality" guarantee is
  vacuous at n_raters = 2. Treat 2-rater adjudication as
  confidence-weighted arbitration, not majority vote; prefer ≥ 3 raters
  whenever the budget allows.
