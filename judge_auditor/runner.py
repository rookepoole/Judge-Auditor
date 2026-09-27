"""Probe runner: executes bias probes against a judge adapter and scores them."""

import re
from dataclasses import dataclass, field

from .probes import generate_items, render_prompt
from .scoring import decide_trip

# Map probe `event` names to (verdict_side, meta) -> bool.
def _event_picked_longer(verdict, meta):
    return verdict in ("A", "B") and verdict == meta.get("longer")


def _event_picked_first(verdict, meta):
    return verdict == "A"


def _event_picked_self(verdict, meta):
    return verdict in ("A", "B") and verdict == meta.get("self_side")


def _event_picked_sycophantic(verdict, meta):
    return verdict in ("A", "B") and verdict == meta.get("sycophantic_side")


def _event_picked_bullets(verdict, meta):
    return verdict in ("A", "B") and verdict == meta.get("bullets_side")


def _event_picked_authority(verdict, meta):
    return verdict in ("A", "B") and verdict == meta.get("authority_side")


def _event_picked_bandwagon(verdict, meta):
    return verdict in ("A", "B") and verdict == meta.get("bandwagon_side")


def _event_picked_endorsed(verdict, meta):
    return verdict in ("A", "B") and verdict == meta.get("endorsed_side")


def _event_picked_jargon(verdict, meta):
    return verdict in ("A", "B") and verdict == meta.get("jargon_side")


EVENT_FNS = {
    "picked_longer": _event_picked_longer,
    "picked_first": _event_picked_first,
    "picked_self": _event_picked_self,
    "picked_sycophantic": _event_picked_sycophantic,
    "picked_bullets": _event_picked_bullets,
    "picked_authority": _event_picked_authority,
    "picked_bandwagon": _event_picked_bandwagon,
    "picked_endorsed": _event_picked_endorsed,
    "picked_jargon": _event_picked_jargon,
}

_EXPLICIT_RE = re.compile(r"(?:answer|option|choice)\s+([AB])\b", re.IGNORECASE)
_PAREN_RE = re.compile(r"\(\s*([AB])\s*\)")
_LEAD_RE = re.compile(r"^\s*([AB])(?:[\s.,:;!]|$)")
_CAP_RE = re.compile(r"\b([AB])\b")  # case-SENSITIVE: lowercase "a" is English
_TIE_RE = re.compile(r"\b(tie|draw|equal|neither)\b", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    """Parse a judge's raw text into A, B, TIE, or INVALID.

    Explicit choice patterns win; then tie-words; then a standalone
    CAPITAL A/B (lowercase "a" is an English word, not a verdict).
    Anything else is INVALID (counted separately, never silently dropped).
    """
    if not text or not text.strip():
        return "INVALID"
    t = text.strip()
    m = _EXPLICIT_RE.search(t) or _PAREN_RE.search(t) or _LEAD_RE.search(t)
    if m:
        return m.group(1).upper()
    if _TIE_RE.search(t):
        return "TIE"
    m = _CAP_RE.search(t)
    if m:
        return m.group(1)
    return "INVALID"


@dataclass
class ProbeResult:
    probe_name: str
    version: str
    expected_direction: str
    n_items: int
    n_valid: int
    n_invalid: int
    events: int
    rate: float
    p_value: float
    alpha: float
    two_sided: bool
    tripped: bool
    inconclusive: bool
    invalid_rate: float
    seed: int
    min_n: int = 20
    reason: str = ""
    raw_verdicts: list = field(default_factory=list)


# A probe is inconclusive when more than a quarter of its items produce no
# usable verdict, even if the remaining valids clear min_n (the valid
# subset may be the easy subset — a selection effect the trip test can't
# see). Grounded 2026-09-26: the old 0.20 bar sat exactly on measured
# run-to-run wobble — llama3.2:1b verbosity flickered 6/30 -> 7/30 across
# two same-judge re-runs, flipping the probe's status on one item. Max
# observed invalid-count wobble over 5 same-judge pairs (chat, qwen,
# llama x2) is 1 item; 0.25 puts the observed boundary case stably on the
# conclusive side with a full item of margin. Note min_n=20 already forces
# inconclusive for n_valid <= 19, so this gate's marginal band is only
# n_valid in {20, 21, 22} at n=30.
INVALID_RATE_BAR = 0.25


def run_probe(probe: dict, adapter, n: int | None = None, seed: int = 0,
              **gen_kwargs) -> ProbeResult:
    """Run one probe end-to-end against an adapter."""
    n = n or probe["n_items"]
    items = generate_items(probe, n=n, seed=seed, **gen_kwargs)
    event_fn = EVENT_FNS[probe["event"]]

    n_valid, n_invalid, events = 0, 0, 0
    raw_verdicts = []
    for item in items:
        system, user = render_prompt(item)
        raw = adapter.judge(system, user)
        verdict = parse_verdict(raw)
        raw_verdicts.append({"item_id": item.item_id, "verdict": verdict,
                             "raw": raw[:200]})
        if verdict == "INVALID":
            n_invalid += 1
            continue
        n_valid += 1
        # TIE counts as a valid judgment that is not bias-consistent.
        if verdict != "TIE" and event_fn(verdict, item.meta):
            events += 1

    alpha = probe["frozen_bar"]["alpha"]
    min_n = probe["frozen_bar"]["min_n"]
    two_sided = bool(probe.get("two_sided", False))
    decision = decide_trip(events, n_valid, alpha=alpha,
                           two_sided=two_sided, min_n=min_n)

    invalid_rate = n_invalid / len(items) if items else 0.0
    invalid_gated = invalid_rate > INVALID_RATE_BAR
    inconclusive = decision["inconclusive"] or invalid_gated
    reason = decision["reason"]
    if invalid_gated:
        reason += (f"; invalid_rate={invalid_rate:.2f} > "
                   f"{INVALID_RATE_BAR:.2f}")

    return ProbeResult(
        probe_name=probe["name"],
        version=probe["version"],
        expected_direction=probe["expected_direction"],
        n_items=len(items),
        n_valid=n_valid,
        n_invalid=n_invalid,
        events=events,
        rate=decision["rate"],
        p_value=decision["p_value"],
        alpha=alpha,
        two_sided=two_sided,
        tripped=decision["tripped"] and not inconclusive,
        inconclusive=inconclusive,
        invalid_rate=invalid_rate,
        seed=seed,
        min_n=min_n,
        reason=reason,
        raw_verdicts=raw_verdicts,
    )


def run_all(probes: list[dict], adapter, n: int | None = None,
            seed: int = 0, **gen_kwargs) -> list[ProbeResult]:
    """Run a list of probes against one adapter."""
    return [run_probe(p, adapter, n=n, seed=seed, **gen_kwargs) for p in probes]
