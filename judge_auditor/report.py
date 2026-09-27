"""Judge-health report: JSON + Markdown rendering of probe results."""

import datetime
import json
from pathlib import Path

from .runner import ProbeResult


def grade(results: list[ProbeResult]) -> str:
    """Overall grade from per-probe outcomes.

    FAIL: 2+ probes tripped.
    NEEDS ATTENTION: exactly 1 probe tripped.
    INCONCLUSIVE: no trips but at least one inconclusive probe.
    PASS: otherwise.
    """
    tripped = sum(1 for r in results if r.tripped)
    inconclusive = sum(1 for r in results if r.inconclusive and not r.tripped)
    if tripped >= 2:
        return "FAIL"
    if tripped == 1:
        return "NEEDS ATTENTION"
    if inconclusive:
        return "INCONCLUSIVE"
    return "PASS"


def build_report(results: list[ProbeResult], adapter_name: str,
                 calibration: dict | None = None) -> dict:
    probe_entries = []
    for r in results:
        probe_entries.append({
            "probe": r.probe_name,
            "version": r.version,
            "expected_direction": r.expected_direction,
            "n_items": r.n_items,
            "n_valid": r.n_valid,
            "n_invalid": r.n_invalid,
            "events": r.events,
            "rate": round(r.rate, 4),
            "p_value": r.p_value,
            "alpha": r.alpha,
            "min_n": r.min_n,
            "two_sided": r.two_sided,
            "tripped": r.tripped,
            "inconclusive": r.inconclusive,
            "reason": r.reason,
            "seed": r.seed,
            # Per-item verdicts (item_id + verdict only, no raw text) so
            # calibration and re-scoring can join labels back to items.
            "verdicts": [{"item_id": v["item_id"], "verdict": v["verdict"]}
                         for v in r.raw_verdicts],
        })
    report = {
        "tool": "judge-auditor",
        "adapter": adapter_name,
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "grade": grade(results),
        "n_probes": len(results),
        "n_tripped": sum(1 for r in results if r.tripped),
        "probes": probe_entries,
    }
    if calibration is not None:
        report["calibration"] = calibration
    return report


def to_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2))
    return path


def to_markdown(report: dict) -> str:
    lines = [
        "# Judge Health Report",
        "",
        f"- Adapter: `{report['adapter']}`",
        f"- Generated: {report['generated_at']}",
        f"- **Grade: {report['grade']}** "
        f"({report['n_tripped']}/{report['n_probes']} probes tripped)",
        "",
        "| Probe | Direction | n | Rate | p-value | Bar | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in report["probes"]:
        if p["inconclusive"]:
            result = "INCONCLUSIVE"
        elif p["tripped"]:
            result = "**TRIPPED**"
        else:
            result = "pass"
        bar = f"p<{p['alpha']}{' (2-sided)' if p['two_sided'] else ''}"
        lines.append(
            f"| {p['probe']} | {p['expected_direction']} | {p['n_valid']} "
            f"| {p['rate']:.3f} | {p['p_value']:.4g} | {bar} | {result} |"
        )
    lines += [
        "",
        "### Notes",
        "- Rate = fraction of valid judgments consistent with the expected bias direction.",
        "- A probe trips only when the exact binomial p-value is below the frozen alpha.",
        "- Probes with >20% unparseable judgments, or fewer valid judgments than the "
        "frozen min_n, are INCONCLUSIVE rather than passed.",
    ]
    cal = report.get("calibration")
    if cal is not None:
        lines += _calibration_markdown(cal)
    return "\n".join(lines) + "\n"


def _calibration_markdown(cal: dict) -> list[str]:
    """Render the ## Calibration section from a calibration dict."""
    lines = [
        "",
        "## Calibration",
        "",
        f"- Protocol: `{cal.get('protocol_sha256') or 'none (ad hoc)'}`",
        f"- Labeler: `{cal.get('labeler_id', '')}` | "
        f"Label schema: {cal.get('label_schema_version', '')} | "
        f"Tie rule: {cal.get('tie_rule', '')}",
        f"- Labels: {cal.get('n_labels', 0)} "
        f"({cal.get('n_unmatched', 0)} unmatched)",
        "",
        "| Probe | n_labeled | TPR [95% CI] | TNR [95% CI] | κ (Po, Pe) |",
        "|---|---|---|---|---|",
    ]
    for name, m in cal.get("per_probe", {}).items():
        tpr_lo, tpr_hi = m["tpr_ci"]
        tnr_lo, tnr_hi = m["tnr_ci"]
        lines.append(
            f"| {name} | {m['n_labeled']} "
            f"| {m['tpr']:.3f} [{tpr_lo:.3f}, {tpr_hi:.3f}] "
            f"| {m['tnr']:.3f} [{tnr_lo:.3f}, {tnr_hi:.3f}] "
            f"| {m['kappa']:.3f} ({m['po']:.3f}, {m['pe']:.3f}) |"
        )
    lines += [
        "",
        "TPR/TNR = judge agreement with decisive human labels (A-better / "
        "B-better); judge TIE against a decisive label counts as a miss. "
        "κ is Cohen's kappa over {A, B, TIE}; human TIEs count in κ but not "
        "in TPR/TNR. CIs are exact Clopper-Pearson 95%.",
    ]
    return lines
