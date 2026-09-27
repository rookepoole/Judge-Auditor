"""CLI for the Judge Auditor v0 probe battery."""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

from .adapters import (AnthropicAdapter, CustomHTTPAdapter, OllamaAdapter,
                       OpenAIAdapter)
from .calibration import (CalibrationRefusal, ProtocolMismatch,
                          calibrate_files, calibrate_multi_files,
                          freeze as freeze_protocol,
                          load_labels, verify as verify_protocol)
from .monitor import (DEFAULT_DB, compare, get_baseline, get_snapshot,
                      list_snapshots, pin_baseline, snapshot_battery,
                      verify_chain)
from .monitor.chat_import import snapshot_from_chat
from .probes import generate_items, load_probe, render_prompt
from .report import build_report, to_json, to_markdown
from .runner import run_all

PROBES_DIR = Path(__file__).resolve().parent.parent / "probes"


def _fail(message):
    """Loud nonzero exit for protocol/calibration failures."""
    print("\n" + "!" * 70, file=sys.stderr)
    print(f"!! {message}", file=sys.stderr)
    print("!" * 70 + "\n", file=sys.stderr)
    raise SystemExit(1)


def _discover_probes(names=None):
    probes = [load_probe(p) for p in sorted(PROBES_DIR.glob("*.yaml"))]
    if names:
        wanted = set(names)
        probes = [p for p in probes if p["name"] in wanted]
        missing = wanted - {p["name"] for p in probes}
        if missing:
            raise SystemExit(f"unknown probe(s): {sorted(missing)}")
    return probes


def list_probes():
    for path in sorted(PROBES_DIR.glob("*.yaml")):
        probe = load_probe(path)
        print(f"{probe['name']}  v{probe['version']}  - {probe['description']}")


def build_adapter(args):
    name = args.adapter
    if name == "openai":
        return OpenAIAdapter(api_key=args.api_key,
                             base_url=args.base_url or
                             "https://api.openai.com/v1",
                             model=args.model or "gpt-4o-mini")
    if name == "anthropic":
        return AnthropicAdapter(api_key=args.api_key,
                                base_url=args.base_url or
                                "https://api.anthropic.com",
                                model=args.model or "claude-haiku-4-5")
    if name == "ollama":
        return OllamaAdapter(base_url=args.base_url or
                             "http://localhost:11434/v1",
                             model=args.model or "llama3")
    if name == "custom":
        if not args.url:
            raise SystemExit("--url is required for the custom adapter")
        headers = {}
        for h in args.header or []:
            key, _, value = h.partition(":")
            headers[key.strip()] = value.strip()
        return CustomHTTPAdapter(url=args.url, headers=headers)
    raise SystemExit(f"unknown adapter: {name}")


def cmd_run(args):
    if args.probe == "all":
        probes = [load_probe(p) for p in sorted(PROBES_DIR.glob("*.yaml"))]
    else:
        probes = [load_probe(PROBES_DIR / f"{args.probe}.yaml")]
    adapter = build_adapter(args)
    gen_kwargs = {}
    if args.self_name:
        gen_kwargs["self_name"] = args.self_name
    results = run_all(probes, adapter, n=args.n, seed=args.seed, **gen_kwargs)
    report = build_report(results, adapter.name)

    out_dir = Path(args.out_dir or "judge-report")
    out_dir.mkdir(parents=True, exist_ok=True)
    to_json(report, out_dir / "report.json")
    (out_dir / "report.md").write_text(to_markdown(report))

    print(to_markdown(report))
    print(f"Report written to {out_dir}/report.json and {out_dir}/report.md")
    return 0 if report["grade"] == "PASS" else 1


# -- calibration subcommands ---------------------------------------------

def _parse_probe_selection(value):
    if value is None or value == "all":
        return None
    return [p.strip() for p in value.split(",") if p.strip()]


def cmd_protocol_freeze(args):
    probes = _discover_probes(_parse_probe_selection(args.probes))
    if not probes:
        raise SystemExit("no probes selected")
    protocol = freeze_protocol(probes, args.n, args.seed, args.out)
    print(f"protocol frozen -> {args.out}")
    print(f"sha256: {protocol['sha256']} (+ {args.out}.sha256 sidecar)")
    return 0


def cmd_protocol_verify(args):
    try:
        protocol = verify_protocol(args.protocol, args.run,
                                   labels_path=args.labels)
    except (ProtocolMismatch, ValueError) as e:
        _fail(f"PROTOCOL VERIFICATION FAILED: {e}")
    print(f"protocol verified OK: {args.protocol}")
    print(f"  sha256: {protocol['sha256']}")
    print(f"  probes: {', '.join(p['name'] for p in protocol['probes'])} "
          f"(n={protocol['n']}, seed={protocol['seed']})")
    return 0


def cmd_calibrate(args):
    try:
        if args.protocol:
            for lp in args.labels:
                verify_protocol(args.protocol, args.run, labels_path=lp)
            with open(args.protocol, encoding="utf-8") as f:
                protocol_sha = json.load(f).get("sha256")
        else:
            protocol_sha = None
        cal = calibrate_files(args.run, args.labels, args.judge_id)
    except CalibrationRefusal as e:
        _fail(f"CALIBRATION REFUSED: {e}")
    except (ProtocolMismatch, ValueError) as e:
        _fail(f"CALIBRATION FAILED: {e}")
    if protocol_sha:
        cal["protocol_sha256"] = protocol_sha

    out_dir = Path(args.out_dir or "judge-calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(
        json.dumps(cal, indent=2, sort_keys=True) + "\n")

    with open(args.run, encoding="utf-8") as f:
        report = json.load(f)
    report["calibration"] = cal
    to_json(report, out_dir / "report.json")
    (out_dir / "report.md").write_text(to_markdown(report))

    print(f"calibration written to {out_dir}/calibration.json")
    print(f"calibrated report written to {out_dir}/report.json and "
          f"{out_dir}/report.md")
    for name, m in cal["per_probe"].items():
        print(f"  {name}: n={m['n_labeled']} tpr={m['tpr']:.3f} "
              f"tnr={m['tnr']:.3f} kappa={m['kappa']:.3f}")
    return 0


def cmd_calibrate_multi(args):
    try:
        if args.protocol:
            for lp in args.labels:
                verify_protocol(args.protocol, args.run, labels_path=lp)
            with open(args.protocol, encoding="utf-8") as f:
                protocol_sha = json.load(f).get("sha256")
        else:
            protocol_sha = None
        cal = calibrate_multi_files(args.run, args.labels, args.judge_id)
    except CalibrationRefusal as e:
        _fail(f"CALIBRATION REFUSED: {e}")
    except (ProtocolMismatch, ValueError) as e:
        _fail(f"CALIBRATION FAILED: {e}")
    if protocol_sha:
        cal["protocol_sha256"] = protocol_sha

    out_dir = Path(args.out_dir or "judge-calibration-multi")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration-multi.json").write_text(
        json.dumps(cal, indent=2, sort_keys=True) + "\n")

    print(f"multi-labeler calibration written to "
          f"{out_dir}/calibration-multi.json")
    print(f"labelers: {', '.join(cal['labeler_ids'])} "
          f"(n_items={cal['n_items']}, unresolved={cal['n_unresolved']})")
    ir = cal["inter_rater"]
    print(f"Fleiss kappa: {ir['fleiss_kappa']:.3f} "
          f"(n={ir['fleiss_n_items']})")
    for lid, vs in cal["labeler_vs_gold"].items():
        print(f"  vs-gold {lid}: kappa={vs['kappa']:.3f} "
              f"agreement={vs['agreement']:.3f} n={vs['n']}")
    flagged = [f for f in cal["position_flags"] if f["flagged"]]
    for f in flagged:
        print(f"  POSITION FLAG: {f['labeler']}/{f['probe']}: "
              f"A-rate {f['a_rate']:.2f} (n={f['n']}, p={f['p_value']:.4f})")
    if not flagged:
        print("  position screen: no flags")
    for name, m in cal["per_probe"].items():
        print(f"  {name}: n={m['n_labeled']} tpr={m['tpr']:.3f} "
              f"tnr={m['tnr']:.3f} kappa={m['kappa']:.3f}")
    return 0


def cmd_export_labeling_set(args):
    with open(args.run, encoding="utf-8") as f:
        report = json.load(f)
    wanted = _parse_probe_selection(args.probes)
    entries = []
    for entry in report.get("probes", []):
        if wanted is not None and entry["probe"] not in wanted:
            continue
        # NOTE: items are regenerated from the probe YAML with the report's
        # seed/n. If the original run used generation overrides (e.g.
        # --self-name), the same overrides must be used here; they are not
        # recorded in report.json v1.
        probe = load_probe(PROBES_DIR / f"{entry['probe']}.yaml")
        for item in generate_items(probe, n=entry["n_items"],
                                   seed=entry["seed"]):
            system, user = render_prompt(item)
            entries.append({"probe": entry["probe"],
                            "item_id": item.item_id,
                            "system": system, "user": user})
    rng = random.Random(args.seed)
    rng.shuffle(entries)

    out = Path(args.out)
    sealed_map = {}
    with open(out, "w", encoding="utf-8") as f:
        for k, e in enumerate(entries):
            lid = f"labeling-{k:04d}"
            sealed_map[lid] = {"probe": e["probe"],
                               "item_id": e["item_id"]}
            f.write(json.dumps({"labeling_id": lid,
                                "system": e["system"],
                                "user": e["user"]}) + "\n")
    sealed_path = out.with_suffix(".sealed.json")
    sealed_path.write_text(json.dumps(sealed_map, indent=1) + "\n")
    print(f"wrote {len(entries)} labeling items -> {out}")
    print(f"sealed map -> {sealed_path} (the labeler must not read it)")
    return 0


def cmd_import_labels(args):
    with open(args.map, encoding="utf-8") as f:
        sealed = json.load(f)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    seen = set()
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"replies line {line_no}: not valid JSON: {e}")
            lid = d.get("labeling_id")
            if lid in seen:
                raise SystemExit(
                    f"replies line {line_no}: duplicate labeling_id {lid!r}")
            seen.add(lid)
            if lid not in sealed:
                raise SystemExit(
                    f"replies line {line_no}: unknown labeling_id {lid!r}")
            label = d.get("label")
            if label not in ("A", "B", "TIE"):
                raise SystemExit(
                    f"replies line {line_no}: invalid field 'label': "
                    f"expected 'A'|'B'|'TIE', got {label!r}")
            s = sealed[lid]
            records.append({
                "item_id": s["item_id"],
                "label": label,
                "labeler_id": args.labeler_id,
                "timestamp": now,
                "probe": s["probe"],
            })
    # Final gate: the emitted file must parse under labels-v1 rules.
    load_labels_path = Path(args.out)
    load_labels_path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    load_labels(str(load_labels_path))
    print(f"wrote {len(records)} labels -> {args.out}")
    return 0


# -- drift monitoring subcommands --------------------------------------

def _drift_db(args):
    return args.db or str(DEFAULT_DB)


def cmd_drift_snapshot(args):
    adapter = build_adapter(args)
    snap_id = snapshot_battery(
        adapter, args.adapter_name or adapter.name,
        n=args.n, seed=args.seed, label=args.label or "",
        db_path=_drift_db(args))
    snap = get_snapshot(_drift_db(args), snap_id)
    broken = verify_chain(_drift_db(args))
    print(f"snapshot {snap_id} stored "
          f"(adapter={snap['adapter_name']}, grade={snap['grade']}, "
          f"seed={snap['seed']}, n={snap['n']}, "
          f"protocol={snap['protocol_sha256'][:12]}...)")
    if broken is not None:
        print(f"WARNING: hash chain broken at snapshot {broken}",
              file=sys.stderr)
        return 1
    return 0


def cmd_drift_baseline(args):
    snap_id = pin_baseline(_drift_db(args), args.id, reason=args.reason or "")
    snap = get_snapshot(_drift_db(args), snap_id)
    print(f"baseline pinned to snapshot {snap_id} "
          f"(adapter={snap['adapter_name']}, label={snap['label']!r}, "
          f"grade={snap['grade']})")
    return 0


def cmd_drift_report(args):
    db = _drift_db(args)
    broken = verify_chain(db)
    if broken is not None:
        print("\n" + "!" * 70, file=sys.stderr)
        print(f"!! DRIFT DB INTEGRITY FAILURE: hash chain broken at "
              f"snapshot {broken}.", file=sys.stderr)
        print("!! Snapshots may have been edited or deleted; treat all "
              "comparisons below as suspect.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)

    snaps = list_snapshots(db, since=args.since)
    if not snaps:
        print("no snapshots in db")
        return 0
    baseline_id = args.baseline
    if baseline_id is None:
        baseline_id = get_baseline(db)
    if baseline_id is None:
        baseline_id = snaps[0]["id"]
    target_id = args.snapshot if args.snapshot is not None else snaps[-1]["id"]

    print(f"# Drift report (db: {db})")
    print("")
    print("| id | ts | adapter | grade | seed | n | label | baseline |")
    print("|---|---|---|---|---|---|---|---|")
    for s in snaps:
        mark = " <- baseline" if s["id"] == baseline_id else ""
        mark += " <- target" if s["id"] == target_id else ""
        print(f"| {s['id']} | {s['ts'][:19]} | {s['adapter_name']} | "
              f"{s['grade']} | {s['seed']} | {s['n']} | {s['label']} |{mark} |")
    print("")

    try:
        base = get_snapshot(db, baseline_id)
        target = get_snapshot(db, target_id)
    except KeyError as e:
        print(f"drift-report: {e}", file=sys.stderr)
        return 1
    try:
        cmp = compare(base, target)
    except ValueError as e:
        print(f"comparison refused: {e}", file=sys.stderr)
        return 1
    if cmp["seed_mismatch"]:
        print(cmp["seed_warning"])
        print("")
    print(f"## {base['adapter_name']} (id {baseline_id}) -> "
          f"{target['adapter_name']} (id {target_id})")
    print(f"protocol: {cmp['protocol_sha256'][:16]}... "
          f"(compared {cmp['summary']['n_compared']} probes)")
    print("")
    print("| probe | rate_a | rate_b | delta | fisher p | holm | "
          "trip_flip | alert | watch | reason |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, d in cmp["per_probe"].items():
        if d.get("status") != "compared" and not d.get("alert"):
            print(f"| {name} | — | — | — | — | — | — | no | no | {d['status']} |")
            continue
        reason = d.get("alert_reason") or d.get("watch_reason") or d.get("status", "")
        print(f"| {name} | {d.get('rate_a', float('nan')):.3f} | "
              f"{d.get('rate_b', float('nan')):.3f} | "
              f"{d.get('delta', float('nan')):+.3f} | "
              f"{d.get('fisher_p', float('nan')):.4g} | "
              f"{'yes' if d.get('holm_significant') else 'no'} | "
              f"{'yes' if d.get('trip_flip') else 'no'} | "
              f"{'ALERT' if d.get('alert') else 'no'} | "
              f"{'WATCH' if d.get('watch') else 'no'} | "
              f"{reason} |")
    print("")
    s = cmp["summary"]
    print(f"alerts: {s['n_alerts']} "
          f"({', '.join(s['alerting_probes']) or 'none'})")
    print(f"watch: {s['n_watch']} "
          f"({', '.join(s['watch_probes']) or 'none'})")
    return 1 if broken is not None or s["n_alerts"] else 0


def cmd_drift_import_chat(args):
    probes = ([p.strip() for p in args.probes.split(",") if p.strip()]
              if args.probes else None)
    snap_id = snapshot_from_chat(
        args.blinded, args.verdicts, args.adapter_name,
        label=args.label or "", db_path=_drift_db(args),
        n=args.n, probes=probes, seed=args.seed)
    snap = get_snapshot(_drift_db(args), snap_id)
    print(f"chat snapshot {snap_id} stored "
          f"(adapter={snap['adapter_name']}, grade={snap['grade']}, "
          f"n={snap['n']}, seed={snap['seed']})")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="judge-auditor",
                                     description="Audit LLM judges for bias.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-probes", help="List available probes.")

    run = sub.add_parser("run", help="Run probes against a judge.")
    run.add_argument("--probe", default="all",
                     help="Probe name or 'all'.")
    run.add_argument("--adapter", default="openai",
                     choices=["openai", "anthropic", "ollama", "custom"])
    run.add_argument("--base-url", default=None)
    run.add_argument("--url", default=None,
                     help="Endpoint URL (custom adapter).")
    run.add_argument("--model", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--header", action="append", default=None,
                     help="Extra header as Key:Value (custom adapter; repeatable).")
    run.add_argument("--n", type=int, default=None)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--self-name", default=None,
                     help="Override the self-attribution label.")
    run.add_argument("--out-dir", default="judge-report")

    pf = sub.add_parser("protocol-freeze",
                        help="Freeze a pre-registered calibration protocol.")
    pf.add_argument("--probes", default="all",
                    help="'all' or comma-separated probe names.")
    pf.add_argument("--n", type=int, required=True)
    pf.add_argument("--seed", type=int, required=True)
    pf.add_argument("--out", required=True, help="Protocol JSON path.")

    pv = sub.add_parser("protocol-verify",
                        help="Verify a run report (and labels) against a protocol.")
    pv.add_argument("--protocol", required=True)
    pv.add_argument("--run", required=True, help="report.json path.")
    pv.add_argument("--labels", default=None, help="labels.jsonl path.")

    cal = sub.add_parser("calibrate",
                         help="Calibrate judge verdicts against human labels.")
    cal.add_argument("--labels", required=True, action="append",
                     help="labels.jsonl path (repeatable; records are "
                     "concatenated).")
    cal.add_argument("--run", required=True, help="report.json path.")
    cal.add_argument("--judge-id", required=True,
                     help="Identity of the judge under audit.")
    cal.add_argument("--protocol", default=None,
                     help="Verify against this protocol first.")
    cal.add_argument("--out-dir", default="judge-calibration")

    cm = sub.add_parser("calibrate-multi",
                        help="Calibrate against adjudicated multi-labeler "
                             "gold (adjudication-v1).")
    cm.add_argument("--labels", required=True, action="append",
                    help="labels.jsonl path, repeatable "
                    "(one or more labelers, labels-v2; records from all "
                    "files are concatenated).")
    cm.add_argument("--run", required=True, help="report.json path.")
    cm.add_argument("--judge-id", required=True,
                    help="Identity of the judge under audit.")
    cm.add_argument("--protocol", default=None,
                    help="Verify against this protocol first.")
    cm.add_argument("--out-dir", default="judge-calibration-multi")

    ex = sub.add_parser("export-labeling-set",
                        help="Export a blinded labeling set from a run report.")
    ex.add_argument("--run", required=True, help="report.json path.")
    ex.add_argument("--probes", default="all",
                    help="'all' or comma-separated probe names.")
    ex.add_argument("--seed", type=int, default=20260926)
    ex.add_argument("--out", required=True, help="labeling.jsonl path.")

    im = sub.add_parser("import-labels",
                        help="Convert labeler replies into labels.jsonl.")
    im.add_argument("--in", dest="input", required=True,
                    help="replies.jsonl path.")
    im.add_argument("--map", required=True,
                    help="labeling.sealed.json path.")
    im.add_argument("--labeler-id", required=True)
    im.add_argument("--out", required=True, help="labels.jsonl path.")

    ds = sub.add_parser("drift-snapshot",
                        help="Run the battery and store a drift snapshot.")
    ds.add_argument("--adapter", default="openai",
                    choices=["openai", "anthropic", "ollama", "custom"])
    ds.add_argument("--base-url", default=None)
    ds.add_argument("--url", default=None,
                    help="Endpoint URL (custom adapter).")
    ds.add_argument("--model", default=None)
    ds.add_argument("--api-key", default=None)
    ds.add_argument("--header", action="append", default=None,
                    help="Extra header as Key:Value (custom adapter; repeatable).")
    ds.add_argument("--adapter-name", default=None,
                    help="Name recorded on the snapshot (default: adapter name).")
    ds.add_argument("--seed", type=int, default=0)
    ds.add_argument("--n", type=int, default=30)
    ds.add_argument("--label", default=None)
    ds.add_argument("--db", default=None,
                    help="Drift db path (default: audit/drift.db).")

    db_ = sub.add_parser("drift-baseline",
                         help="Pin a snapshot as the drift baseline.")
    db_.add_argument("--id", type=int, required=True,
                     help="Snapshot id to pin.")
    db_.add_argument("--reason", default=None)
    db_.add_argument("--db", default=None)

    dr = sub.add_parser("drift-report",
                        help="Trend table + baseline-vs-target drift report.")
    dr.add_argument("--baseline", type=int, default=None,
                    help="Baseline snapshot id (default: pinned, else earliest).")
    dr.add_argument("--snapshot", type=int, default=None,
                    help="Target snapshot id (default: latest).")
    dr.add_argument("--since", default=None,
                    help="Only list snapshots with ts >= this ISO date.")
    dr.add_argument("--db", default=None)

    di = sub.add_parser("drift-import-chat",
                        help="Store a snapshot from recorded chat verdicts.")
    di.add_argument("--blinded", required=True, help="blinded.jsonl path.")
    di.add_argument("--verdicts", required=True, help="verdicts.jsonl path.")
    di.add_argument("--adapter-name", required=True)
    di.add_argument("--label", default=None)
    di.add_argument("--n", type=int, default=None,
                    help="Items per probe (default: len(blinded)/n_probes).")
    di.add_argument("--probes", default=None,
                    help="Comma-separated probe subset (default: all).")
    di.add_argument("--seed", type=int, default=0,
                    help="Item-generation seed (must match the seed used to "
                         "build the blinded set).")
    di.add_argument("--db", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "list-probes":
        list_probes()
        return 0
    if args.cmd == "protocol-freeze":
        return cmd_protocol_freeze(args)
    if args.cmd == "protocol-verify":
        return cmd_protocol_verify(args)
    if args.cmd == "calibrate":
        return cmd_calibrate(args)
    if args.cmd == "calibrate-multi":
        return cmd_calibrate_multi(args)
    if args.cmd == "export-labeling-set":
        return cmd_export_labeling_set(args)
    if args.cmd == "import-labels":
        return cmd_import_labels(args)
    if args.cmd == "drift-snapshot":
        return cmd_drift_snapshot(args)
    if args.cmd == "drift-baseline":
        return cmd_drift_baseline(args)
    if args.cmd == "drift-report":
        return cmd_drift_report(args)
    if args.cmd == "drift-import-chat":
        return cmd_drift_import_chat(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
