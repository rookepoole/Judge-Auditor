"""Tests for drift monitoring: Fisher exact, Holm, snapshots/chain, compare.

Fisher cases are hand-computed independently (see comments); the
[[2,8],[8,2]] case was verified by manual enumeration of the
hypergeometric tables.
"""

import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from judge_auditor.adapters.base import JudgeAdapterError
from judge_auditor.monitor import (
    compare,
    drift_protocol_hash,
    ensure_drift_protocol,
    get_baseline,
    get_snapshot,
    holm_reject,
    init_db,
    list_snapshots,
    pin_baseline,
    snapshot_battery,
    verify_chain,
)
from judge_auditor.monitor.chat_import import (
    replay_adapter_for,
    snapshot_from_chat,
)
from judge_auditor.monitor.snapshots import discover_probes
from judge_auditor.scoring import decide_trip, fisher_exact_2x2
from tests.mock_judge import MockJudgeServer
from judge_auditor.adapters import OpenAIAdapter
from judge_auditor.probes import generate_items, render_prompt


def _mock_adapter(server, behavior):
    return OpenAIAdapter(base_url=server.base_url + f"/v1?behavior={behavior}",
                         api_key="mock")


def _probe(name, events, n_valid, tripped, inconclusive=False):
    return {"probe_name": name, "version": "1.0.0",
            "expected_direction": "test", "n_items": n_valid,
            "n_valid": n_valid, "n_invalid": 0, "events": events,
            "rate": events / n_valid if n_valid else 0.0,
            "p_value": 1.0, "alpha": 0.05, "two_sided": False,
            "tripped": tripped, "inconclusive": inconclusive,
            "invalid_rate": 0.0, "seed": 0, "min_n": 20, "reason": ""}


def _snap(sid, proto, seed, probe_rows):
    return {"id": sid, "protocol_sha256": proto, "seed": seed,
            "probes": probe_rows}


class TestFisherExact(unittest.TestCase):
    def test_strongly_significant(self):
        # [[30,0],[15,15]]: margins R=(30,30), C=(45,15), N=60; x in 15..30.
        # Observed x=30 has numerator C(30,30)*C(30,15)=155117520; the only
        # other table with numerator <= that is x=15 (equal by symmetry).
        # p = 2*155117520 / C(60,45) = 5.832133695832843e-06.
        import math
        expected = 2 * math.comb(30, 15) / math.comb(60, 45)
        p = fisher_exact_2x2(30, 0, 15, 15)
        self.assertLess(p, 0.05)
        self.assertAlmostEqual(p, expected, places=15)

    def test_null_is_exactly_one(self):
        # [[15,15],[15,15]]: the observed table is the modal table, so
        # every table qualifies and the probabilities sum to 1.
        p = fisher_exact_2x2(15, 15, 15, 15)
        self.assertAlmostEqual(p, 1.0, places=9)

    def test_hand_computed_small(self):
        # [[2,8],[8,2]]: R=(10,10), C=(10,10), N=20. Observed numerator
        # C(10,2)*C(10,8)=2025. Tables with numerator <= 2025 are
        # x in {0,1,2,8,9,10} with numerators 1,100,2025,2025,100,1.
        # p = 4252 / C(20,10) = 4252/184756.
        p = fisher_exact_2x2(2, 8, 8, 2)
        self.assertAlmostEqual(p, 4252 / 184756, places=12)

    def test_degenerate_margins(self):
        self.assertEqual(fisher_exact_2x2(0, 0, 5, 5), 1.0)
        self.assertEqual(fisher_exact_2x2(0, 30, 0, 30), 1.0)
        self.assertEqual(fisher_exact_2x2(0, 0, 0, 0), 1.0)

    def test_bounds(self):
        for args in [(30, 0, 15, 15), (0, 30, 30, 0), (12, 18, 7, 23)]:
            p = fisher_exact_2x2(*args)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


class TestHolm(unittest.TestCase):
    def test_step_down_rejects_exactly_first(self):
        # alpha=0.05, m=9: thresholds 0.05/9=0.0056, 0.05/8=0.00625, ...
        # p_(0)=0.001 <= 0.0056 -> reject; p_(1)=0.01 > 0.00625 -> stop.
        ps = [0.001, 0.01, 0.02, 0.03, 0.04, 0.2, 0.5, 0.7, 0.9]
        self.assertEqual(holm_reject(ps, 0.05), {0})

    def test_all_reject(self):
        self.assertEqual(holm_reject([0.001] * 9, 0.05), set(range(9)))

    def test_none_reject(self):
        self.assertEqual(holm_reject([0.5] * 9, 0.05), set())

    def test_empty(self):
        self.assertEqual(holm_reject([], 0.05), set())


class TestSnapshots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "drift.db")
        self.server = MockJudgeServer().start()

    def tearDown(self):
        self.server.stop()
        self.tmp.cleanup()

    def test_round_trip_and_chain(self):
        a = _mock_adapter(self.server, "neutral_alternating")
        sid = snapshot_battery(a, "demo-neutral", n=30, seed=0,
                               label="S1", db_path=self.db)
        snap = get_snapshot(self.db, sid)
        self.assertEqual(snap["adapter_name"], "demo-neutral")
        self.assertEqual(snap["seed"], 0)
        self.assertEqual(snap["n"], 30)
        self.assertEqual(snap["label"], "S1")
        self.assertEqual(len(snap["probes"]), 9)
        self.assertEqual(snap["grade"], "PASS")
        names = [p["probe_name"] for p in snap["probes"]]
        self.assertEqual(names, sorted(names))
        entry = snap["probes"][0]
        for key in ("probe_name", "version", "events", "n_valid", "rate",
                    "p_value", "tripped", "inconclusive", "seed"):
            self.assertIn(key, entry)
        # Protocol hash is the stable drift-protocol hash.
        probes = discover_probes()
        self.assertEqual(snap["protocol_sha256"],
                         drift_protocol_hash(probes, 30, self.db))
        # Chain verifies; a second snapshot links cleanly.
        self.assertIsNone(verify_chain(self.db))
        sid2 = snapshot_battery(a, "demo-neutral", n=30, seed=0,
                                label="S2", db_path=self.db)
        self.assertGreater(sid2, sid)
        self.assertIsNone(verify_chain(self.db))
        snaps = list_snapshots(self.db)
        self.assertEqual([s["id"] for s in snaps], [sid, sid2])

    def test_tamper_detection(self):
        a = _mock_adapter(self.server, "neutral_alternating")
        sid1 = snapshot_battery(a, "m", n=30, seed=0, db_path=self.db)
        sid2 = snapshot_battery(a, "m", n=30, seed=0, db_path=self.db)
        self.assertIsNone(verify_chain(self.db))
        conn = sqlite3.connect(self.db)
        # Edit a stored field -> chain flags that id.
        row = conn.execute("SELECT probes_json FROM snapshots WHERE id=?",
                           (sid1,)).fetchone()
        tampered = row[0].replace('"events":', '"events":9, "x":', 1)
        self.assertNotEqual(tampered, row[0])
        conn.execute("UPDATE snapshots SET probes_json=? WHERE id=?",
                     (tampered, sid1))
        conn.commit()
        self.assertEqual(verify_chain(self.db), sid1)
        # Delete the tampered row -> the next surviving row's chain breaks.
        conn.execute("DELETE FROM snapshots WHERE id=?", (sid1,))
        conn.commit()
        conn.close()
        self.assertEqual(verify_chain(self.db), sid2)

    def test_baseline_pinning_and_log(self):
        a = _mock_adapter(self.server, "neutral_alternating")
        sid1 = snapshot_battery(a, "m", n=30, seed=0, db_path=self.db)
        sid2 = snapshot_battery(a, "m", n=30, seed=0, db_path=self.db)
        self.assertIsNone(get_baseline(self.db))
        pin_baseline(self.db, sid1, reason="initial")
        self.assertEqual(get_baseline(self.db), sid1)
        pin_baseline(self.db, sid2, reason="re-baseline")
        self.assertEqual(get_baseline(self.db), sid2)
        conn = sqlite3.connect(self.db)
        log = conn.execute(
            "SELECT old_id, new_id, reason FROM baseline_log ORDER BY ts"
        ).fetchall()
        conn.close()
        self.assertEqual([(r[0], r[1], r[2]) for r in log],
                         [(None, sid1, "initial"), (sid1, sid2, "re-baseline")])
        with self.assertRaises(KeyError):
            pin_baseline(self.db, 9999)

    def test_protocol_hash_stable_across_snapshots(self):
        a = _mock_adapter(self.server, "neutral_alternating")
        sid1 = snapshot_battery(a, "m", n=30, seed=0, db_path=self.db)
        sid2 = snapshot_battery(a, "m", n=30, seed=0, db_path=self.db)
        h1 = get_snapshot(self.db, sid1)["protocol_sha256"]
        h2 = get_snapshot(self.db, sid2)["protocol_sha256"]
        self.assertEqual(h1, h2)
        # ... but a battery change re-freezes to a new hash.
        probes = discover_probes()
        other = drift_protocol_hash(probes, n=10, db_path=self.db)
        self.assertNotEqual(h1, other)

    def test_identity_stable_across_cache_refreeze(self):
        # Regression (2026-09-26, protocol-v2): the v1 calibration hash
        # covered created_at, so deleting the drift_protocol.json cache and
        # re-freezing the byte-identical battery minted a new identity and
        # compare() hard-refused. Under v2 the identity is content-defined.
        from judge_auditor.monitor import snapshots as S
        probes = discover_probes()
        h1 = drift_protocol_hash(probes, 30, self.db)
        cache = Path(self.db).parent / "drift_protocol.json"
        self.assertTrue(cache.exists())
        cache.unlink()
        h2 = drift_protocol_hash(probes, 30, self.db)
        self.assertTrue(cache.exists())
        self.assertEqual(h1, h2)
        body = json.loads(cache.read_text(encoding="utf-8"))
        self.assertEqual(body["protocol_version"], "protocol-v2")
        self.assertEqual(body["drift_identity"], h1)

    def test_probe_content_edit_without_version_bump_changes_identity(self):
        # Red-team regression (attack 4b, 2026-09-26): the calibration
        # hash covers probe name+version only, so a bank edit without a
        # version bump used to leave the drift identity unchanged,
        # allowing silent comparison across different instruments.
        import shutil
        from judge_auditor.monitor import snapshots as S
        a = _mock_adapter(self.server, "neutral_alternating")
        sid1 = snapshot_battery(a, "m", n=10, seed=0, db_path=self.db)
        h1 = get_snapshot(self.db, sid1)["protocol_sha256"]
        alt = Path(self.tmp.name) / "probes_alt"
        shutil.copytree(S.PROBES_DIR, alt)
        vfile = alt / "verbosity.yaml"
        vfile.write_text(
            vfile.read_text(encoding="utf-8")
            + "\n# red-team: content edit, version NOT bumped\n",
            encoding="utf-8")
        old_dir = S.PROBES_DIR
        S.PROBES_DIR = alt
        try:
            sid2 = snapshot_battery(a, "m", n=10, seed=0, db_path=self.db)
        finally:
            S.PROBES_DIR = old_dir
        s1, s2 = get_snapshot(self.db, sid1), get_snapshot(self.db, sid2)
        self.assertNotEqual(s1["protocol_sha256"], s2["protocol_sha256"])
        self.assertEqual(h1, s1["protocol_sha256"])
        with self.assertRaises(ValueError):
            compare(s1, s2)

    def test_list_since_filter(self):
        a = _mock_adapter(self.server, "neutral_alternating")
        snapshot_battery(a, "m", n=30, seed=0, db_path=self.db)
        snaps = list_snapshots(self.db)
        self.assertEqual(len(list_snapshots(self.db, since="2000-01-01")), 1)
        self.assertEqual(list_snapshots(self.db, since="2999-01-01"), [])
        self.assertEqual(
            len(list_snapshots(self.db, since=snaps[0]["ts"][:10])), 1)


class TestCompare(unittest.TestCase):
    def test_protocol_mismatch_raises(self):
        a = _snap(1, "aaa", 0, [_probe("verbosity", 12, 30, False)])
        b = _snap(2, "bbb", 0, [_probe("verbosity", 30, 30, True)])
        with self.assertRaises(ValueError) as ctx:
            compare(a, b)
        self.assertIn("different protocols", str(ctx.exception))

    def test_seed_mismatch_proceeds_with_flag(self):
        rows = [_probe("verbosity", 12, 30, False)]
        c = compare(_snap(1, "h", 0, rows), _snap(2, "h", 1, rows))
        self.assertTrue(c["seed_mismatch"])
        self.assertIn("seed mismatch", c["seed_warning"])
        self.assertEqual(c["summary"]["n_alerts"], 0)

    def test_trip_flip_alerts(self):
        a = _snap(1, "h", 0, [_probe("verbosity", 12, 30, False)])
        b = _snap(2, "h", 0, [_probe("verbosity", 30, 30, True)])
        c = compare(a, b)
        d = c["per_probe"]["verbosity"]
        self.assertTrue(d["trip_flip"])
        self.assertTrue(d["alert"])
        self.assertIn("trip_flip", d["alert_reason"])
        self.assertIn("rate_shift", d["alert_reason"])
        self.assertEqual(c["summary"]["alerting_probes"], ["verbosity"])

    def test_inconclusive_involved_no_flip_no_alert(self):
        a = _snap(1, "h", 0,
                 [_probe("verbosity", 12, 30, False, inconclusive=True)])
        b = _snap(2, "h", 0, [_probe("verbosity", 30, 30, True)])
        c = compare(a, b)
        d = c["per_probe"]["verbosity"]
        self.assertEqual(d["status"], "inconclusive-involved")
        self.assertFalse(d["trip_flip"])
        self.assertFalse(d["alert"])
        self.assertEqual(c["summary"]["n_alerts"], 0)

    def test_no_alert_on_small_shift(self):
        a = _snap(1, "h", 0, [_probe("verbosity", 12, 30, False)])
        b = _snap(2, "h", 0, [_probe("verbosity", 14, 30, False)])
        c = compare(a, b)
        d = c["per_probe"]["verbosity"]
        self.assertFalse(d["alert"])
        self.assertEqual(c["summary"]["n_alerts"], 0)


class TestDriftPower(unittest.TestCase):
    """Sub-threshold sensitivity: what the frozen alert rules actually detect.

    The trip boundary at n=30 is analytic (one-sided exact binomial vs 0.5,
    p<0.05 trips iff events >= 20). The tests below pin that boundary through
    the real decide_trip() and verify the alert rules' behavior around it
    through the real compare().
    """

    def test_trip_boundary_is_20_at_n30(self):
        self.assertFalse(decide_trip(19, 30)["tripped"])
        self.assertTrue(decide_trip(20, 30)["tripped"])

    def test_subthreshold_trip_flip_alerts(self):
        # One vote (delta=1/30 ~= 0.033, far below the 0.15 effect gate)
        # flips the trip state 19 -> 20 and must alert via trip flip alone.
        a = _snap(1, "h", 0,
                 [_probe("verbosity", 19, 30, decide_trip(19, 30)["tripped"])])
        b = _snap(2, "h", 0,
                 [_probe("verbosity", 20, 30, decide_trip(20, 30)["tripped"])])
        c = compare(a, b)
        d = c["per_probe"]["verbosity"]
        self.assertTrue(d["trip_flip"])
        self.assertTrue(d["alert"])
        self.assertIn("trip_flip", d["alert_reason"])
        self.assertNotIn("rate_shift", d["alert_reason"])

    def test_power_smoke_seeded(self):
        # Tiny Monte Carlo through the real compare(): no shift -> alerts
        # rare; a large shift on a mid-rate probe -> reliably detected.
        # Margins are generous; the full power curve was computed offline
        # during development via Monte Carlo through the real compare().
        rng = random.Random(20260926)
        n_reps, n = 60, 30
        base_events = 12  # mid-rate probe, far from the trip boundary
        alerts_0 = alerts_big = 0
        for _ in range(n_reps):
            e0 = sum(rng.random() < base_events / n for _ in range(n))
            e_big = sum(rng.random() < 0.9 for _ in range(n))
            a = _snap(1, "h", 0, [_probe("p", e0, n,
                                         decide_trip(e0, n)["tripped"])])
            b = _snap(2, "h", 0, [_probe("p", e0, n,
                                         decide_trip(e0, n)["tripped"])])
            if compare(a, b)["per_probe"]["p"]["alert"]:
                alerts_0 += 1
            b2 = _snap(2, "h", 0, [_probe("p", e_big, n,
                                          decide_trip(e_big, n)["tripped"])])
            if compare(a, b2)["per_probe"]["p"]["alert"]:
                alerts_big += 1
        self.assertLessEqual(alerts_0 / n_reps, 0.15)
        self.assertGreaterEqual(alerts_big / n_reps, 0.80)


class TestWatch(unittest.TestCase):
    def test_near_bar_shift_watches_without_alert(self):
        # 12/30 -> 16/30: |delta|=0.133 >= WATCH_MIN_SHIFT, no trip flip,
        # not Holm-significant at n=30 -> WATCH, no ALERT.
        a = _snap(1, "h", 0, [_probe("verbosity", 12, 30, False)])
        b = _snap(2, "h", 0, [_probe("verbosity", 16, 30, False)])
        c = compare(a, b)
        d = c["per_probe"]["verbosity"]
        self.assertFalse(d["alert"])
        self.assertTrue(d["watch"])
        self.assertIn("near-bar", d["watch_reason"])
        self.assertEqual(c["summary"]["watch_probes"], ["verbosity"])
        self.assertEqual(c["summary"]["n_watch"], 1)
        # WATCH does not touch the alert set.
        self.assertEqual(c["summary"]["n_alerts"], 0)

    def test_tiny_shift_stays_silent(self):
        # 12/30 -> 14/30: |delta|=0.067 < 0.075 -> no WATCH (below the
        # measured run-to-run noise floor).
        a = _snap(1, "h", 0, [_probe("verbosity", 12, 30, False)])
        b = _snap(2, "h", 0, [_probe("verbosity", 14, 30, False)])
        c = compare(a, b)
        d = c["per_probe"]["verbosity"]
        self.assertFalse(d["alert"])
        self.assertFalse(d["watch"])
        self.assertEqual(c["summary"]["n_watch"], 0)

    def test_alerted_probe_does_not_watch(self):
        # Trip flip alerts; WATCH never co-fires with ALERT.
        a = _snap(1, "h", 0, [_probe("verbosity", 12, 30, False)])
        b = _snap(2, "h", 0, [_probe("verbosity", 30, 30, True)])
        d = compare(a, b)["per_probe"]["verbosity"]
        self.assertTrue(d["alert"])
        self.assertFalse(d["watch"])
        self.assertEqual(d["watch_reason"], "")

    def test_inconclusive_involved_no_watch(self):
        a = _snap(1, "h", 0,
                 [_probe("verbosity", 12, 30, False, inconclusive=True)])
        b = _snap(2, "h", 0, [_probe("verbosity", 18, 30, False)])
        d = compare(a, b)["per_probe"]["verbosity"]
        self.assertEqual(d["status"], "inconclusive-involved")
        self.assertFalse(d["alert"])
        self.assertFalse(d["watch"])

    def test_subgate_holm_significant_watch(self):
        # 100/200 -> 128/200: |delta|=0.14 < 0.15 (no rate-shift alert),
        # but Fisher p=0.0063 is Holm-significant at m=1 -> WATCH with the
        # sub-gate reason. (This branch is unreachable at n=30 — proven by
        # exhaustive scan of all (a, b) pairs — but live at larger n.)
        a = _snap(1, "h", 0, [_probe("verbosity", 100, 200, False)])
        b = _snap(2, "h", 0, [_probe("verbosity", 128, 200, False)])
        c = compare(a, b)
        d = c["per_probe"]["verbosity"]
        self.assertTrue(d["holm_significant"])
        self.assertFalse(d["alert"])
        self.assertTrue(d["watch"])
        self.assertIn("sub-gate", d["watch_reason"])


class TestChatImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "drift.db")
        self.dir = Path(self.tmp.name) / "chat"
        self.dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_chat_files(self, n_per_probe=2, seed=0):
        blinded, verdicts = [], []
        k = 0
        for path in sorted((Path(__file__).resolve().parent.parent
                            / "probes").glob("*.yaml")):
            from judge_auditor.probes import load_probe
            probe = load_probe(path)
            for item in generate_items(probe, n=n_per_probe, seed=seed):
                system, user = render_prompt(item)
                bid = f"item-{k:03d}"
                k += 1
                blinded.append({"id": bid, "system": system, "user": user})
                verdicts.append({"id": bid, "verdict": "A"})
        with open(self.dir / "blinded.jsonl", "w") as f:
            for r in blinded:
                f.write(json.dumps(r) + "\n")
        with open(self.dir / "verdicts.jsonl", "w") as f:
            for r in verdicts:
                f.write(json.dumps(r) + "\n")
        return len(blinded)

    def test_replay_adapter(self):
        n = self._write_chat_files()
        adapter = replay_adapter_for(self.dir / "blinded.jsonl",
                                     self.dir / "verdicts.jsonl")
        self.assertEqual(n, 18)
        from judge_auditor.probes import load_probe
        probe = load_probe(Path(__file__).resolve().parent.parent
                           / "probes" / "verbosity.yaml")
        item = generate_items(probe, n=2, seed=0)[0]
        system, user = render_prompt(item)
        self.assertEqual(adapter.judge(system, user), "A")
        with self.assertRaises(JudgeAdapterError):
            adapter.judge("nope", "nothing")

    def test_missing_verdict_raises(self):
        self._write_chat_files()
        lines = (self.dir / "verdicts.jsonl").read_text().splitlines()
        (self.dir / "verdicts.jsonl").write_text("\n".join(lines[1:]) + "\n")
        with self.assertRaises(ValueError):
            replay_adapter_for(self.dir / "blinded.jsonl",
                               self.dir / "verdicts.jsonl")

    def test_snapshot_from_chat(self):
        self._write_chat_files()
        sid = snapshot_from_chat(self.dir / "blinded.jsonl",
                                 self.dir / "verdicts.jsonl",
                                 "chat-judge", label="chat",
                                 db_path=self.db)
        snap = get_snapshot(self.db, sid)
        self.assertEqual(snap["adapter_name"], "chat-judge")
        self.assertEqual(len(snap["probes"]), 9)
        self.assertTrue(all(p["n_items"] == 2 for p in snap["probes"]))
        self.assertTrue(all(p["inconclusive"] for p in snap["probes"]))
        self.assertIsNone(verify_chain(self.db))

    def test_snapshot_from_chat_reseed_roundtrip(self):
        self._write_chat_files(seed=5)
        sid = snapshot_from_chat(self.dir / "blinded.jsonl",
                                 self.dir / "verdicts.jsonl",
                                 "chat-judge", label="reseed",
                                 db_path=self.db, seed=5)
        snap = get_snapshot(self.db, sid)
        self.assertEqual(snap["seed"], 5)
        self.assertEqual(len(snap["probes"]), 9)
        self.assertTrue(all(p["n_items"] == 2 for p in snap["probes"]))
        self.assertIsNone(verify_chain(self.db))

    def test_snapshot_from_chat_seed_mismatch_raises(self):
        self._write_chat_files(seed=5)
        with self.assertRaises(JudgeAdapterError):
            snapshot_from_chat(self.dir / "blinded.jsonl",
                               self.dir / "verdicts.jsonl",
                               "chat-judge", db_path=self.db, seed=6)


if __name__ == "__main__":
    unittest.main()
