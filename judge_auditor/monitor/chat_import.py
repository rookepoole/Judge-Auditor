"""Chat-loop snapshot import.

Replays recorded in-chat verdicts (blinded.jsonl + verdicts.jsonl, as
produced by audit/make_blinded.py and the in-chat audit loop) through
the real pipeline: items are regenerated per probe with run_all and the
replay adapter maps each rendered (system, user) prompt to the recorded
verdict. audit/score_blinded.py is untouched and keeps working as-is;
this module only adds snapshot storage on top of the same replay idea.
"""

import json
from pathlib import Path

from ..adapters.base import JudgeAdapter, JudgeAdapterError
from .snapshots import DEFAULT_DB, discover_probes, snapshot_battery


def _read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path} line {line_no}: not valid JSON: {e}")
    return records


def replay_adapter_for(blinded_path, verdicts_path):
    """Build a JudgeAdapter replaying recorded verdicts.

    blinded.jsonl: {"id", "system", "user"} per line.
    verdicts.jsonl: {"id", "verdict"} per line.
    judge(system, user) returns the recorded verdict, or raises
    JudgeAdapterError when the prompt or verdict is missing.
    """
    prompt_to_id = {}
    for rec in _read_jsonl(blinded_path):
        try:
            key = (rec["system"], rec["user"])
        except KeyError as e:
            raise ValueError(
                f"{blinded_path}: record missing field {e}")
        if key in prompt_to_id:
            raise ValueError(
                f"{blinded_path}: duplicate rendered prompt")
        prompt_to_id[key] = rec["id"]

    verdicts = {}
    for rec in _read_jsonl(verdicts_path):
        if "id" not in rec or "verdict" not in rec:
            raise ValueError(
                f"{verdicts_path}: record missing 'id' or 'verdict'")
        verdicts[rec["id"]] = rec["verdict"]

    missing = set(prompt_to_id.values()) - set(verdicts)
    if missing:
        raise ValueError(
            f"{len(missing)} blinded items have no recorded verdict: "
            f"{sorted(missing)[:5]}...")

    class ChatReplayAdapter(JudgeAdapter):
        name = "chat-replay"

        def judge(self, system_prompt, user_prompt):
            bid = prompt_to_id.get((system_prompt, user_prompt))
            if bid is None:
                raise JudgeAdapterError(
                    "chat-replay: no blinded verdict for rendered prompt")
            verdict = verdicts.get(bid)
            if verdict is None:
                raise JudgeAdapterError(
                    f"chat-replay: no verdict recorded for {bid}")
            return verdict

    return ChatReplayAdapter()


def snapshot_from_chat(blinded_path, verdicts_path, adapter_name: str,
                       label: str = "", db_path=DEFAULT_DB, seed: int = 0,
                       n: int | None = None, probes: list | None = None,
                       **gen_kwargs) -> int:
    """Store a snapshot from recorded chat verdicts.

    Items are regenerated per probe with run_all (seed=0, same as
    audit/score_blinded.py); the replay adapter supplies the recorded
    verdicts. `probes` is an optional list of probe names (default: all
    9); `n` defaults to len(blinded)/len(probes) and must divide evenly.
    """
    blinded_path = Path(blinded_path)
    n_blinded = sum(1 for line in blinded_path.read_text(
        encoding="utf-8").splitlines() if line.strip())
    all_probes = discover_probes()
    if probes is not None:
        wanted = set(probes)
        selected = [p for p in all_probes if p["name"] in wanted]
        missing = wanted - {p["name"] for p in selected}
        if missing:
            raise ValueError(f"unknown probe(s): {sorted(missing)}")
    else:
        selected = all_probes
    if n is None:
        if n_blinded % len(selected) != 0:
            raise ValueError(
                f"{blinded_path}: {n_blinded} items is not divisible by "
                f"{len(selected)} probes; pass n explicitly")
        n = n_blinded // len(selected)
    adapter = replay_adapter_for(blinded_path, verdicts_path)
    return snapshot_battery(adapter, adapter_name, n=n, seed=seed,
                            label=label, db_path=db_path, probes=selected,
                            **gen_kwargs)
