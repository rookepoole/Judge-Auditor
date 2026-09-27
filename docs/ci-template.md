# Judge Auditor CI templates

Drop-in GitHub Actions workflows that use Judge Auditor as a CI gate
against **your** judge. All commands below reference the real CLI
(`judge_auditor/cli.py`). Replace every `{{PLACEHOLDER}}` with your own
values — nothing here works until they are filled in.

CLI facts these templates rely on:

- `python -m judge_auditor run --adapter ... --out-dir judge-report`
  writes `report.json` + `report.md` and **exits 0 on PASS, 1 otherwise**.
  The gate is the exit code.
- Adapters: `openai`, `anthropic`, `ollama`, `custom`. Options:
  `--base-url`, `--url`, `--model`, `--api-key`,
  `--header "Key:Value"` (repeatable, custom adapter only).

## Template A — Gate a PR on judge health

Runs the full 9-probe battery against your judge; the build fails when
the grade isn't PASS. Pin a judge version you trust and keep it in this
workflow.

```yaml
name: judge-gate

on:
  push:
  pull_request:

jobs:
  judge-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Judge Auditor
        run: |
          pip install pyyaml requests
          {{YOUR_CHECKOUT_OR_PIP_INSTALL}}  # e.g. git clone/pull of the judge-auditor repo, or pip install judge-auditor

      - name: Run probe battery against our judge
        env:
          JUDGE_API_KEY: ${{ secrets.JUDGE_API_KEY }}
        run: |
          python -m judge_auditor run \
            --adapter {{YOUR_ADAPTER}} \          # openai | anthropic | ollama | custom
            --base-url {{YOUR_BASE_URL}} \        # e.g. https://your-judge.example.com/v1
            --model {{YOUR_MODEL}} \             # e.g. your-judge-v3
            --api-key "$JUDGE_API_KEY" \         # or --header "Authorization: Bearer $TOKEN" for custom
            --out-dir judge-report
          # Exit code is the gate: 0 = PASS, 1 = NEEDS ATTENTION / FAIL / INCONCLUSIVE.

      - name: Upload judge report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: judge-report
          path: judge-report/
```

If you want the failure message to say exactly which probes tripped,
add an explicit assertion step after the run (the report is JSON):

```yaml
      - name: Assert PASS grade
        run: |
          python - <<'EOF'
          import json, sys
          grade = json.load(open("judge-report/report.json"))["grade"]
          print("grade:", grade)
          sys.exit(0 if grade == "PASS" else 1)
          EOF
```

## Template B — Scheduled drift audit (weekly)

Re-runs the battery on a cron schedule, snapshots the result into
`drift.db`, uploads the database as a workflow artifact, and compares
against the pinned baseline with the drift report.

> The `drift-snapshot`, `drift-baseline`, and `drift-report`
> subcommands ship with the Phase 3 monitor slice (see BUILD_PLAN.md).
> Until it lands, Templates B/C are a preview — pin the repo ref whose
> `cli.py` actually has them.

```yaml
name: judge-drift-audit

on:
  schedule:
    - cron: '0 9 * * 1'   # every Monday, 09:00 UTC
  workflow_dispatch:        # plus a manual "Run workflow" button

jobs:
  drift-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Judge Auditor
        run: |
          pip install pyyaml requests
          {{YOUR_CHECKOUT_OR_PIP_INSTALL}}

      - name: Weekly battery run + snapshot
        env:
          JUDGE_API_KEY: ${{ secrets.JUDGE_API_KEY }}
        run: |
          python -m judge_auditor run \
            --adapter {{YOUR_ADAPTER}} \
            --base-url {{YOUR_BASE_URL}} \
            --model {{YOUR_MODEL}} \
            --api-key "$JUDGE_API_KEY" \
            --out-dir judge-report
          python -m judge_auditor drift-snapshot \
            --run judge-report/report.json \
            --db drift.db \
            --judge-id {{YOUR_JUDGE_ID}}        # e.g. your-judge-v3
          # Note: run exits 1 on non-PASS, so the scheduled job alerts
          # on grade changes even before drift analysis.

      - name: Drift report vs pinned baseline
        run: |
          python -m judge_auditor drift-report \
            --db drift.db \
            --baseline {{YOUR_BASELINE_LABEL}} \  # pinned with drift-baseline
            --out drift-report.md
          cat drift-report.md

      - name: Upload drift database
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: drift-db
          path: drift.db
```

Pin the baseline once, from a run you trust, then re-pin deliberately
when the judge changes (re-pinning without cause hides regressions):

```bash
python -m judge_auditor drift-baseline --db drift.db --label good-baseline --run judge-report/report.json
```

## Template C — Mock gate (no secrets at all)

Want the CI hygiene without touching a live judge? Point the gate at a
local mock with a planted bias, assert the battery trips it, and assert
a neutral mock stays clean. This is the offline version of the repo's
own trip matrix, as a reusable step:

```yaml
      - name: Mock gate (planted bias must trip, neutral must not)
        run: |
          python tests/start_mock.py --behavior verbosity-biased --port 18091 &  # {{YOUR_MOCK_LAUNCHER}}
          sleep 1
          python -m judge_auditor run --adapter custom --url http://localhost:18091 --out-dir planted-report
          # run exits 1 on non-PASS — here we REQUIRE non-PASS:
          test ! -f planted-report/report.json && exit 1
          python - <<'EOF'
          import json, sys
          trips = [p for p, r in json.load(open("planted-report/report.json"))["probes"].items() if r["tripped"]]
          sys.exit(0 if "verbosity" in trips else 1)
          EOF
```

## Where secrets go (and where they don't)

| Step | Needs a secret? |
|---|---|
| Unit tests / trip matrix (`python -m unittest discover -s tests`) | **No** — stdlib mock judges on localhost. |
| Drift falsifier demo (`python audit/demo_drift.py`) | **No** — self-contained offline demo. |
| Protocol freeze / verify (`protocol-freeze`, `protocol-verify`) | **No** — sha256 hashing, fully local. |
| Template A/B gate against **your real judge** | **Yes** — your judge's API key via **GitHub Secrets** (`Settings → Secrets and variables → Actions`), injected as an env var and referenced as `${{ secrets.JUDGE_API_KEY }}`. Never hardcode keys in YAML, never echo them in logs. |
| Template C mock gate | **No** — same offline mocks as the repo suite. |

The rule of thumb: measurement machinery (tests, mocks, protocols,
drift math) runs offline with zero secrets; only the step that talks
to your production judge gets a key, and it gets it from GitHub
Secrets, never from the workflow file.
