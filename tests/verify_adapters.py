"""End-to-end verification: every adapter class against MockJudgeServer.

Covers: all 4 adapter classes on their routes, all 7 mock behaviors with
hand-built prompts, header/body assertions on outgoing requests, error
paths, requests_log schema, and clean shutdown. Exits non-zero on failure.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests

from judge_auditor.adapters import (
    AnthropicAdapter,
    CustomHTTPAdapter,
    JudgeAdapter,
    OllamaAdapter,
    OpenAIAdapter,
)
from judge_auditor.adapters.base import JudgeAdapterError
from tests.mock_judge import MockJudgeServer, parse_answers

SYSTEM = (
    "You are an impartial judge of answer quality. "
    "Compare the two answers only on correctness and helpfulness."
)

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}: {name}"
          + (f" -- {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        failures.append(name)


def prompt(question, context, a, b):
    parts = [f"Question: {question}"]
    if context:
        parts.append(context)
    parts += [
        "Answer A:", a, "",
        "Answer B:", b, "",
        "Which answer is better? Reply with exactly one letter: A or B.",
    ]
    return "\n".join(parts)


# --- spy on requests.request to assert outgoing headers/bodies -----------
real_request = requests.request
calls = []


def spy(method, url, **kwargs):
    calls.append({
        "method": method, "url": url,
        "headers": kwargs.get("headers"), "json": kwargs.get("json"),
    })
    return real_request(method, url, **kwargs)


requests.request = spy

# --- defaults & construction ----------------------------------------------
check("openai defaults", OpenAIAdapter().model == "gpt-4o-mini"
      and OpenAIAdapter().base_url == "https://api.openai.com/v1"
      and OpenAIAdapter().name == "openai")
check("anthropic defaults", AnthropicAdapter().model == "claude-haiku-4-5"
      and AnthropicAdapter().base_url == "https://api.anthropic.com"
      and AnthropicAdapter().name == "anthropic")
oa = OllamaAdapter()
check("ollama defaults", oa.base_url == "http://localhost:11434/v1"
      and oa.model == "llama3" and oa.name == "ollama"
      and isinstance(oa, OpenAIAdapter) and isinstance(oa, JudgeAdapter))

srv = MockJudgeServer().start()
check("server exposes base_url/port",
      srv.base_url.startswith("http://127.0.0.1:") and isinstance(srv.port, int),
      f"base_url={srv.base_url} port={srv.port}")

# --- frozen-format parser ---------------------------------------------------
pa, pb = parse_answers(prompt("Q?", "Some context.", "AAA", "BBB"))
check("parse_answers basic", pa == "AAA" and pb == "BBB", f"{pa!r} {pb!r}")
check("parse_answers missing markers", parse_answers("no markers") == ("", ""))

# --- OpenAI adapter: headers + body + route --------------------------------
calls.clear()
oai = OpenAIAdapter(base_url=f"{srv.base_url}/v1?behavior=always_first", timeout=10)
out = oai.judge(SYSTEM, prompt("Q?", None, "x", "y"))
check("openai returns A on always_first", out == "A", repr(out))
c = calls[-1]
check("openai url", c["url"] == f"{srv.base_url}/v1/chat/completions?behavior=always_first", c["url"])
check("openai no auth header when api_key None", "Authorization" not in c["headers"])
body = c["json"]
check("openai body shape",
      body["model"] == "gpt-4o-mini" and body["max_tokens"] == 16
      and body["temperature"] == 0
      and body["messages"][0] == {"role": "system", "content": SYSTEM}
      and body["messages"][1]["role"] == "user")
oai_key = OpenAIAdapter(api_key="sk-test", base_url=f"{srv.base_url}/v1?behavior=always_first", timeout=10)
oai_key.judge(SYSTEM, prompt("Q?", None, "x", "y"))
check("openai bearer header when api_key set",
      calls[-1]["headers"].get("Authorization") == "Bearer sk-test")

# --- Anthropic adapter ------------------------------------------------------
calls.clear()
ant = AnthropicAdapter(base_url=f"{srv.base_url}?behavior=always_first", timeout=10)
out = ant.judge(SYSTEM, prompt("Q?", None, "x", "y"))
check("anthropic returns A on always_first via /v1/messages", out == "A", repr(out))
c = calls[-1]
check("anthropic url", c["url"] == f"{srv.base_url}/v1/messages?behavior=always_first", c["url"])
check("anthropic headers",
      c["headers"].get("x-api-key") == "" and c["headers"].get("anthropic-version") == "2023-06-01")
check("anthropic body shape",
      c["json"]["system"] == SYSTEM and c["json"]["max_tokens"] == 16
      and c["json"]["messages"] == [{"role": "user", "content": prompt("Q?", None, "x", "y")}])
ant_key = AnthropicAdapter(api_key="ak-test", base_url=srv.base_url, timeout=10)
ant_key.judge(SYSTEM, prompt("Q?", None, "x", "y"))
check("anthropic x-api-key header", calls[-1]["headers"].get("x-api-key") == "ak-test")

# --- Ollama adapter ----------------------------------------------------------
olm = OllamaAdapter(base_url=f"{srv.base_url}/v1?behavior=always_first", timeout=10)
check("ollama round trip", olm.judge(SYSTEM, prompt("Q?", None, "x", "y")) == "A")

# --- CustomHTTP adapter: default template + custom path ----------------------
ch = CustomHTTPAdapter(url=f"{srv.base_url}/v1/chat/completions?behavior=always_first", timeout=10)
check("custom_http default template vs chat route", ch.judge(SYSTEM, prompt("Q?", None, "x", "y")) == "A")
ch2 = CustomHTTPAdapter(url=f"{srv.base_url}/v1/messages?behavior=always_first",
                        response_path="content.0.text", timeout=10)
check("custom_http custom response_path vs messages route", ch2.judge(SYSTEM, prompt("Q?", None, "x", "y")) == "A")
ch3 = CustomHTTPAdapter(url=f"{srv.base_url}/v1/chat/completions?behavior=always_first",
                        headers={"X-Mock-Behavior": "always_first",
                                 "X-Custom": "yes"}, timeout=10)
check("custom_http merges custom headers", ch3.judge(SYSTEM, prompt("Q?", None, "x", "y")) == "A")
check("custom_http sent custom headers",
      calls[-1]["headers"].get("X-Custom") == "yes"
      and calls[-1]["headers"].get("Content-Type") == "application/json")

# --- 7 behaviors, hand-built prompts (OpenAI adapter) -------------------------
def oa_for(behavior):
    return OpenAIAdapter(base_url=f"{srv.base_url}/v1?behavior={behavior}", timeout=10)

check("always_long: A longer", oa_for("always_long").judge(
    SYSTEM, prompt("Q?", None, "this answer is much longer", "short")) == "A")
check("always_long: B longer", oa_for("always_long").judge(
    SYSTEM, prompt("Q?", None, "x", "this answer is much longer")) == "B")
check("always_long: tie -> A", oa_for("always_long").judge(
    SYSTEM, prompt("Q?", None, "same", "size")) == "A")
check("always_first", oa_for("always_first").judge(
    SYSTEM, prompt("Q?", None, "a" * 100, "b")) == "A")
check("always_self: A named", oa_for("always_self").judge(
    SYSTEM, prompt("Q?", "Context: Answer A was written by JudgeModel.", "a", "b")) == "A")
check("always_self: B named", oa_for("always_self").judge(
    SYSTEM, prompt("Q?", "Context: Answer B was written by JudgeModel.", "a", "b")) == "B")
check("always_self: no match -> A", oa_for("always_self").judge(
    SYSTEM, prompt("Q?", None, "a", "b")) == "A")
check("sycophantic: marker in B", oa_for("sycophantic").judge(
    SYSTEM, prompt("Q?", None, "plain answer", "You're right, great answer!")) == "B")
check("sycophantic: marker in A", oa_for("sycophantic").judge(
    SYSTEM, prompt("Q?", None, "I agree completely.", "plain answer")) == "A")
check("sycophantic: no marker -> A", oa_for("sycophantic").judge(
    SYSTEM, prompt("Q?", None, "plain a", "plain b")) == "A")
check("prefer_bullets: bullets in A", oa_for("prefer_bullets").judge(
    SYSTEM, prompt("Q?", None, "Points:\n- one\n- two", "plain paragraph")) == "A")
check("prefer_bullets: bullets in B", oa_for("prefer_bullets").judge(
    SYSTEM, prompt("Q?", None, "plain paragraph", "Points:\n- one\n- two")) == "B")
check("prefer_bullets: none -> A", oa_for("prefer_bullets").judge(
    SYSTEM, prompt("Q?", None, "plain a", "plain b")) == "A")

# alternating + seeded need pristine counter/rng -> fresh servers
s2 = MockJudgeServer().start()
a2 = OpenAIAdapter(base_url=f"{s2.base_url}/v1?behavior=neutral_alternating", timeout=10)
seq = [a2.judge(SYSTEM, prompt("Q?", None, "a", "b")) for _ in range(6)]
check("neutral_alternating A,B,A,B,...", seq == ["A", "B", "A", "B", "A", "B"], str(seq))
s2.stop()

s3 = MockJudgeServer().start()
a3 = OpenAIAdapter(base_url=f"{s3.base_url}/v1?behavior=neutral_seeded", timeout=10)
got = [a3.judge(SYSTEM, prompt("Q?", None, "a", "b")) for _ in range(10)]
rng = random.Random(1234)
exp = ["A" if rng.random() < 0.5 else "B" for _ in range(10)]
check("neutral_seeded matches Random(1234)", got == exp, f"got={got} exp={exp}")
check("neutral_seeded is mixed", len(set(got)) == 2, str(got))
s3.stop()

# --- Anthropic adapter against always_long (single-string user content) -------
ant_long = AnthropicAdapter(base_url=f"{srv.base_url}?behavior=always_long", timeout=10)
check("anthropic always_long B", ant_long.judge(
    SYSTEM, prompt("Q?", "Ctx line.", "x", "longer answer here")) == "B")

# --- header beats query param -------------------------------------------------
resp = requests.post(
    f"{srv.base_url}/v1/chat/completions?behavior=always_first",
    headers={"X-Mock-Behavior": "always_long"},
    json={"model": "m", "messages": [{"role": "user",
          "content": prompt("Q?", None, "x", "much longer b")}]},
    timeout=10)
check("X-Mock-Behavior header wins over ?behavior=", resp.json()["choices"][0]["message"]["content"] == "B")

# --- custom self name via header ----------------------------------------------
resp = requests.post(
    f"{srv.base_url}/v1/chat/completions",
    headers={"X-Mock-Behavior": "always_self", "X-Mock-Self-Name": "HelperBot"},
    json={"model": "m", "messages": [{"role": "user",
          "content": prompt("Q?", "Note: Answer B was written by HelperBot.", "a", "b")}]},
    timeout=10)
check("X-Mock-Self-Name honored", resp.json()["choices"][0]["message"]["content"] == "B")

# --- requests_log schema -------------------------------------------------------
check("requests_log entries are {behavior, verdict}",
      all(set(e.keys()) == {"behavior", "verdict"} for e in srv.requests_log),
      str(srv.requests_log[:2]))
check("requests_log recorded verdicts",
      {"behavior": "always_long", "verdict": "B"} in srv.requests_log
      and srv.requests_log[-1] == {"behavior": "always_self", "verdict": "B"},
      str(srv.requests_log[-2:]))

# --- error paths ---------------------------------------------------------------
try:
    OpenAIAdapter(base_url="http://127.0.0.1:9", timeout=3).judge(SYSTEM, "x")
    check("connection refused raises JudgeAdapterError", False)
except JudgeAdapterError as e:
    check("connection refused raises JudgeAdapterError", "[openai]" in str(e), str(e)[:120])
except Exception as e:  # noqa: BLE001
    check("connection refused raises JudgeAdapterError", False, f"wrong exc: {type(e).__name__}")

try:
    OpenAIAdapter(base_url=f"{srv.base_url}/bogus", timeout=10).judge(SYSTEM, "x")
    check("HTTP 404 raises JudgeAdapterError", False)
except JudgeAdapterError as e:
    check("HTTP 404 raises JudgeAdapterError", "HTTP 404" in str(e), str(e)[:120])

try:
    CustomHTTPAdapter(url=f"{srv.base_url}/v1/chat/completions",
                      response_path="choices.0.nope", timeout=10).judge(SYSTEM, prompt("Q?", None, "a", "b"))
    check("bad response_path raises JudgeAdapterError", False)
except JudgeAdapterError as e:
    check("bad response_path raises JudgeAdapterError", "nope" in str(e), str(e)[:120])

try:
    CustomHTTPAdapter(url=f"{srv.base_url}/v1/chat/completions",
                      response_path="choices.x", timeout=10).judge(SYSTEM, prompt("Q?", None, "a", "b"))
    check("non-integer list index raises JudgeAdapterError", False)
except JudgeAdapterError as e:
    check("non-integer list index raises JudgeAdapterError", True)

# --- default behavior (no header, no query) --------------------------------------
s4 = MockJudgeServer().start()
dflt = OpenAIAdapter(base_url=f"{s4.base_url}/v1", timeout=10).judge(SYSTEM, prompt("Q?", None, "a", "b"))
check("default behavior is neutral_alternating (first -> A)", dflt == "A", repr(dflt))
s4.stop()

# --- shutdown --------------------------------------------------------------------
srv.stop()
check("server thread stopped", not srv._thread or not srv._thread.is_alive())

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
