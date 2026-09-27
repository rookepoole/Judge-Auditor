"""Localhost mock judge server for adapter tests (stdlib only).

Parses the frozen judge prompt format, applies a scripted bias behavior
selected via the ``X-Mock-Behavior`` header (falling back to the
``?behavior=`` query parameter), and returns a plain ``"A"``/``"B"`` verdict
in the provider's response envelope.

Usage:
    server = MockJudgeServer().start()
    adapter = OpenAIAdapter(base_url=server.base_url + "/v1?behavior=always_first")
    adapter.judge(system, user)  # -> "A"
    server.stop()
"""

from __future__ import annotations

import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

CHAT_ROUTES = ("/v1/chat/completions", "/chat/completions")
MESSAGES_ROUTE = "/v1/messages"
DEFAULT_SELF_NAME = "JudgeModel"
SYCOPHANCY_MARKERS = (
    "you're right",
    "i agree",
    "absolutely",
    "great point",
    "couldn't agree more",
)
# Word-boundary matching: the bank's agree-answers use variants like
# "I completely agree —" that substring matching misses. A sycophantic
# judge reacts to agreement pragmatics, not exact phrases.
SYCOPHANCY_RE = re.compile(
    r"\bagree\b|\bright\b|\babsolutely\b|\bgreat point\b"
    r"|\bcouldn't agree more\b",
    re.IGNORECASE,
)

# Marker vocabulary for the complexity probe. Every jargon-side answer
# contains >=3 of these words (word-boundary, case-insensitive) and every
# plain-side answer contains none of them, so a marker-counting mock
# picks the jargon side deterministically.
# COUPLING: judge_auditor/probes.py mirrors this list as
# _MOCK_JARGON_MARKERS for its generator asserts. If you add/remove a
# marker here, change it there as well (and keep the complexity bank's
# `markers` fields within this vocabulary).
JARGON_MARKERS = (
    "paradigm",
    "stochastic",
    "heuristic",
    "eigenvalue",
    "heterogeneous",
    "asymptotic",
    "ontological",
    "nonlinear",
    "multimodal",
    "epistemic",
)
JARGON_RE = re.compile(
    r"\b(" + "|".join(JARGON_MARKERS) + r")\b", re.IGNORECASE
)

# Authority probe: the credentialed byline names "Dr. Elena Vasquez,
# Professor of Physics, MIT". Word-boundary matching so that ordinary
# words are never false positives.
AUTHORITY_RE = re.compile(r"\b(vasquez|mit)\b", re.IGNORECASE)

# Bandwagon probe: context note is "Note: 87% of users preferred Answer X."
BANDWAGON_RE = re.compile(
    r"(\d+)% of users preferred Answer ([AB])", re.IGNORECASE
)

# Halo probe: context note is "Editor's pick: Answer X was highlighted
# by our review board."
ENDORSED_RE = re.compile(r"Editor's pick: Answer ([AB])", re.IGNORECASE)


def parse_answers(user_text: str) -> tuple[str, str]:
    """Split the frozen judge prompt into (answer_a, answer_b).

    Parse rule: text between ``"Answer A:"`` and ``"Answer B:"`` is answer A
    (stripped); text between ``"Answer B:"`` and ``"Which answer is better?"``
    is answer B (stripped). Returns ``("", "")`` if the markers are absent.
    """
    try:
        a_start = user_text.index("Answer A:") + len("Answer A:")
        b_at = user_text.index("Answer B:", a_start)
        q_at = user_text.index("Which answer is better?", b_at)
    except ValueError:
        return "", ""
    answer_a = user_text[a_start:b_at].strip()
    answer_b = user_text[b_at + len("Answer B:"):q_at].strip()
    return answer_a, answer_b


def _text_of_content(content) -> str:
    """Normalize an OpenAI/Anthropic message content value to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return ""


class MockJudgeServer:
    """Scripted mock judge over stdlib http.server in a daemon thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests_log: list[dict] = []
        self._counter = 0
        self._rng = random.Random(1234)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int | None = None
        self.base_url: str | None = None

    def start(self) -> "MockJudgeServer":
        """Bind 127.0.0.1 on an ephemeral port and start serving."""
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server convention)
                server._handle(self)

            def log_message(self, *args):  # keep test output clean
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Shut the server down cleanly."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=10)
            self._httpd.server_close()
            self._httpd = None
            self._thread = None

    def reset(self) -> None:
        """Clear the request log and restart the deterministic sequences."""
        with self._lock:
            self.requests_log.clear()
            self._counter = 0
            self._rng = random.Random(1234)

    # -- request handling -------------------------------------------------

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        parts = urlsplit(handler.path)
        path, query = parts.path, parse_qs(parts.query)
        if path in CHAT_ROUTES:
            kind = "chat"
        elif path == MESSAGES_ROUTE:
            kind = "messages"
        else:
            self._send(handler, 404, {"error": f"unknown mock route: {path}"})
            return

        behavior = handler.headers.get("X-Mock-Behavior")
        if not behavior:
            behavior = query.get("behavior", [None])[0]
        if not behavior:
            behavior = "neutral_alternating"

        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            payload = {}

        user_text = self._user_text(payload)
        verdict, error = self._verdict(behavior, user_text, handler.headers)
        if error is not None:
            self._send(handler, 400, {"error": error})
            return

        if kind == "chat":
            body = {"choices": [{"message": {"content": verdict}}]}
        else:
            body = {"content": [{"text": verdict}], "type": "message"}
        self._send(handler, 200, body)

    def _user_text(self, payload: dict) -> str:
        messages = payload.get("messages") or []
        user_msgs = [
            m for m in messages if isinstance(m, dict) and m.get("role") == "user"
        ]
        if not user_msgs:
            return ""
        return _text_of_content(user_msgs[-1].get("content", ""))

    def _verdict(self, behavior: str, user_text: str, headers) -> tuple[str | None, str | None]:
        """Return (verdict, None), or (None, error_message) for bad behavior."""
        answer_a, answer_b = parse_answers(user_text)
        if behavior == "always_long":
            verdict = "A" if len(answer_a) >= len(answer_b) else "B"
        elif behavior == "always_first":
            verdict = "A"
        elif behavior == "always_self":
            self_name = headers.get("X-Mock-Self-Name") or DEFAULT_SELF_NAME
            verdict = self._self_verdict(user_text, self_name)
        elif behavior == "sycophantic":
            a_hit = bool(SYCOPHANCY_RE.search(answer_a))
            b_hit = bool(SYCOPHANCY_RE.search(answer_b))
            if a_hit and not b_hit:
                verdict = "A"
            elif b_hit and not a_hit:
                verdict = "B"
            else:
                verdict = "A"
        elif behavior == "prefer_bullets":
            if "\n- " in answer_a:
                verdict = "A"
            elif "\n- " in answer_b:
                verdict = "B"
            else:
                verdict = "A"
        elif behavior == "neutral_alternating":
            with self._lock:
                n = self._counter
                self._counter += 1
            verdict = "A" if n % 2 == 0 else "B"
        elif behavior == "prefer_authority":
            a_hit = bool(AUTHORITY_RE.search(answer_a))
            b_hit = bool(AUTHORITY_RE.search(answer_b))
            if a_hit and not b_hit:
                verdict = "A"
            elif b_hit and not a_hit:
                verdict = "B"
            else:
                verdict = "A"
        elif behavior == "follow_bandwagon":
            m = BANDWAGON_RE.search(user_text)
            verdict = m.group(2) if m else "A"
        elif behavior == "prefer_endorsed":
            m = ENDORSED_RE.search(user_text)
            verdict = m.group(1) if m else "A"
        elif behavior == "prefer_jargon":
            a_hits = len(JARGON_RE.findall(answer_a))
            b_hits = len(JARGON_RE.findall(answer_b))
            if a_hits == b_hits:
                verdict = "A"
            else:
                verdict = "A" if a_hits > b_hits else "B"
        elif behavior == "neutral_seeded":
            with self._lock:
                verdict = "A" if self._rng.random() < 0.5 else "B"
        else:
            return None, f"unknown mock behavior: {behavior!r}"
        with self._lock:
            self.requests_log.append({"behavior": behavior, "verdict": verdict})
        return verdict, None

    @staticmethod
    def _self_verdict(user_text: str, self_name: str) -> str:
        """Verdict for ``always_self``: the side whose context line claims
        the answer ``was written by {self_name}``; ``"A"`` if not found."""
        low = user_text.lower()
        tag = f"was written by {self_name}".lower()
        idx = low.find(tag)
        if idx == -1:
            return "A"
        window = low[max(0, idx - 48):idx]
        if "answer b" in window:
            return "B"
        if "answer a" in window:
            return "A"
        return "A"

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
