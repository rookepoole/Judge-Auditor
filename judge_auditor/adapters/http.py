"""Fully configurable HTTP judge adapter."""

from __future__ import annotations

import json

from .base import JudgeAdapter, JudgeAdapterError

DEFAULT_REQUEST_TEMPLATE = (
    '{"model": "judge", "messages": ['
    '{"role": "system", "content": "{system}"}, '
    '{"role": "user", "content": "{prompt}"}], '
    '"max_tokens": 16, "temperature": 0}'
)

# Sentinels used so a prompt that literally contains "{prompt}" (or the
# system prompt containing "{system}") cannot corrupt the substitution.
_SYS_SENTINEL = "\x00SYSTEM\x00"
_PROMPT_SENTINEL = "\x00PROMPT\x00"


def _json_escape(text: str) -> str:
    """Escape *text* so it can be inlined into a JSON string literal."""
    return json.dumps(text)[1:-1]


class CustomHTTPAdapter(JudgeAdapter):
    """Judge via an arbitrary HTTP endpoint.

    ``request_template`` is a JSON string with ``{system}`` and ``{prompt}``
    placeholders; both are JSON-escaped before substitution so multi-line
    prompts (and quotes) render to valid JSON. ``response_path`` is a
    dot-separated path into the JSON response, with list indices written as
    integers, e.g. ``"choices.0.message.content"``.
    """

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: dict | None = None,
        request_template: str | None = None,
        response_path: str = "choices.0.message.content",
        timeout: int = 60,
    ):
        self.name = "custom_http"
        self.url = url
        self.method = method
        self.headers = dict(headers) if headers else {}
        self.request_template = (
            request_template if request_template is not None else DEFAULT_REQUEST_TEMPLATE
        )
        self.response_path = response_path
        self.timeout = timeout

    def _render(self, system_prompt: str, user_prompt: str) -> dict:
        body_str = self.request_template.replace("{system}", _SYS_SENTINEL)
        body_str = body_str.replace("{prompt}", _PROMPT_SENTINEL)
        body_str = body_str.replace(_SYS_SENTINEL, _json_escape(system_prompt))
        body_str = body_str.replace(_PROMPT_SENTINEL, _json_escape(user_prompt))
        try:
            return json.loads(body_str)
        except ValueError as exc:
            raise JudgeAdapterError(
                f"[{self.name}] request_template did not render to valid JSON: "
                f"{body_str[:300]}"
            ) from exc

    def _extract(self, data):
        current = data
        for part in self.response_path.split("."):
            if isinstance(current, list):
                try:
                    idx = int(part)
                except ValueError as exc:
                    raise JudgeAdapterError(
                        f"[{self.name}] response_path {self.response_path!r}: "
                        f"{part!r} is not an integer index into a list"
                    ) from exc
                try:
                    current = current[idx]
                except IndexError as exc:
                    raise JudgeAdapterError(
                        f"[{self.name}] response_path {self.response_path!r}: "
                        f"index {idx} out of range (list of length {len(current)})"
                    ) from exc
            elif isinstance(current, dict):
                if part not in current:
                    raise JudgeAdapterError(
                        f"[{self.name}] response_path {self.response_path!r}: "
                        f"key {part!r} missing; available keys: "
                        f"{sorted(current)[:10]}"
                    )
                current = current[part]
            else:
                raise JudgeAdapterError(
                    f"[{self.name}] response_path {self.response_path!r}: "
                    f"cannot descend into {type(current).__name__} at {part!r}"
                )
        if not isinstance(current, str):
            raise JudgeAdapterError(
                f"[{self.name}] value at response_path {self.response_path!r} "
                f"was not a string: {current!r}"
            )
        return current

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        headers.update(self.headers)
        payload = self._render(system_prompt, user_prompt)
        data = self._post(self.url, payload, headers, method=self.method)
        return self._extract(data)
