"""Anthropic messages-API judge adapter."""

from __future__ import annotations

from .base import JudgeAdapter, JudgeAdapterError


class AnthropicAdapter(JudgeAdapter):
    """Judge via the Anthropic ``/v1/messages`` endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-haiku-4-5",
        timeout: int = 60,
    ):
        self.name = "anthropic"
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        url = self._join_url(self.base_url, "/v1/messages")
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 16,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        data = self._post(url, payload, headers)
        try:
            text = data["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeAdapterError(
                f"[{self.name}] unexpected response shape from {url}: "
                f"{str(data)[:300]}"
            ) from exc
        if not isinstance(text, str):
            raise JudgeAdapterError(
                f"[{self.name}] judge text was not a string from {url}: {text!r}"
            )
        return text
