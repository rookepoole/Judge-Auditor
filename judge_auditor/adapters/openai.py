"""OpenAI-compatible chat-completions judge adapter."""

from __future__ import annotations

from .base import JudgeAdapter, JudgeAdapterError


class OpenAIAdapter(JudgeAdapter):
    """Judge via an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 60,
    ):
        self.name = "openai"
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        url = self._join_url(self.base_url, "/chat/completions")
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 16,
            "temperature": 0,
        }
        data = self._post(url, payload, headers)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeAdapterError(
                f"[{self.name}] unexpected response shape from {url}: "
                f"{str(data)[:300]}"
            ) from exc
        if not isinstance(content, str):
            raise JudgeAdapterError(
                f"[{self.name}] judge content was not a string from {url}: {content!r}"
            )
        return content
