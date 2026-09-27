"""Ollama judge adapter (OpenAI-compatible endpoint)."""

from __future__ import annotations

from .openai import OpenAIAdapter


class OllamaAdapter(OpenAIAdapter):
    """Judge via Ollama's OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3",
        timeout: int = 60,
    ):
        super().__init__(
            api_key=api_key, base_url=base_url, model=model, timeout=timeout
        )
        self.name = "ollama"
