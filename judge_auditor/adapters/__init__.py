"""Judge adapters: translate judge(system, user) -> str into provider HTTP calls."""

from .base import JudgeAdapter, JudgeAdapterError
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter
from .ollama import OllamaAdapter
from .http import CustomHTTPAdapter

__all__ = [
    "JudgeAdapter",
    "JudgeAdapterError",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
    "CustomHTTPAdapter",
]
