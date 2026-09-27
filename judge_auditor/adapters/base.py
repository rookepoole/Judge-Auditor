"""Shared base class for judge adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import requests


class JudgeAdapterError(Exception):
    """Raised when a judge call fails in a way the caller can act on.

    Covers HTTP errors, timeouts, connection failures and unparseable
    responses. The message always names the adapter, the URL and what went
    wrong -- no bare stack trace for the runner to puzzle over.
    """


class JudgeAdapter(ABC):
    """Translate ``judge(system_prompt, user_prompt) -> str`` into HTTP."""

    name: str
    timeout: int = 60

    @abstractmethod
    def judge(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompts to the judge backend; return its raw text."""
        raise NotImplementedError

    # -- shared helpers -------------------------------------------------

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        """Append *path* to *base_url*, preserving any query string.

        The query string is how tests select mock behaviors
        (``?behavior=...``), so it must survive URL construction instead of
        being mangled into the middle of the path.
        """
        if "?" in base_url:
            base, _, qs = base_url.partition("?")
            return base.rstrip("/") + path + "?" + qs
        return base_url.rstrip("/") + path

    def _post(self, url: str, payload: dict, headers: dict, *, method: str = "POST") -> dict:
        """Send a JSON request; return the decoded JSON body or raise clearly."""
        label = self.name
        try:
            resp = requests.request(
                method, url, json=payload, headers=headers, timeout=self.timeout
            )
        except requests.exceptions.Timeout as exc:
            raise JudgeAdapterError(
                f"[{label}] timed out after {self.timeout}s: {method} {url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise JudgeAdapterError(
                f"[{label}] could not connect: {method} {url} ({exc})"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise JudgeAdapterError(
                f"[{label}] request failed: {method} {url} ({exc})"
            ) from exc
        if resp.status_code >= 400:
            raise JudgeAdapterError(
                f"[{label}] HTTP {resp.status_code} from {url}: {resp.text[:300]}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise JudgeAdapterError(
                f"[{label}] response was not JSON from {url}: {resp.text[:300]}"
            ) from exc
