"""Lightweight client wrappers for interacting with chat-based LLM APIs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


class LLMClientError(RuntimeError):
    """Raised when the LLM backend returns an error or cannot be reached."""


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient:
    """Minimal OpenAI-compatible chat completion client."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        default_temperature: float = 0.1,
    ) -> None:
        self.base_url = base_url or os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.timeout = timeout
        self.default_temperature = default_temperature

        if self.api_key is None:
            raise LLMClientError(
                "API key not provided. Set LLM_API_KEY environment variable or pass api_key explicitly."
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def create_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        payload.update(extra)

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=self._headers(), json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            raise LLMClientError(f"Failed to reach LLM backend: {exc}") from exc

        data = response.json()
        if "choices" not in data:
            raise LLMClientError(f"Unexpected response payload: {json.dumps(data)[:200]}")
        return data

    def extract_message(self, response: Dict[str, Any]) -> str:
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Malformed response: {response}") from exc


class EchoLLMClient(LLMClient):
    """Offline fallback that echoes the prompt for deterministic testing."""

    def __init__(self, *_, **__):  # type: ignore[no-untyped-def]
        self.base_url = ""
        self.api_key = ""
        self.model = "echo"
        self.timeout = 0.0
        self.default_temperature = 0.0

    def create_chat_completion(self, *, system_prompt: str, user_prompt: str, **extra: Any) -> Dict[str, Any]:
        pseudo_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "final_diagnosis": "undetermined",
                                "confidence": 0,
                                "rationale": ["Echo backend: review manually."],
                                "checks": ["Re-run measurement"],
                                "maintenance": ["Schedule manual inspection"],
                                "system_prompt": system_prompt,
                                "user_prompt": user_prompt,
                            },
                            ensure_ascii=False,
                        ),
                    }
                }
            ]
        }
        return pseudo_response

    def extract_message(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]


__all__ = ["LLMClient", "LLMClientError", "ChatMessage", "EchoLLMClient"]
