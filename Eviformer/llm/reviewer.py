"""Coordinator that runs the LLM review loop and parses responses."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .client import EchoLLMClient, LLMClient, LLMClientError
from .prompts import build_review_prompt


@dataclass
class ReviewVerdict:
    final_diagnosis: str
    confidence: float
    rationale: Any
    checks: Any
    maintenance: Any
    raw_response: str
    conflict_with_primary: bool


def _load_json_fragment(payload: str) -> Dict[str, Any]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(payload[start : end + 1])


def run_llm_review(
    feature_packet: Dict[str, Any],
    primary_decision: Dict[str, Any],
    *,
    client: Optional[LLMClient] = None,
    language: str = "zh",
    temperature: float = 0.2,
) -> ReviewVerdict:
    """Execute the LLM review and return a structured verdict."""

    system_prompt, user_prompt = build_review_prompt(feature_packet, primary_decision, language=language)
    client = client or EchoLLMClient()

    try:
        response = client.create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        message = client.extract_message(response)
    except LLMClientError as exc:  # pragma: no cover - network failure path
        raise

    try:
        parsed = _load_json_fragment(message)
    except json.JSONDecodeError:
        parsed = {
            "final_diagnosis": "undetermined",
            "confidence": 0.0,
            "rationale": ["Failed to parse LLM response."],
            "checks": ["Review manually"],
            "maintenance": [],
        }

    final_diagnosis = str(parsed.get("final_diagnosis", "undetermined"))
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    primary_label = str(primary_decision.get("predicted_label"))
    conflict = final_diagnosis.lower() != primary_label.lower()

    verdict = ReviewVerdict(
        final_diagnosis=final_diagnosis,
        confidence=confidence,
        rationale=parsed.get("rationale"),
        checks=parsed.get("checks"),
        maintenance=parsed.get("maintenance"),
        raw_response=message,
        conflict_with_primary=conflict,
    )
    return verdict


__all__ = ["run_llm_review", "ReviewVerdict"]
