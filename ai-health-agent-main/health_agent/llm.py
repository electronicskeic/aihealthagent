from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str


def get_llm_config() -> LLMConfig | None:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    return LLMConfig(api_key=api_key, model=model)


def chat_with_llm(*, system_prompt: str, messages: list[dict[str, Any]], model: str) -> str:
    """
    Uses the OpenAI Python SDK if available/configured.

    This is intentionally defensive because environments vary (openai v0 vs v1+).
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # Prefer OpenAI v1+ client.
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        pass

    # Fallback to older openai.ChatCompletion API (legacy).
    import openai  # type: ignore

    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=0.4,
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()

