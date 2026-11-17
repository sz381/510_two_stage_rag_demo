"""Thin DeepSeek client — the API is OpenAI-compatible."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from . import config


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce an answer."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def available() -> bool:
    return bool(config.DEEPSEEK_API_KEY)


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
    if not available():
        raise LLMError("DEEPSEEK_API_KEY is not set (add it to .env or Space secrets).")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    candidates = [config.DEEPSEEK_MODEL]
    if config.DEEPSEEK_MODEL != "deepseek-chat":
        candidates.append("deepseek-chat")  # fallback if the alias is unavailable

    last_error: Exception | None = None
    for model in candidates:
        try:
            response = _client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if model.lower() not in str(exc).lower():
                break  # auth/network problem: retrying another model will not help
    raise LLMError(f"DeepSeek request failed: {last_error}")
