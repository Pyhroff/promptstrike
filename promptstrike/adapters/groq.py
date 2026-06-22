import asyncio
import re

from groq import AsyncGroq, RateLimitError

from .base import BaseAdapter, ChatMessage


class GroqAdapter(BaseAdapter):
    """Async Groq adapter with exponential-backoff retry on 429s."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        super().__init__()
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"groq/{self._model}"

    async def chat(self, messages: list[ChatMessage], model: str | None = None) -> str:
        target_model = model or self._model
        payload = [{"role": m.role, "content": m.content} for m in messages]

        for attempt in range(4):
            try:
                resp = await self._client.chat.completions.create(
                    model=target_model,
                    messages=payload,
                    temperature=1.0,
                    max_tokens=1024,
                )
                self.stats.calls_made += 1
                usage = resp.usage
                if usage:
                    self.stats.tokens_used += usage.total_tokens
                return resp.choices[0].message.content or ""

            except RateLimitError:
                self.stats.rate_limit_retries += 1
                if attempt == 3:
                    raise
                wait = 5 * (2 ** attempt)   # 5s → 10s → 20s → give up
                await asyncio.sleep(wait)

        return ""   # unreachable, satisfies type checker


def adapter_from_model_string(model_str: str, api_key: str) -> GroqAdapter:
    """Parse 'groq/llama-3.3-70b-versatile' → GroqAdapter."""
    parts = model_str.split("/", 1)
    model = parts[1] if len(parts) == 2 else parts[0]
    return GroqAdapter(api_key=api_key, model=model)
