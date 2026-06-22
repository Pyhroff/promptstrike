import asyncio

from openai import AsyncOpenAI, RateLimitError

from .base import BaseAdapter, ChatMessage


class OpenAIAdapter(BaseAdapter):
    """Async OpenAI adapter — works with any OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

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
                if resp.usage:
                    self.stats.tokens_used += resp.usage.total_tokens
                return resp.choices[0].message.content or ""

            except RateLimitError:
                self.stats.rate_limit_retries += 1
                if attempt == 3:
                    raise
                await asyncio.sleep(5 * (2 ** attempt))

        return ""
