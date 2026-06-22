import httpx

from .base import BaseAdapter, ChatMessage


class OllamaAdapter(BaseAdapter):
    """Adapter for locally running Ollama models (no API key needed)."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__()
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    async def chat(self, messages: list[ChatMessage], model: str | None = None) -> str:
        target_model = model or self._model
        payload = {
            "model": target_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            self.stats.calls_made += 1
            return data["message"]["content"]
