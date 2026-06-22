from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class AdapterStats:
    calls_made: int = 0
    tokens_used: int = 0
    rate_limit_retries: int = 0


class BaseAdapter(ABC):
    """Common interface every LLM adapter must implement."""

    def __init__(self) -> None:
        self.stats = AdapterStats()

    @abstractmethod
    async def chat(self, messages: list[ChatMessage], model: str | None = None) -> str:
        """Send messages and return the assistant reply."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter identifier, e.g. groq/llama-3.3-70b."""
        ...
