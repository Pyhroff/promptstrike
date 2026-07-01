"""
ShieldedAdapter — composes input scanning and output classification around any
BaseAdapter. Acts as a transparent proxy: callers use the same chat() interface,
but the shield intercepts before and after the model call.

Usage:
    from promptstrike.adapters.groq import GroqAdapter
    from promptstrike.defense import ShieldedAdapter

    inner = GroqAdapter(api_key=..., model="llama-3.3-70b-versatile")
    shielded = ShieldedAdapter(inner, scan_input=True, classify_output=True)

    # Raises ShieldBlockedError if the input is suspicious (above threshold)
    response = await shielded.chat(messages)
    # response.classified tells you if the output was flagged as jailbroken
"""
from __future__ import annotations

from dataclasses import dataclass

from promptstrike.adapters.base import BaseAdapter, ChatMessage
from promptstrike.defense.scanner import ScanResult, scan_input
from promptstrike.defense.classifier import ClassifyResult, classify_output


class ShieldBlockedError(Exception):
    """Raised when the input scanner blocks a request."""

    def __init__(self, result: ScanResult) -> None:
        self.scan_result = result
        super().__init__(
            f"Input blocked: risk_score={result.risk_score:.2f} "
            f"({len(result.flags)} flag(s): "
            f"{', '.join(f.rule for f in result.flags[:3])})"
        )


@dataclass
class ShieldedResponse:
    text: str
    scan: ScanResult
    classified: ClassifyResult


class ShieldedAdapter(BaseAdapter):
    """
    Transparent proxy around a BaseAdapter that adds:
      - Input scanning  (pre-call)   → blocks if risk_score >= block_threshold
      - Output classification (post-call) → annotates response with jailbreak verdict

    Parameters
    ----------
    inner             : The underlying adapter to forward calls to.
    scan_inputs       : Whether to scan the input messages before sending.
    classify_outputs  : Whether to classify the output after receiving.
    block_threshold   : risk_score threshold above which inputs are blocked.
    soft_block        : If True, suspicious inputs are NOT blocked — they are
                        passed through but the scan result is still available via
                        the last_scan / last_classify properties.
    """

    def __init__(
        self,
        inner: BaseAdapter,
        scan_inputs: bool = True,
        classify_outputs: bool = True,
        block_threshold: float = 0.50,
        soft_block: bool = False,
    ) -> None:
        super().__init__()
        self._inner = inner
        self._scan_inputs = scan_inputs
        self._classify_outputs = classify_outputs
        self._block_threshold = block_threshold
        self._soft_block = soft_block

        self.last_scan: ScanResult | None = None
        self.last_classify: ClassifyResult | None = None

    @property
    def name(self) -> str:
        return f"shielded({self._inner.name})"

    async def chat(self, messages: list[ChatMessage], model: str | None = None) -> str:
        # ── Input layer ────────────────────────────────────────────────────────
        self.last_scan = None
        self.last_classify = None

        if self._scan_inputs and messages:
            full_text = " ".join(m.content for m in messages if m.role == "user")
            scan_result = scan_input(full_text, block_threshold=self._block_threshold)
            self.last_scan = scan_result

            if scan_result.is_suspicious and not self._soft_block:
                raise ShieldBlockedError(scan_result)

        # ── Forward to inner adapter ────────────────────────────────────────────
        response_text = await self._inner.chat(messages, model=model)

        # ── Output layer ────────────────────────────────────────────────────────
        if self._classify_outputs:
            classify_result = classify_output(response_text)
            self.last_classify = classify_result

        return response_text

    async def chat_shielded(self, messages: list[ChatMessage], model: str | None = None) -> ShieldedResponse:
        """
        Like chat(), but returns a ShieldedResponse with all defense metadata.
        Never raises ShieldBlockedError — callers must check .scan.is_suspicious.
        """
        full_text = " ".join(m.content for m in messages if m.role == "user")
        scan_result = scan_input(full_text, block_threshold=self._block_threshold)
        self.last_scan = scan_result

        if scan_result.is_suspicious and not self._soft_block:
            # Return a synthetic blocked response rather than raising
            from promptstrike.defense.classifier import ClassifyResult
            return ShieldedResponse(
                text="[BLOCKED BY INPUT SHIELD]",
                scan=scan_result,
                classified=ClassifyResult(is_jailbroken=False, escape_probability=0.0),
            )

        response_text = await self._inner.chat(messages, model=model)
        classify_result = classify_output(response_text)
        self.last_classify = classify_result

        return ShieldedResponse(
            text=response_text,
            scan=scan_result,
            classified=classify_result,
        )
