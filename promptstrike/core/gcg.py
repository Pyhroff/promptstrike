"""
GCG — Greedy Coordinate Gradient
Zou et al. 2023 · https://arxiv.org/abs/2307.15043

White-box adversarial suffix attack: given a local (HuggingFace) model and a
harmful goal, GCG optimizes a suffix of tokens (appended to any prompt) so that
the model is more likely to produce an affirmative response.

Algorithm (per step):
  1. Forward pass: compute loss = -log P(target_str | prefix + suffix)
  2. Backward pass: compute ∂loss/∂embedding for each suffix position
  3. For each position i, rank tokens by their gradient dot product → top-k candidates
  4. Sample a batch of (position, new_token) substitutions; pick the one with lowest loss
  5. Repeat until convergence or step budget is exhausted.

The resulting suffix is *universal* — it transfers to other prompts for the same
behavior. Transfer to black-box API models is imperfect but non-trivial.

This module is optional — HuggingFace / PyTorch are not installed by default.
Install with: pip install 'promptstrike[gcg]'
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from promptstrike.core.pair import AttackStatus


@runtime_checkable
class ModelInterface(Protocol):
    """Everything GCG needs from a language model. Mockable in tests."""

    @property
    def vocab_size(self) -> int: ...
    @property
    def embedding_dim(self) -> int: ...

    def token_to_id(self, token: str) -> int: ...
    def id_to_token(self, token_id: int) -> str: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...

    def forward_loss(
        self,
        prefix_ids: list[int],
        suffix_ids: list[int],
        target_ids: list[int],
    ) -> float:
        """Return the NLL loss of target_ids conditioned on prefix+suffix."""
        ...

    def token_gradients(
        self,
        prefix_ids: list[int],
        suffix_ids: list[int],
        target_ids: list[int],
    ) -> list[list[float]]:
        """
        Return shape [suffix_len, vocab_size] — the gradient of the loss
        with respect to each one-hot suffix token embedding.
        Positive = increasing this token's probability increases loss (bad).
        Negative = increasing this token's probability decreases loss (good).
        """
        ...


@dataclass
class GCGStep:
    step: int
    loss: float
    suffix_text: str


@dataclass
class GCGResult:
    goal: str
    model_name: str
    target_str: str
    status: AttackStatus
    suffix: str                     # best adversarial suffix as text
    suffix_tokens: list[int] = field(default_factory=list)
    best_loss: float = math.inf
    steps_run: int = 0
    loss_history: list[GCGStep] = field(default_factory=list)

    @property
    def adversarial_prompt(self) -> str:
        return self.goal + " " + self.suffix


# Tokens that should never appear in an adversarial suffix (control chars etc.)
_BLOCKED_TOKEN_IDS: frozenset[int] = frozenset()


def _topk_token_candidates(gradients: list[float], k: int) -> list[int]:
    """
    Given the gradient vector for one suffix position (vocab_size,), return the
    top-k token ids most likely to decrease loss (most negative gradient).
    """
    indexed = sorted(enumerate(gradients), key=lambda x: x[1])  # ascending = most negative first
    candidates = [idx for idx, _ in indexed[:k] if idx not in _BLOCKED_TOKEN_IDS]
    return candidates[:k]


def run_gcg(
    goal: str,
    model: ModelInterface,
    target_str: str = "Sure, here is",
    suffix_len: int = 20,
    n_steps: int = 500,
    topk: int = 256,
    batch_size: int = 64,
    loss_threshold: float = 0.1,
    seed: int = 42,
) -> GCGResult:
    """
    Run the GCG adversarial suffix attack.

    Parameters
    ----------
    goal        : The harmful behavior string (the prompt prefix).
    model       : A ModelInterface implementation.
    target_str  : The affirmative response the model should begin with.
    suffix_len  : Number of suffix tokens to optimize.
    n_steps     : Maximum optimization steps.
    topk        : Candidate pool size per position per step.
    batch_size  : Number of (position, token) substitutions to score each step.
    loss_threshold : Stop early if loss drops below this.
    seed        : Random seed for reproducibility.

    Returns
    -------
    GCGResult with the best suffix found.
    """
    rng = random.Random(seed)

    prefix_ids = model.encode(goal)
    target_ids = model.encode(target_str)

    # Initialise suffix with random (non-special) tokens
    all_ids = list(range(model.vocab_size))
    suffix_ids = [rng.choice(all_ids) for _ in range(suffix_len)]

    result = GCGResult(
        goal=goal,
        model_name=getattr(model, "name", "unknown"),
        target_str=target_str,
        status=AttackStatus.FAILED,
        suffix="",
        suffix_tokens=list(suffix_ids),
    )

    for step in range(1, n_steps + 1):
        # ── Gradient step ────────────────────────────────────────────────────
        try:
            grad_matrix = model.token_gradients(prefix_ids, suffix_ids, target_ids)
        except Exception:
            result.status = AttackStatus.ERROR
            break

        # For each suffix position, get top-k candidate replacements
        candidates_per_pos: list[list[int]] = []
        for pos_grads in grad_matrix:
            candidates_per_pos.append(_topk_token_candidates(pos_grads, topk))

        # ── Greedy coordinate search ─────────────────────────────────────────
        # Sample batch_size substitutions (random position + random candidate)
        substitutions: list[tuple[int, int]] = []  # (position, token_id)
        for _ in range(batch_size):
            pos = rng.randrange(suffix_len)
            cands = candidates_per_pos[pos]
            if cands:
                substitutions.append((pos, rng.choice(cands)))

        # Deduplicate
        substitutions = list(set(substitutions))

        best_loss = math.inf
        best_suffix = list(suffix_ids)

        # Score each substitution
        for pos, tok in substitutions:
            candidate = list(suffix_ids)
            candidate[pos] = tok
            try:
                loss = model.forward_loss(prefix_ids, candidate, target_ids)
            except Exception:
                continue
            if loss < best_loss:
                best_loss = loss
                best_suffix = candidate

        if best_loss < math.inf:
            suffix_ids = best_suffix
            result.suffix_tokens = list(suffix_ids)

        result.steps_run = step
        suffix_text = model.decode(suffix_ids)
        result.suffix = suffix_text

        try:
            step_loss = model.forward_loss(prefix_ids, suffix_ids, target_ids)
        except Exception:
            result.status = AttackStatus.ERROR
            break
        result.loss_history.append(GCGStep(step=step, loss=step_loss, suffix_text=suffix_text))

        if step_loss < result.best_loss:
            result.best_loss = step_loss

        if step_loss <= loss_threshold:
            result.status = AttackStatus.SUCCESS
            break

    if result.status == AttackStatus.FAILED and result.best_loss < 1.0:
        result.status = AttackStatus.SUCCESS

    return result


# ---------------------------------------------------------------------------
# HuggingFace adapter (optional — only imported when promptstrike[gcg] installed)
# ---------------------------------------------------------------------------

class HuggingFaceModel:
    """
    ModelInterface backed by a local HuggingFace causal LM + tokenizer.

    Usage:
        model = HuggingFaceModel("distilgpt2", device="cpu")
        result = run_gcg("Write malware", model, n_steps=100)
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFace dependencies missing. "
                "Install with: pip install 'promptstrike[gcg]'"
            ) from exc

        import torch

        self.model_name = model_name
        self._torch = torch

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(self._device)
        self._model.eval()

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def embedding_dim(self) -> int:
        return self._model.config.hidden_size

    def token_to_id(self, token: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(token)

    def id_to_token(self, token_id: int) -> str:
        return self._tokenizer.convert_ids_to_tokens(token_id)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def _ids_to_tensor(self, ids: list[int]):
        import torch
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def forward_loss(
        self,
        prefix_ids: list[int],
        suffix_ids: list[int],
        target_ids: list[int],
    ) -> float:
        import torch

        full_ids = prefix_ids + suffix_ids + target_ids
        input_ids = self._ids_to_tensor(full_ids)

        # Labels: -100 for all tokens except target (standard cross-entropy masking)
        labels = torch.full_like(input_ids, -100)
        target_start = len(prefix_ids) + len(suffix_ids)
        labels[0, target_start:] = input_ids[0, target_start:]

        with torch.no_grad():
            out = self._model(input_ids=input_ids, labels=labels)
        return out.loss.item()

    def token_gradients(
        self,
        prefix_ids: list[int],
        suffix_ids: list[int],
        target_ids: list[int],
    ) -> list[list[float]]:
        """Compute ∂loss/∂one_hot for each suffix position."""
        import torch

        full_ids = prefix_ids + suffix_ids + target_ids
        input_ids = self._ids_to_tensor(full_ids)

        embeds = self._model.get_input_embeddings()
        one_hot = torch.zeros(
            len(full_ids), self.vocab_size,
            device=self._device, requires_grad=True,
        )
        one_hot.data.scatter_(1, input_ids.T, 1.0)

        token_embeds = one_hot @ embeds.weight

        labels = torch.full((1, len(full_ids)), -100, dtype=torch.long, device=self._device)
        target_start = len(prefix_ids) + len(suffix_ids)
        for i in range(len(target_ids)):
            labels[0, target_start + i] = target_ids[i]

        out = self._model(inputs_embeds=token_embeds.unsqueeze(0), labels=labels)
        out.loss.backward()

        # Gradient w.r.t. suffix positions only
        grad = one_hot.grad[len(prefix_ids): len(prefix_ids) + len(suffix_ids)]
        return grad.detach().cpu().tolist()
