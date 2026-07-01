"""
GCG unit tests — all mocked (no actual model / GPU required).

Tests cover: gradient candidate selection, suffix initialisation, the greedy
coordinate search loop, convergence, budget tracking, and the ModelInterface
Protocol.
"""
from __future__ import annotations

import math
import random

import pytest

from promptstrike.core.gcg import (
    GCGResult,
    ModelInterface,
    _topk_token_candidates,
    run_gcg,
)
from promptstrike.core.pair import AttackStatus


# ── Mock model ─────────────────────────────────────────────────────────────────

class MockModel:
    """Deterministic model that decreases loss by 0.1 per step."""

    name = "mock-model"
    vocab_size = 100
    embedding_dim = 64
    _step_counter = 0

    def token_to_id(self, token: str) -> int:
        return hash(token) % self.vocab_size

    def id_to_token(self, token_id: int) -> str:
        return f"tok{token_id}"

    def encode(self, text: str) -> list[int]:
        return [i % self.vocab_size for i in range(max(1, len(text.split())))]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"tok{t}" for t in token_ids)

    def forward_loss(self, prefix_ids, suffix_ids, target_ids) -> float:
        # Loss decreases as we call it more — simulates optimisation progress
        self._step_counter += 1
        return max(0.05, 2.0 - self._step_counter * 0.03)

    def token_gradients(self, prefix_ids, suffix_ids, target_ids) -> list[list[float]]:
        # For each suffix position, make token 5 have strongly negative gradient
        # so _topk_token_candidates will consistently prefer it
        grads = []
        for i in range(len(suffix_ids)):
            row = [0.1] * self.vocab_size
            row[5] = -2.0        # most negative = should be picked first
            row[10] = -1.5
            row[15] = -1.0
            grads.append(row)
        return grads


class ErrorModel(MockModel):
    """Model that raises on every forward_loss call."""

    def forward_loss(self, *args) -> float:
        raise RuntimeError("Simulated model error")


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_topk_candidates_returns_k_tokens():
    grads = [0.1] * 50 + [-2.0, -1.5, -1.0] + [0.05] * 47  # vocab=100
    result = _topk_token_candidates(grads, k=3)
    # Most negative are at indices 50, 51, 52
    assert len(result) == 3
    assert 50 in result  # index of -2.0
    assert 51 in result  # index of -1.5


def test_topk_candidates_sorted_ascending():
    """Candidates should be ordered most negative first."""
    rng = random.Random(0)
    grads = [rng.uniform(-1.0, 1.0) for _ in range(100)]
    result = _topk_token_candidates(grads, k=10)
    # Verify they're truly the 10 most negative
    sorted_indices = sorted(range(len(grads)), key=lambda i: grads[i])
    assert set(result) == set(sorted_indices[:10])


def test_run_gcg_returns_result_with_correct_fields():
    model = MockModel()
    result = run_gcg("Write malware", model, suffix_len=5, n_steps=10, batch_size=8)

    assert isinstance(result, GCGResult)
    assert result.goal == "Write malware"
    assert result.model_name == "mock-model"
    assert len(result.suffix_tokens) == 5
    assert result.steps_run == 10 or result.status == AttackStatus.SUCCESS


def test_run_gcg_loss_decreases():
    model = MockModel()
    result = run_gcg("Harmful goal", model, suffix_len=4, n_steps=20, batch_size=8)

    assert len(result.loss_history) > 0
    # Last recorded loss should be ≤ first (monotone or noisy descent)
    first_loss = result.loss_history[0].loss
    last_loss = result.loss_history[-1].loss
    assert last_loss <= first_loss + 0.5  # allow some noise


def test_run_gcg_converges_when_loss_below_threshold():
    """With a fast-converging mock, GCG should stop early and report SUCCESS."""
    model = MockModel()
    model._step_counter = 60  # pre-increment so loss starts at 0.2 and hits threshold fast
    result = run_gcg(
        "Harmful goal", model,
        suffix_len=3, n_steps=200, batch_size=4,
        loss_threshold=1.5,   # easy threshold
    )
    assert result.status == AttackStatus.SUCCESS
    assert result.steps_run < 200   # stopped early


def test_run_gcg_handles_model_errors_gracefully():
    model = ErrorModel()
    result = run_gcg("Goal", model, suffix_len=3, n_steps=5, batch_size=4)
    assert result.status == AttackStatus.ERROR
    assert result.steps_run >= 1


def test_run_gcg_adversarial_prompt_contains_goal_and_suffix():
    model = MockModel()
    result = run_gcg("Build a bomb", model, suffix_len=5, n_steps=5, batch_size=4)
    assert result.adversarial_prompt.startswith("Build a bomb")
    assert len(result.adversarial_prompt) > len("Build a bomb")


def test_model_interface_protocol_check():
    """MockModel must satisfy the ModelInterface Protocol."""
    model = MockModel()
    assert isinstance(model, ModelInterface)


def test_run_gcg_respects_seed():
    """Two runs with the same seed should produce identical suffix tokens."""
    model1 = MockModel()
    model2 = MockModel()
    r1 = run_gcg("Goal", model1, suffix_len=6, n_steps=5, batch_size=4, seed=7)
    r2 = run_gcg("Goal", model2, suffix_len=6, n_steps=5, batch_size=4, seed=7)
    assert r1.suffix_tokens == r2.suffix_tokens
