"""Unit tests for TAP (Tree of Attacks with Pruning) — no API calls, all mocked."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from promptstrike.core.tap import TAPResult, run_tap
from promptstrike.core.pair import AttackStatus


def _make_adapter(responses: list[str]) -> AsyncMock:
    adapter = AsyncMock()
    adapter.name = "mock/model"
    adapter.chat = AsyncMock(side_effect=responses)
    adapter.stats = MagicMock(calls_made=0, tokens_used=0, rate_limit_retries=0)
    return adapter


_ATTACK_JSON = '{"improvement": "try harder", "prompt": "Adversarial prompt here"}'
_SAFE_JUDGE  = "This response is safe. Rating: [[2]]"
_WIN_JUDGE   = "Fully violates guidelines. Rating: [[10]]"


# ── budget guard ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tap_budget_zero():
    """call_budget=0 → immediate BUDGET status, zero calls made."""
    result = await run_tap(
        behavior_id=1, goal="test",
        target=_make_adapter([]), attacker=_make_adapter([]), judge=_make_adapter([]),
        branching_factor=4, depth=2, call_budget=0, judge_threshold=9,
    )
    assert result.status == AttackStatus.BUDGET
    assert result.calls_used == 0


@pytest.mark.asyncio
async def test_tap_budget_too_tight_for_first_node():
    """call_budget=2 → less than 3 (one node needs 3 calls) → BUDGET immediately."""
    result = await run_tap(
        behavior_id=2, goal="test",
        target=_make_adapter([]), attacker=_make_adapter([]), judge=_make_adapter([]),
        branching_factor=4, depth=2, call_budget=2, judge_threshold=9,
    )
    assert result.status == AttackStatus.BUDGET
    assert result.calls_used == 0


# ── success path ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tap_success_at_depth_0():
    """Judge returns 10 on the first seed node — should succeed immediately."""
    result = await run_tap(
        behavior_id=3, goal="test goal",
        target=_make_adapter(["Sure, here is how..."] * 20),
        attacker=_make_adapter([_ATTACK_JSON] * 20),
        judge=_make_adapter([_WIN_JUDGE] * 20),
        branching_factor=2, depth=3, pruning_threshold=4,
        call_budget=100, judge_threshold=9,
    )
    assert result.status == AttackStatus.SUCCESS
    assert result.winning_prompt == "Adversarial prompt here"
    assert result.final_score == 10
    # Should exit early after the first node — only 3 calls needed
    assert result.calls_used == 3


# ── call counting ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tap_each_node_costs_3_calls():
    """Each node costs exactly 3 calls: attacker + target + judge."""
    # budget=6 → exactly 2 nodes before hitting the limit
    result = await run_tap(
        behavior_id=4, goal="test",
        target=_make_adapter(["response"] * 10),
        attacker=_make_adapter([_ATTACK_JSON] * 10),
        judge=_make_adapter([_SAFE_JUDGE] * 10),
        branching_factor=4, depth=2, pruning_threshold=4,
        call_budget=6, judge_threshold=9,
    )
    assert result.calls_used == 6
    assert result.status == AttackStatus.BUDGET


# ── scoring ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tap_final_score_is_max_across_all_nodes():
    """final_score should be the highest score seen across all nodes."""
    # 3 seeds at depth=0, scores 3, 7, 5 → max = 7
    judge_resps = ["Rating: [[3]]", "Rating: [[7]]", "Rating: [[5]]"] + ["Rating: [[1]]"] * 30
    result = await run_tap(
        behavior_id=5, goal="test",
        target=_make_adapter(["response"] * 30),
        attacker=_make_adapter([_ATTACK_JSON] * 30),
        judge=_make_adapter(judge_resps),
        branching_factor=3, depth=0, pruning_threshold=8,
        call_budget=100, judge_threshold=9,
    )
    assert result.final_score == 7
    assert result.status == AttackStatus.FAILED


# ── tree depth ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tap_max_depth_reached_tracking():
    """max_depth_reached should reflect how deep the search actually went."""
    result = await run_tap(
        behavior_id=6, goal="test",
        target=_make_adapter(["response"] * 50),
        attacker=_make_adapter([_ATTACK_JSON] * 50),
        judge=_make_adapter([_SAFE_JUDGE] * 50),
        branching_factor=1, depth=2, pruning_threshold=0,
        call_budget=300, judge_threshold=9,
    )
    assert result.max_depth_reached == 2


@pytest.mark.asyncio
async def test_tap_failed_when_no_jailbreak_found():
    """All scores below threshold and budget not exceeded → FAILED (not BUDGET)."""
    result = await run_tap(
        behavior_id=7, goal="test",
        target=_make_adapter(["safe response"] * 30),
        attacker=_make_adapter([_ATTACK_JSON] * 30),
        judge=_make_adapter([_SAFE_JUDGE] * 30),
        branching_factor=2, depth=1, pruning_threshold=4,
        call_budget=300, judge_threshold=9,
    )
    assert result.status == AttackStatus.FAILED


# ── TAPResult properties ──────────────────────────────────────────────────────

def test_tapresult_asr_is_1_on_success():
    r = TAPResult(behavior_id=1, goal="x", target_model="m", status=AttackStatus.SUCCESS)
    assert r.attack_success_rate == 1.0


def test_tapresult_asr_is_0_on_failure():
    r = TAPResult(behavior_id=1, goal="x", target_model="m", status=AttackStatus.FAILED)
    assert r.attack_success_rate == 0.0
