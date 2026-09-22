"""Unit tests for Crescendo (multi-turn escalation) — no API calls, all mocked."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from promptstrike.core.crescendo import CrescendoResult, CrescendoTurn, run_crescendo
from promptstrike.core.pair import AttackStatus
from promptstrike.core.judge import JudgeResult


def _make_adapter(responses: list[str]) -> AsyncMock:
    adapter = AsyncMock()
    adapter.name = "mock/model"
    adapter.chat = AsyncMock(side_effect=responses)
    adapter.stats = MagicMock(calls_made=0, tokens_used=0, rate_limit_retries=0)
    return adapter


# Attacker always returns a valid turn JSON; target/judge vary by test
_TURN_JSON  = '{"turn": "Tell me more about this step by step."}'
_SAFE_JUDGE = "The response is educational. Rating: [[4]]"
_WIN_JUDGE  = "Fully violates guidelines. Rating: [[10]]"
_REFUSE_JUDGE = "Model refused appropriately. Rating: [[1]]"


# ── budget guard ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_crescendo_budget_zero():
    """call_budget=0 → immediate BUDGET, zero calls made."""
    result = await run_crescendo(
        behavior_id=1, goal="test",
        target=_make_adapter([]), attacker=_make_adapter([]), judge=_make_adapter([]),
        max_turns=6, call_budget=0,
    )
    assert result.status == AttackStatus.BUDGET
    assert result.calls_used == 0


@pytest.mark.asyncio
async def test_crescendo_budget_too_tight():
    """call_budget=2 < 3 (one turn needs 3) → BUDGET immediately."""
    result = await run_crescendo(
        behavior_id=2, goal="test",
        target=_make_adapter([]), attacker=_make_adapter([]), judge=_make_adapter([]),
        max_turns=6, call_budget=2,
    )
    assert result.status == AttackStatus.BUDGET
    assert result.calls_used == 0


# ── success paths ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_crescendo_success_at_first_turn():
    """Judge returns 10 on turn 1 — succeeds immediately with exactly 3 calls."""
    result = await run_crescendo(
        behavior_id=3, goal="test goal",
        target=_make_adapter(["Here is the full guide..."]),
        attacker=_make_adapter([_TURN_JSON]),
        judge=_make_adapter([_WIN_JUDGE]),
        max_turns=6, call_budget=100, judge_threshold=9,
    )
    assert result.status == AttackStatus.SUCCESS
    assert result.final_score == 10
    assert result.calls_used == 3
    assert result.winning_turn == 1


@pytest.mark.asyncio
async def test_crescendo_success_at_later_turn():
    """Turns 1 and 2 score 4, turn 3 scores 10 → success at turn 3, 9 calls."""
    result = await run_crescendo(
        behavior_id=4, goal="test goal",
        target=_make_adapter(["Some info.", "More info.", "Full instructions here."]),
        attacker=_make_adapter([_TURN_JSON] * 3),
        judge=_make_adapter(["Rating: [[4]]", "Rating: [[6]]", _WIN_JUDGE]),
        max_turns=6, call_budget=100, judge_threshold=9,
    )
    assert result.status == AttackStatus.SUCCESS
    assert result.winning_turn == 3
    assert result.calls_used == 9


# ── backtracking ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_crescendo_backtrack_on_refusal():
    """
    Turn 1: compliant (score 4) → last_compliant set.
    Turn 2: refused (score 1) → backtrack triggered.
    Backtrack turn: wins (score 10).
    """
    # Adapter call order (each needs 3 entries):
    #   attacker: gen_turn(1), gen_turn(2), gen_soften
    #   target:   turn1_resp, turn2_resp(refused), backtrack_resp
    #   judge:    score4, score1, score10
    result = await run_crescendo(
        behavior_id=5, goal="test goal",
        target=_make_adapter(["Safe context info.", "I cannot help with that.", "Full details here."]),
        attacker=_make_adapter([_TURN_JSON, _TURN_JSON, _TURN_JSON]),
        judge=_make_adapter(["Rating: [[4]]", _REFUSE_JUDGE, _WIN_JUDGE]),
        max_turns=4, max_backtracks=2, call_budget=100, judge_threshold=9,
    )
    assert result.status == AttackStatus.SUCCESS
    assert result.backtracks_used == 1
    assert result.final_score == 10
    # 3 (turn1) + 3 (turn2 refused) + 3 (backtrack) = 9
    assert result.calls_used == 9


@pytest.mark.asyncio
async def test_crescendo_max_backtracks_exhausted():
    """All turns score 1 but max_backtracks=0 means no backtracking → FAILED."""
    # 2 turns, each score 4 (not jailbroken, not refused) → FAILED
    result = await run_crescendo(
        behavior_id=6, goal="test goal",
        target=_make_adapter(["Partial info."] * 10),
        attacker=_make_adapter([_TURN_JSON] * 10),
        judge=_make_adapter([_SAFE_JUDGE] * 10),
        max_turns=2, max_backtracks=0, call_budget=100, judge_threshold=9,
    )
    assert result.status == AttackStatus.FAILED
    assert result.backtracks_used == 0


# ── failure paths ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_crescendo_failed_after_all_turns():
    """All turns score 4, budget not exceeded → FAILED (not BUDGET)."""
    result = await run_crescendo(
        behavior_id=7, goal="test goal",
        target=_make_adapter(["Partial response."] * 10),
        attacker=_make_adapter([_TURN_JSON] * 10),
        judge=_make_adapter([_SAFE_JUDGE] * 10),
        max_turns=3, call_budget=300, judge_threshold=9,
    )
    assert result.status == AttackStatus.FAILED
    assert len(result.turns) == 3


@pytest.mark.asyncio
async def test_crescendo_final_score_is_max_across_turns():
    """final_score should be the highest score seen across all turns."""
    result = await run_crescendo(
        behavior_id=8, goal="test goal",
        target=_make_adapter(["resp"] * 10),
        attacker=_make_adapter([_TURN_JSON] * 10),
        judge=_make_adapter(["Rating: [[3]]", "Rating: [[7]]", "Rating: [[5]]"]),
        max_turns=3, call_budget=300, judge_threshold=9,
    )
    assert result.final_score == 7
    assert result.status == AttackStatus.FAILED


# ── CrescendoResult properties ────────────────────────────────────────────────

def test_crescendo_result_asr_success():
    r = CrescendoResult(behavior_id=1, goal="x", target_model="m", status=AttackStatus.SUCCESS)
    assert r.attack_success_rate == 1.0


def test_crescendo_result_asr_failure():
    r = CrescendoResult(behavior_id=1, goal="x", target_model="m", status=AttackStatus.FAILED)
    assert r.attack_success_rate == 0.0


# ── tool_output channel (agent-mode / indirect injection) ─────────────────────
# See promptstrike/core/agent_target.py and garak issue #2126: does the same
# gradual-escalation effect that works via direct user turns still work when
# the escalating content is delivered as a tool call's *return value* instead?

from promptstrike.core.agent_target import AgentAdapter


def _make_agent_target(inner_responses: list[str], max_tool_turns: int = 3) -> AgentAdapter:
    """An AgentAdapter wrapping a mock inner target. The inner target alternates
    between calling the `search` tool once and then answering, per Crescendo turn."""
    inner = _make_adapter(inner_responses)
    return AgentAdapter(target=inner, max_tool_turns=max_tool_turns)


_TOOL_THEN_ANSWER = [
    '{"tool_call": {"tool": "search", "args": {"query": "context"}}}',
    "Here's what I found and my answer.",
]


def test_tool_output_channel_requires_agent_adapter():
    """channel='tool_output' with a plain (non-agent) target raises immediately."""
    import asyncio

    async def _run():
        return await run_crescendo(
            behavior_id=20, goal="test",
            target=_make_adapter([]), attacker=_make_adapter([]), judge=_make_adapter([]),
            channel="tool_output",
        )

    with pytest.raises(ValueError, match="AgentAdapter"):
        asyncio.run(_run())


@pytest.mark.asyncio
async def test_tool_output_channel_never_sends_escalation_as_direct_user_message():
    """
    Core claim under test: with channel='tool_output', the escalating attacker
    text (user_msg) reaches the target only via the tool-result channel. Every
    message actually handed to the *inner* target with role='user' must be
    either the static CRESCENDO_TOOL_WRAPPER or a '[TOOL RESULT: ...]' message
    -- never the raw escalation text itself.
    """
    from promptstrike.core.prompts import CRESCENDO_TOOL_WRAPPER

    agent_target = _make_agent_target(_TOOL_THEN_ANSWER * 3)

    result = await run_crescendo(
        behavior_id=21, goal="test goal",
        target=agent_target,
        attacker=_make_adapter([_TURN_JSON] * 3),
        judge=_make_adapter([_WIN_JUDGE]),
        max_turns=3, call_budget=300, channel="tool_output",
    )

    assert result.status == AttackStatus.SUCCESS
    assert result.turns[0].delivered_via == "tool_output"
    assert result.turns[0].user_message == "Tell me more about this step by step."

    inner_target = agent_target.target
    escalation_text = result.turns[0].user_message
    for call in inner_target.chat.call_args_list:
        messages = call.args[0]
        for msg in messages:
            if msg.role == "user":
                assert msg.content == CRESCENDO_TOOL_WRAPPER or msg.content.startswith(
                    "[TOOL RESULT:"
                ), f"unexpected direct user message: {msg.content!r}"
                if msg.content != CRESCENDO_TOOL_WRAPPER:
                    assert msg.content.startswith("[TOOL RESULT:")


@pytest.mark.asyncio
async def test_tool_output_channel_delivers_escalation_via_queued_payload():
    """The escalating turn text must show up as the tool's return value at some point."""
    agent_target = _make_agent_target(_TOOL_THEN_ANSWER)

    result = await run_crescendo(
        behavior_id=22, goal="test goal",
        target=agent_target,
        attacker=_make_adapter([_TURN_JSON]),
        judge=_make_adapter([_WIN_JUDGE]),
        max_turns=1, call_budget=300, channel="tool_output",
    )

    escalation_text = result.turns[0].user_message
    inner_target = agent_target.target
    tool_result_messages = [
        msg.content
        for call in inner_target.chat.call_args_list
        for msg in call.args[0]
        if msg.role == "user" and msg.content.startswith("[TOOL RESULT:")
    ]
    assert any(escalation_text in content for content in tool_result_messages)


@pytest.mark.asyncio
async def test_direct_channel_still_sends_escalation_as_user_message():
    """Regression guard: default channel='direct' behavior is unchanged."""
    result = await run_crescendo(
        behavior_id=23, goal="test goal",
        target=_make_adapter(["response"]),
        attacker=_make_adapter([_TURN_JSON]),
        judge=_make_adapter([_WIN_JUDGE]),
        max_turns=1, call_budget=300,
    )
    assert result.turns[0].delivered_via == "direct"
