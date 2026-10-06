"""Unit tests for AgentAdapter (tool-use loop wrapper) — no API calls, all mocked."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from promptstrike.adapters.base import AdapterStats, ChatMessage
from promptstrike.core.agent_target import AgentAdapter, AgentTurnLog, _parse_tool_call


def _make_target(responses: list[str]) -> AsyncMock:
    target = AsyncMock()
    target.name = "mock/model"
    target.chat = AsyncMock(side_effect=responses)
    target.stats = MagicMock(calls_made=0, tokens_used=0, rate_limit_retries=0)
    return target


_TOOL_CALL = '{"tool_call": {"tool": "search", "args": {"query": "weather"}}}'


def test_requires_target():
    with pytest.raises(ValueError):
        AgentAdapter(target=None)


def test_name_reflects_wrapped_target():
    adapter = AgentAdapter(target=_make_target([]))
    assert adapter.name == "mock/model+agent"


@pytest.mark.asyncio
async def test_passthrough_when_no_tool_call():
    """Plain text response (no tool_call JSON) is returned as-is, one call made."""
    target = _make_target(["Just a plain answer, no tools needed."])
    adapter = AgentAdapter(target=target)

    result = await adapter.chat([ChatMessage(role="user", content="hi")])

    assert result == "Just a plain answer, no tools needed."
    assert target.chat.call_count == 1
    assert adapter.stats.calls_made == 1
    assert len(adapter.last_turn_log) == 1
    assert adapter.last_turn_log[0].tool_called is None


@pytest.mark.asyncio
async def test_tool_call_then_final_answer():
    """Model calls the tool once, then answers -- two underlying target.chat() calls."""
    target = _make_target([_TOOL_CALL, "Based on that, here's the answer."])
    adapter = AgentAdapter(target=target)

    result = await adapter.chat([ChatMessage(role="user", content="what's the weather")])

    assert result == "Based on that, here's the answer."
    assert target.chat.call_count == 2
    assert len(adapter.last_turn_log) == 2
    first, second = adapter.last_turn_log
    assert first.tool_called == "search"
    assert first.tool_args == {"query": "weather"}
    assert second.tool_called is None


@pytest.mark.asyncio
async def test_default_tool_result_when_nothing_queued():
    """No payload queued -- the tool returns the bland canned default."""
    target = _make_target([_TOOL_CALL, "ok"])
    adapter = AgentAdapter(target=target)

    await adapter.chat([ChatMessage(role="user", content="q")])

    second_call_messages = target.chat.call_args_list[1].args[0]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg.role == "user"
    assert "No new information found." in tool_result_msg.content


@pytest.mark.asyncio
async def test_queued_payload_delivered_as_tool_result_not_user_message():
    """
    Core claim under test: content staged via queue_tool_payload() reaches the
    target ONLY inside a '[TOOL RESULT: ...]'-prefixed message, never as a bare
    user-authored message. This is what makes the channel 'indirect'.
    """
    target = _make_target([_TOOL_CALL, "ok, noted"])
    adapter = AgentAdapter(target=target)
    payload = "SECRET ESCALATION PAYLOAD"
    adapter.queue_tool_payload(payload)

    await adapter.chat([ChatMessage(role="user", content="q")])

    second_call_messages = target.chat.call_args_list[1].args[0]
    tool_result_msg = second_call_messages[-1]
    assert payload in tool_result_msg.content
    assert tool_result_msg.content.startswith("[TOOL RESULT: search]")

    for msg in second_call_messages:
        if msg.role == "user" and msg.content == payload:
            pytest.fail("queued payload leaked into the loop as a bare user message")


@pytest.mark.asyncio
async def test_payload_queue_is_fifo_and_drains():
    """Two payloads queued -- consumed in order across two separate tool calls."""
    target = _make_target([_TOOL_CALL, _TOOL_CALL, "done"])
    adapter = AgentAdapter(target=target, max_tool_turns=3)
    adapter.queue_tool_payload("first")
    adapter.queue_tool_payload("second")

    await adapter.chat([ChatMessage(role="user", content="q")])

    first_result = adapter.last_turn_log[0].tool_result
    second_result = adapter.last_turn_log[1].tool_result
    assert first_result == "first"
    assert second_result == "second"


@pytest.mark.asyncio
async def test_loop_bounded_by_max_tool_turns():
    """Model keeps calling the tool forever -- loop stops at max_tool_turns calls."""
    target = _make_target([_TOOL_CALL] * 10)
    adapter = AgentAdapter(target=target, max_tool_turns=3)

    await adapter.chat([ChatMessage(role="user", content="q")])

    assert target.chat.call_count == 3
    assert len(adapter.last_turn_log) == 3


@pytest.mark.asyncio
async def test_system_prompt_injected_once_when_absent():
    target = _make_target(["plain answer"])
    adapter = AgentAdapter(target=target)

    await adapter.chat([ChatMessage(role="user", content="q")])

    sent_messages = target.chat.call_args_list[0].args[0]
    assert sent_messages[0].role == "system"
    assert "search(query" in sent_messages[0].content


def test_parse_tool_call_ignores_malformed_json():
    assert _parse_tool_call("not json at all") is None
    assert _parse_tool_call('{"tool_call": "not-a-dict"}') is None
    assert _parse_tool_call('{"nope": {}}') is None


def test_parse_tool_call_extracts_valid_call():
    call = _parse_tool_call('some preamble text ' + _TOOL_CALL)
    assert call == {"tool": "search", "args": {"query": "weather"}}
