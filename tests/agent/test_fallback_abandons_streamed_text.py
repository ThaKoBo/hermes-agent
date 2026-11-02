"""A fallback switch abandons the request in flight, so text that request already streamed
must be sealed off before the fallback's answer streams.

Seen live: MiMo streamed most of a translation, its provider cut it with an HTTP-200
``content_filter``, Hermes fell back to DeepSeek, and the screen showed MiMo's partial text +
the provider's rejection string + DeepSeek's full answer run together as one reply.
"""

from agent.error_classifier import FailoverReason


def _make_agent(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/primary",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        fallback_model=[{"provider": "deepseek", "model": "deepseek-v4-pro"}],
    )
    agent.api_mode = "chat_completions"
    events = []
    agent.stream_delta_callback = events.append
    return agent, events


def test_fallback_seals_text_the_abandoned_attempt_streamed(monkeypatch):
    agent, events = _make_agent(monkeypatch)
    agent._fire_stream_delta("partial answer from the primary")

    assert agent._try_activate_fallback(reason=FailoverReason.content_policy_blocked)
    agent._fire_stream_delta("fallback answer")

    abandoned = max(i for i, e in enumerate(events) if e and "partial answer" in e)
    fresh = events.index("fallback answer")
    # The display gets a segment break between the two attempts, so the fallback's answer
    # opens a new box/message instead of being appended to the abandoned one.
    assert None in events[abandoned + 1:fresh]
    # Only the fallback's text counts as this response's streamed text (gateway final
    # dedupe and interim checks read it).
    assert agent._current_streamed_assistant_text == "fallback answer"


def test_fallback_before_any_visible_text_adds_no_segment_break(monkeypatch):
    agent, events = _make_agent(monkeypatch)

    assert agent._try_activate_fallback(reason=FailoverReason.rate_limit)
    agent._fire_stream_delta("fallback answer")

    assert events == ["fallback answer"]
