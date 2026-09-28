"""Claude 클라이언트 설정 — 네트워크 없이 SDK 생성 인자만 본다."""

import anthropic
import pytest

from app.adapters.llm.claude import ClaudeClient
from app.config import get_settings


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    yield seen
    get_settings.cache_clear()


def test_workspace_header_sent_when_configured(monkeypatch, captured):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
    get_settings.cache_clear()
    ClaudeClient()
    assert captured["default_headers"] == {"anthropic-workspace-id": "wrkspc_123"}


def test_no_workspace_header_by_default(monkeypatch, captured):
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    get_settings.cache_clear()
    client = ClaudeClient(model="claude-haiku-4-5")
    assert captured["default_headers"] is None
    assert client.model == "claude-haiku-4-5"


def test_effort_omitted_for_haiku():
    from app.adapters.llm.claude import _supports_effort

    assert _supports_effort("claude-sonnet-5")
    assert not _supports_effort("claude-haiku-4-5")
