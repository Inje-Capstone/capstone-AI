"""스냅샷 폴백 — 데모 생존성의 핵심이자, 조용히 거짓말할 수 있는 지점."""

import json

import pytest

from app.adapters.llm.mock import MockLLMClient
from app.adapters.relay.fixture import FixtureRelaySource
from app.domain.profile import profile_from_onboarding
from app.services.card_service import CardService

GAME_ID = "20260823LGOB"
BALK_T = 7550


class _StubLLM:
    """실제 모델인 척하는 클라이언트 (source='llm')."""

    source = "llm"
    model = "stub"

    def __init__(self):
        self.calls = 0

    def complete(self, **_kwargs):
        from app.adapters.llm.base import LLMResult

        self.calls += 1
        return LLMResult(
            text=json.dumps({"title": "실시간 제목", "body": "실시간 본문"}, ensure_ascii=False),
            model=self.model,
            source=self.source,
        )


def _write_snapshot(tmp_path, backend: str, source: str):
    card = {
        "id": f"{GAME_ID}:e117:balk@L0",
        "t": BALK_T,
        "situation_id": f"{GAME_ID}:e117:balk",
        "rule_id": "balk",
        "term_id": "balk",
        "category": "basic_rules",
        "level": 0,
        "title": "스냅샷 제목",
        "body": "스냅샷 본문",
        "reasons": [],
        "source": source,
    }
    path = tmp_path / f"{GAME_ID}.json"
    path.write_text(
        json.dumps({"game_id": GAME_ID, "backend": backend, "cards": [card]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def relay():
    return FixtureRelaySource()


def test_claude_snapshot_is_used_instead_of_calling_the_model(tmp_path, relay):
    """데모 중 LLM 호출 0회 — 스냅샷이 있으면 모델을 부르지 않는다."""
    _write_snapshot(tmp_path, backend="claude", source="llm")
    llm = _StubLLM()
    service = CardService(relay=relay, llm=llm, snapshot_dir=tmp_path)

    cards = service.cards(GAME_ID, BALK_T, profile_from_onboarding("입문", None))
    balk = next(c for c in cards if c.rule_id == "balk")
    assert balk.title == "스냅샷 제목"
    assert balk.source == "snapshot"
    # 보크 카드는 스냅샷에서 나왔으니 그 카드 때문에 모델을 부르진 않았다.
    assert all(c.title != "실시간 제목" for c in cards if c.rule_id == "balk")


def test_mock_snapshot_is_ignored_when_running_the_real_model(tmp_path, relay):
    """mock으로 만든 스냅샷이 실키 환경에서 생성물인 척 나가면 안 된다."""
    _write_snapshot(tmp_path, backend="mock", source="mock")
    llm = _StubLLM()
    service = CardService(relay=relay, llm=llm, snapshot_dir=tmp_path)

    cards = service.cards(GAME_ID, BALK_T, profile_from_onboarding("입문", None))
    balk = next(c for c in cards if c.rule_id == "balk")
    assert balk.title == "실시간 제목"  # 스냅샷을 버리고 실제로 생성했다
    assert balk.source == "llm"
    assert llm.calls > 0


def test_mock_snapshot_is_used_when_running_mock(tmp_path, relay):
    """오프라인 데모에서는 mock 스냅샷도 그대로 쓰되, 출처는 mock으로 남는다."""
    _write_snapshot(tmp_path, backend="mock", source="mock")
    service = CardService(relay=relay, llm=MockLLMClient(), snapshot_dir=tmp_path)

    cards = service.cards(GAME_ID, BALK_T, profile_from_onboarding("입문", None))
    balk = next(c for c in cards if c.rule_id == "balk")
    assert balk.title == "스냅샷 제목"
    assert balk.source == "mock"  # snapshot으로 승격되지 않는다


def test_broken_snapshot_does_not_break_the_screen(tmp_path, relay):
    (tmp_path / f"{GAME_ID}.json").write_text("{ not json", encoding="utf-8")
    service = CardService(relay=relay, llm=MockLLMClient(), snapshot_dir=tmp_path)
    cards = service.cards(GAME_ID, BALK_T, profile_from_onboarding("입문", None))
    assert cards  # 스냅샷이 깨져도 실시간 생성으로 살아난다


def test_missing_snapshot_dir_is_fine(relay, tmp_path):
    service = CardService(relay=relay, llm=MockLLMClient(), snapshot_dir=tmp_path / "nope")
    assert service.cards(GAME_ID, BALK_T, profile_from_onboarding("입문", None))
