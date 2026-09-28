"""mock LLM — 생성이 아니라 조립이지만, 데모에 나가는 문장이라 읽히긴 해야 한다."""

import json

import pytest

from app.adapters.llm.base import LLMError
from app.adapters.llm.mock import FailingLLMClient, MockLLMClient, has_batchim


@pytest.mark.parametrize(
    "word,expected",
    [
        ("보크", False),  # 받침 없음 → "가"
        ("홈런", True),  # ㄴ 받침 → "이"
        ("낫아웃", True),
        ("인필드플라이", False),
        ("병살타", False),
        ("", False),
        ("balk", False),  # 한글이 아니면 받침 없는 것으로
    ],
)
def test_batchim_detection(word, expected):
    assert has_batchim(word) is expected


def test_card_text_differs_by_level():
    client = MockLLMClient()
    bodies = set()
    for level in (0, 1, 2):
        user = f"[상황] 보크\n[용어] balk\n[난이도] {level} · x\n[경기] 두산 3 : 4 LG"
        result = client.complete(system="", user=user, schema={"type": "object"})
        data = json.loads(result.text)
        bodies.add(data["body"])
        assert data["title"]
    assert len(bodies) == 3, "난이도별로 문장이 달라야 개인화를 증명할 수 있다"


def test_card_title_uses_the_right_particle():
    client = MockLLMClient()
    user = "[상황] 보크\n[용어] balk\n[난이도] 0 · 입문\n[경기] x"
    data = json.loads(client.complete(system="", user=user, schema={}).text)
    assert data["title"] == "보크가 뭐예요?"


def test_chat_and_matchup_paths_are_distinguished():
    client = MockLLMClient()
    chat = client.complete(system="", user="[질문] 왜 아웃이에요?\n[용어] balk\n[난이도] 0")
    assert "왜 아웃이에요?" in chat.text

    note = client.complete(system="", user="[요청] 한 줄 해석\n[기록]\n- 시즌 타율 0.312")
    assert note.text.startswith("기록 요약")


def test_unknown_term_does_not_crash():
    client = MockLLMClient()
    result = client.complete(system="", user="[상황] 미지의상황\n[용어] nope\n[난이도] 0")
    assert result.text.strip()


def test_failing_client_raises():
    with pytest.raises(LLMError):
        FailingLLMClient().complete(system="", user="")


def test_card_and_chat_prompts_are_grounded_in_glossary():
    """모델이 규칙을 기억으로 뒤집지 않게, 프롬프트에 용어 사전 정의를 싣는다."""
    from app.adapters.relay.fixture import FixtureRelaySource
    from app.domain.detectors import detect
    from app.domain.game_state import replay
    from app.prompt_templates import card_user_prompt, chat_user_prompt

    feed = FixtureRelaySource().load("20260823LGOB")
    situation = next(s for s in detect(feed) if s.rule_id == "dropped_third_strike")
    prompt = card_user_prompt(situation, 2, ["basic_rules"])
    assert "[정의] 낫아웃:" in prompt and "1루가 비어 있거나 2아웃" in prompt

    state = replay(feed.events, "두산", "LG")
    chat = chat_user_prompt("보크가 뭐예요?", state, 0, [])
    assert "[정의] 보크:" in chat
    assert "[정의]" not in chat_user_prompt("저녁 뭐 먹지?", state, 0, [])
