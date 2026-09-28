"""프롬프트 로딩·조립.

프롬프트는 코드가 아니라 **콘텐츠**라서 `app/prompts/*.md`에 따로 둔다. 문구를 고칠 때
파이썬을 건드리지 않아도 되고, 시스템 프롬프트가 호출마다 바이트 단위로 동일해야
프롬프트 캐시가 붙는다는 제약과도 맞다(난이도 가이드를 시스템에 전부 넣고, 요청마다
달라지는 값은 user 쪽에만 둔 이유).
"""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.adapters.glossary.seed import load_glossary
from app.domain.models import CATEGORY_LABELS, LEVEL_LABELS, GameState, RelayEvent, Situation

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

CARD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["title", "body"],
    "additionalProperties": False,
}


@lru_cache(maxsize=8)
def system_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _glossary() -> dict:
    return load_glossary()


def definition_line(term_id: str) -> Optional[str]:
    """용어 사전 정의 — 모델이 규칙을 기억에 기대 거꾸로 말하지 않게 하는 근거.

    실측(2026-09-28): 근거 없이 쓰게 하자 Sonnet 5가 낫아웃 성립 조건을 반대로 설명했다.
    """
    entry = _glossary().get(term_id)
    if not entry:
        return None
    parts = [entry.get("standard", ""), entry.get("deep", "")]
    return f"[정의] {entry.get('name', term_id)}: " + " ".join(p for p in parts if p)


def terms_in(text: str, limit: int = 2) -> list[str]:
    """본문에 이름·별칭이 나오는 용어 키 (등장 순)."""
    found = []
    for term_id, entry in _glossary().items():
        words = [entry.get("name", "")] + list(entry.get("aliases", []))
        positions = [text.find(w) for w in words if w and w in text]
        if positions:
            found.append((min(positions), term_id))
    return [t for _, t in sorted(found)][:limit]


def _level_line(level: int) -> str:
    return f"[난이도] {level} · {LEVEL_LABELS.get(level, '입문')}"


def _state_line(state: GameState) -> str:
    # "O1"을 모델이 무사로 읽은 실측 사례가 있어(2026-09-28) 아웃 수를 말로 한 번 더 적는다.
    outs = "무사" if state.outs == 0 else f"{min(state.outs, 3)}아웃"
    return f"[경기] {state.scoreboard_text()} · {outs} · {state.runners_text()}"


def card_user_prompt(situation: Situation, level: int, categories: Sequence[str]) -> str:
    """카드 생성 요청.

    대괄호 태그 형식은 mock LLM이 키 없이 파싱할 수 있게 하려는 것이기도 하다
    (`adapters/llm/mock.py`).
    """
    interests = ", ".join(CATEGORY_LABELS.get(c, c) for c in sorted(categories))
    lines: list[str] = [
        f"[상황] {situation.label}",
        f"[용어] {situation.term_id}",
        _level_line(level),
        f"[관심] {interests}",
        _state_line(situation.state),
        f"[중계 원문] {situation.trigger_text}",
    ]
    definition = definition_line(situation.term_id)
    if definition:
        lines.append(definition)
    if situation.state.batter:
        lines.append(f"[타자] {situation.state.batter}")
    if situation.state.pitcher:
        lines.append(f"[투수] {situation.state.pitcher}")
    lines.append("")
    lines.append("위 장면을 설명하는 카드를 만들어라.")
    return "\n".join(lines)


def chat_user_prompt(
    question: str,
    state: GameState,
    level: int,
    recent_events: Sequence[RelayEvent],
    recent_labels: Optional[Sequence[str]] = None,
    video_lines: Optional[Sequence[str]] = None,
) -> str:
    lines: list[str] = [
        f"[질문] {question.strip()}",
        _level_line(level),
        _state_line(state),
    ]
    if recent_events:
        lines.append("[최근 중계]")
        for event in recent_events:
            lines.append(f"- ({event.inning}회{event.half_label}) {event.text}")
    if recent_labels:
        lines.append(f"[방금 설명한 상황] {', '.join(recent_labels)}")
    context_text = question + " " + " ".join(e.text for e in recent_events)
    for term_id in terms_in(context_text):
        definition = definition_line(term_id)
        if definition:
            lines.append(definition)
    if video_lines:
        # VLM 관찰은 판정 근거가 아니다 — 화면 묘사 참고용으로만 준다.
        lines.append("[영상 장면(자동 관찰, 틀릴 수 있음)]")
        lines.extend(f"- {v}" for v in video_lines)
    lines.append("")
    lines.append("위 정보만 근거로 질문에 답하라.")
    return "\n".join(lines)


def matchup_user_prompt(state: GameState, facts: Sequence[str]) -> str:
    lines = [
        "[요청] 아래 기록을 근거로 이 타석에 대한 한 줄 해석을 써라.",
        _state_line(state),
        "[기록]",
    ]
    lines.extend(f"- {fact}" for fact in facts)
    lines.append("")
    lines.append("한 문장, 40자 이내. 기록에 없는 내용을 덧붙이지 마라.")
    return "\n".join(lines)
