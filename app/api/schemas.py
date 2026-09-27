"""프론트/백엔드에 넘기는 API 계약.

내부 모델을 그대로 노출하지 않고 여기서 한 번 걸러 낸다. 필드 이름과 모양이
와이어프레임 S3·S4·S5의 화면 요소와 1:1로 대응하도록 잡았다.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.domain.models import Card, GameMeta, GameState, Matchup


class GameOut(BaseModel):
    """S3 홈 경기 카드."""

    id: str
    date: str
    stadium: str
    away_team: str
    home_team: str
    score_text: Optional[str] = None
    has_video: bool
    unavailable_reason: Optional[str] = None
    video_duration_sec: int

    @classmethod
    def of(cls, meta: GameMeta) -> "GameOut":
        score = None
        if meta.final_away is not None and meta.final_home is not None:
            score = f"{meta.away_team} {meta.final_away} : {meta.final_home} {meta.home_team}"
        return cls(
            id=meta.id,
            date=meta.date,
            stadium=meta.stadium,
            away_team=meta.away_team,
            home_team=meta.home_team,
            score_text=score,
            has_video=meta.has_video,
            unavailable_reason=meta.unavailable_reason
            or (None if meta.has_video else "영상 미확보"),
            video_duration_sec=meta.video_duration_sec,
        )


class StateOut(BaseModel):
    """S4 좌측 하단 스코어보드."""

    scoreboard: str
    inning: int
    half: str
    balls: int
    strikes: int
    outs: int
    bases: list[bool]
    runners_text: str
    away_team: str
    home_team: str
    away_score: int
    home_score: int
    batter: Optional[str]
    pitcher: Optional[str]
    game_over: bool

    @classmethod
    def of(cls, state: GameState) -> "StateOut":
        return cls(
            scoreboard=state.scoreboard_text(),
            inning=state.inning,
            half=state.half,
            balls=state.balls,
            strikes=state.strikes,
            outs=state.outs,
            bases=list(state.bases),
            runners_text=state.runners_text(),
            away_team=state.away_team,
            home_team=state.home_team,
            away_score=state.away_score,
            home_score=state.home_score,
            batter=state.batter,
            pitcher=state.pitcher,
            game_over=state.game_over,
        )


class CardOut(BaseModel):
    """S4 우측 상단 AI 설명 카드 1장."""

    id: str
    t: int
    title: str
    body: str
    term_id: str = Field(description="용어 사전(S5) 딥링크 키")
    category: str
    rule_id: str
    level: int
    source: str = Field(description="llm | mock | snapshot — 시연 여부를 화면에서 밝히기 위한 값")
    can_simplify: bool
    reasons: list[str] = Field(description="이 카드가 왜 떴는지 (core-belief 4)")

    @classmethod
    def of(cls, card: Card) -> "CardOut":
        return cls(
            id=card.id,
            t=card.t,
            title=card.title,
            body=card.body,
            term_id=card.term_id,
            category=card.category,
            rule_id=card.rule_id,
            level=card.level,
            source=card.source,
            can_simplify=card.level > 0,
            reasons=card.reasons,
        )


class CardsOut(BaseModel):
    game_id: str
    t: Optional[int]
    level: int
    cards: list[CardOut]
    degraded: bool = Field(
        default=False,
        description="True면 설명 생성이 실패해 카드가 비었다는 뜻. 화면은 정상 유지한다.",
    )


class MatchupOut(BaseModel):
    """S4 우측 하단 선수·매치업 분석."""

    available: bool
    unavailable_reason: Optional[str] = None
    batter: Optional[str] = None
    batter_line: Optional[str] = None
    recent_form: Optional[str] = None
    vs_pitcher: Optional[str] = None
    team_form: Optional[str] = None
    ai_note: Optional[str] = None
    source: Optional[str] = Field(
        default=None, description="fixture(가상 기록) | relay(중계에 실린 실제 기록)"
    )

    @classmethod
    def of(cls, matchup: Matchup) -> "MatchupOut":
        if not matchup.available:
            return cls(available=False, unavailable_reason=matchup.unavailable_reason)
        return cls(
            available=True,
            batter=matchup.batter,
            batter_line=matchup.batter_line,
            recent_form=matchup.recent_form,
            vs_pitcher=matchup.vs_pitcher,
            team_form=matchup.team_form,
            ai_note=matchup.ai_note,
            source=matchup.source,
        )


class ChatIn(BaseModel):
    """S4 하단 챗봇 입력."""

    question: str = Field(min_length=1, max_length=500)
    t: Optional[int] = Field(default=None, description="영상 타임코드(초). 없으면 경기 전체 맥락.")
    level: Optional[str] = Field(default=None, description="입문 | 초보 | 익숙")
    categories: Optional[list[str]] = None


class ChatOut(BaseModel):
    answer: str
    ok: bool
    source: str
    context: str = Field(description="답변에 쓴 경기 상황 요약 — 근거 표시용")
    used_event_ids: list[str]


class TermSummaryOut(BaseModel):
    """S5 용어 사전 목록 1행 / 관련 용어 칩."""

    id: str = Field(description="카드의 term_id와 같은 키 — 딥링크 대상")
    name: str
    category: str
    category_label: str
    aliases: list[str]
    summary: str = Field(description="입문 난이도 한 줄 정의")

    @classmethod
    def of(cls, entry: dict[str, Any], category_label: str) -> "TermSummaryOut":
        return cls(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            category=entry.get("category", ""),
            category_label=category_label,
            aliases=entry.get("aliases", []),
            summary=entry.get("easy", ""),
        )


class TermOut(TermSummaryOut):
    """S5 용어 상세."""

    level: int
    body: str = Field(description="요청한 난이도의 설명")
    levels: dict[str, str] = Field(description="난이도 토글용: 입문 | 초보 | 익숙 → 설명")
    related: list[TermSummaryOut]
