"""내부 도메인 모델.

외부 소스(문자중계·기록 API)의 필드명이 이 레이어 위로 새지 않게, 어댑터에서
반드시 이 모델로 변환해 넣는다. 상위 레이어(services/api)는 내부 모델만 안다.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

# ── 관심 카테고리 (온보딩 2단계 선택지와 1:1) ────────────────────────────
CATEGORY_BASIC = "basic_rules"  # 기본 룰
CATEGORY_PITCHING = "pitching"  # 구종 · 투구
CATEGORY_TACTICS = "tactics"  # 전술 · 기록
CATEGORY_CULTURE = "culture"  # 응원 문화

ALL_CATEGORIES = (CATEGORY_BASIC, CATEGORY_PITCHING, CATEGORY_TACTICS, CATEGORY_CULTURE)

CATEGORY_LABELS = {
    CATEGORY_BASIC: "기본 룰",
    CATEGORY_PITCHING: "구종 · 투구",
    CATEGORY_TACTICS: "전술 · 기록",
    CATEGORY_CULTURE: "응원 문화",
}

# ── 지식 수준 (온보딩 1단계) ────────────────────────────────────────────
LEVEL_BEGINNER = 0  # 입문 — 오늘 처음이에요
LEVEL_NOVICE = 1  # 초보 — 몇 번 봤어요
LEVEL_FAMILIAR = 2  # 익숙 — 룰은 알아요

LEVEL_LABELS = {
    LEVEL_BEGINNER: "입문",
    LEVEL_NOVICE: "초보",
    LEVEL_FAMILIAR: "익숙",
}

# ── 중계 이벤트 ─────────────────────────────────────────────────────────
KIND_ATBAT = "atbat"  # 타석 시작
KIND_PITCH = "pitch"  # 투구 1구
KIND_CALL = "call"  # 심판 콜 (보크·인필드플라이·낫아웃 등)
KIND_RESULT = "result"  # 타석 결과
KIND_STEAL = "steal"  # 도루


class RelayEvent(BaseModel):
    """문자중계 1건. `t`는 중계 기준 경과 초이며, 영상 타임코드 변환은 timeline이 맡는다."""

    id: str
    t: int
    inning: int
    half: str  # "top" | "bot"
    kind: str
    text: str
    batter: Optional[str] = None
    pitcher: Optional[str] = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def half_label(self) -> str:
        return "초" if self.half == "top" else "말"


class GameMeta(BaseModel):
    """경기 메타. 홈 화면(S3) 카드와 시청 화면 진입 조건에 쓴다."""

    id: str
    date: str
    stadium: str
    away_team: str
    home_team: str
    has_video: bool = False  # False면 홈 리스트에서 비활성 (user-flow F3)
    unavailable_reason: Optional[str] = None
    video_duration_sec: int = 0
    relay_video_offset_sec: int = 0
    final_away: Optional[int] = None
    final_home: Optional[int] = None

    def title(self) -> str:
        return f"{self.away_team} vs {self.home_team}"


class GameFeed(BaseModel):
    """한 경기의 메타 + 중계 이벤트 전체."""

    meta: GameMeta
    events: list["RelayEvent"] = Field(default_factory=list)


class GameState(BaseModel):
    """특정 시점의 경기 상태. 상황 감지와 챗봇 컨텍스트의 공통 기반."""

    away_team: str
    home_team: str
    inning: int = 1
    half: str = "top"
    balls: int = 0
    strikes: int = 0
    outs: int = 0
    bases: tuple[bool, bool, bool] = (False, False, False)  # 1루, 2루, 3루
    away_score: int = 0
    home_score: int = 0
    batter: Optional[str] = None
    pitcher: Optional[str] = None
    game_over: bool = False
    last_event_id: Optional[str] = None

    @property
    def batting_team(self) -> str:
        return self.away_team if self.half == "top" else self.home_team

    @property
    def fielding_team(self) -> str:
        return self.home_team if self.half == "top" else self.away_team

    @property
    def half_label(self) -> str:
        return "초" if self.half == "top" else "말"

    def runners_text(self) -> str:
        names = [n for n, on in zip(("1루", "2루", "3루"), self.bases) if on]
        if not names:
            return "주자 없음"
        if len(names) == 3:
            return "만루"
        return "주자 " + "·".join(names)

    def scoreboard_text(self) -> str:
        """와이어프레임 S4 좌하단 표기. 예: `두산 3 : 5 LG · 7회말 · B2 S1 O2`

        볼넷·삼진 투구 직후엔 내부 카운트가 B4·S3이 되지만, 전광판에 그런 카운트는 없다.
        표기는 B3·S2에서 멈춘다 — "2스트라이크에서 던진 결정구"로 읽혀야 입문자가 안 헷갈린다.
        """
        return (
            f"{self.away_team} {self.away_score} : {self.home_score} {self.home_team}"
            f" · {self.inning}회{self.half_label}"
            f" · B{min(self.balls, 3)} S{min(self.strikes, 2)} O{self.outs}"
        )


# ── 감지된 상황 / 설명 카드 ─────────────────────────────────────────────
class Situation(BaseModel):
    """감지기가 찾아낸 '설명할 거리' 하나. 아직 문장은 없다."""

    id: str
    t: int
    rule_id: str
    term_id: str
    category: str
    label: str  # 사람이 읽는 상황 이름 ("보크")
    trigger_text: str  # 근거가 된 중계 원문
    event_ids: list[str]
    state: GameState
    priority: float = 0.5  # 규칙 고유 중요도. 프로필 가중치의 밑값.
    reasons: list[str] = Field(default_factory=list)


class Card(BaseModel):
    """AI 설명 카드. 와이어프레임 S4 우상단 패널 1장."""

    id: str
    t: int
    situation_id: str
    rule_id: str
    term_id: str  # 용어 사전(S5) 딥링크 대상
    category: str
    level: int
    title: str
    body: str
    reasons: list[str] = Field(default_factory=list)  # core-belief 4: 근거 없는 노출 금지
    source: str = "llm"  # llm | snapshot | mock


class ExplainProfile(BaseModel):
    """온보딩 답변을 '분기'가 아니라 '가중치'로 바꾼 것 (core-belief 3).

    난이도 × 카테고리 조합마다 if문을 두지 않고, 규칙별 가중치를 계산해
    임계값으로 거른다. 조합이 늘어나도 코드 경로는 하나다.
    """

    level: int = LEVEL_BEGINNER
    categories: frozenset[str] = frozenset(ALL_CATEGORIES)
    threshold: float = 0.5

    @property
    def level_label(self) -> str:
        return LEVEL_LABELS.get(self.level, "입문")

    def key(self) -> str:
        return f"L{self.level}:{','.join(sorted(self.categories))}:{self.threshold}"


class Matchup(BaseModel):
    """S4 우측 하단 선수·매치업 분석 패널. 기록은 조회한 값이고, ai_note만 생성물이다."""

    batter: str
    batter_line: str
    recent_form: str
    vs_pitcher: str
    team_form: str
    ai_note: Optional[str] = None
    source: str = "fixture"  # fixture(가상 기록) | relay(중계에 실린 실제 기록)
    available: bool = True
    unavailable_reason: Optional[str] = None
