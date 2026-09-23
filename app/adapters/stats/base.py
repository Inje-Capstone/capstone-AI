"""선수 기록 소스 어댑터 경계.

⚠️ TODO(스키마 미검증): KBO 공식 기록·STATIZ 등 실제 소스는 아직 조사 전이다.
기록은 **조회하는 값이지 생성하는 값이 아니다** — 이 경계 안에서 지어내면 안 된다.
"""

from typing import Optional, Protocol

from pydantic import BaseModel


class StatsSourceError(RuntimeError):
    """기록 조회 실패. 분석 패널만 '불러올 수 없어요'로 내려가고 화면은 산다."""


class BatterFacts(BaseModel):
    batter: str
    avg: Optional[str] = None
    recent: Optional[str] = None
    vs_pitcher: Optional[str] = None
    team_form: Optional[str] = None

    def as_lines(self) -> list[str]:
        """LLM 한 줄 해석에 넘길 근거. 없는 항목은 아예 넘기지 않는다."""
        lines = []
        if self.avg:
            lines.append(f"{self.batter} 시즌 타율 {self.avg}")
        if self.recent:
            lines.append(f"최근 흐름: {self.recent}")
        if self.vs_pitcher:
            lines.append(f"상대 전적: {self.vs_pitcher}")
        if self.team_form:
            lines.append(f"팀 최근 성적: {self.team_form}")
        return lines


class StatsSource(Protocol):
    def batter_facts(self, batter: str, pitcher: Optional[str], team: str) -> BatterFacts:
        ...
