"""문자중계 소스 어댑터 경계.

네이버 문자중계는 스키마를 실측해 `naver.py`에서 fixture 형식으로 변환한다(오프라인 임포트).
새 소스를 붙일 때도 외부 필드명이 이 Protocol 밖으로 새지 않게 반드시 여기서
`RelayEvent`/`GameMeta`로 변환할 것. 상위 레이어는 내부 모델만 안다.
"""

from typing import Protocol

from app.domain.models import GameFeed, GameMeta


class RelaySourceError(RuntimeError):
    """중계 소스 조회 실패. 상위에서 폴백 판단에 쓴다."""


class RelaySource(Protocol):
    """문자중계 공급자. 구현체: FixtureRelaySource (네이버 변환본도 이걸로 읽는다)."""

    def list_games(self) -> list[GameMeta]:
        ...

    def load(self, game_id: str) -> GameFeed:
        ...
