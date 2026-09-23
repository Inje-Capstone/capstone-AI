"""문자중계 소스 어댑터 경계.

⚠️ TODO(스키마 미검증): 네이버/KBO 문자중계의 실제 응답 스키마는 아직 조사하지 않았다.
새 구현체를 붙일 때 외부 필드명이 이 Protocol 밖으로 새지 않게 반드시 여기서
`RelayEvent`/`GameMeta`로 변환할 것. 상위 레이어는 내부 모델만 안다.
"""

from typing import Protocol

from app.domain.models import GameFeed, GameMeta


class RelaySourceError(RuntimeError):
    """중계 소스 조회 실패. 상위에서 폴백 판단에 쓴다."""


class RelaySource(Protocol):
    """문자중계 공급자. 구현체: FixtureRelaySource (현재) / NaverRelaySource (예정)."""

    def list_games(self) -> list[GameMeta]:
        ...

    def load(self, game_id: str) -> GameFeed:
        ...
