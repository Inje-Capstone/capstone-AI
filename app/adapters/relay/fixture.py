"""파일 기반 중계 소스. `data/fixtures/*.json`을 읽는다.

실데이터 소스가 확보되기 전까지 파이프라인 전체를 돌리는 유일한 공급자이고,
확보된 뒤에도 테스트·오프라인 데모의 기준 소스로 남는다.
"""

import json
from pathlib import Path
from typing import Optional

from app.adapters.relay.base import RelaySourceError
from app.domain.models import GameFeed, GameMeta, RelayEvent

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "fixtures"


class FixtureRelaySource:
    def __init__(self, fixture_dir: Path = DEFAULT_FIXTURE_DIR) -> None:
        self.fixture_dir = Path(fixture_dir)
        self._cache: dict[str, GameFeed] = {}

    # ── RelaySource 구현 ────────────────────────────────────────────────
    def list_games(self) -> list[GameMeta]:
        return sorted(
            (feed.meta for feed in self._load_all().values()),
            key=lambda m: m.date,
            reverse=True,
        )

    def load(self, game_id: str) -> GameFeed:
        feeds = self._load_all()
        if game_id not in feeds:
            raise RelaySourceError(f"중계 데이터가 없는 경기: {game_id}")
        return feeds[game_id]

    # ── 내부 ────────────────────────────────────────────────────────────
    def _load_all(self) -> dict[str, GameFeed]:
        if self._cache:
            return self._cache
        if not self.fixture_dir.is_dir():
            raise RelaySourceError(f"fixture 디렉터리를 찾을 수 없다: {self.fixture_dir}")
        for path in sorted(self.fixture_dir.glob("*.json")):
            feed = _parse_fixture(path)
            if feed is None:  # 경기 파일이 아니면 조용히 건너뛴다
                continue
            self._cache[feed.meta.id] = feed
        if not self._cache:
            raise RelaySourceError(f"fixture가 하나도 없다: {self.fixture_dir}")
        return self._cache


def _parse_fixture(path: Path) -> Optional[GameFeed]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelaySourceError(f"fixture 읽기 실패: {path} ({exc})") from exc

    game = raw.get("game")
    if not game or "id" not in game:
        return None  # 같은 디렉터리에 놓인 다른 용도의 JSON
    final = game.get("final_score") or {}
    meta = GameMeta(
        id=game["id"],
        date=game.get("date", ""),
        stadium=game.get("stadium", ""),
        away_team=game["away_team"],
        home_team=game["home_team"],
        has_video=bool(game.get("has_video", False)),
        unavailable_reason=game.get("unavailable_reason"),
        video_duration_sec=int(game.get("video_duration_sec", 0)),
        relay_video_offset_sec=int(game.get("relay_video_offset_sec", 0)),
        final_away=final.get("away"),
        final_home=final.get("home"),
    )
    events = [RelayEvent(**e) for e in raw.get("events", [])]
    events.sort(key=lambda e: e.t)
    return GameFeed(meta=meta, events=events)
