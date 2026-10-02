"""실시간 모드 — 영상 분석 결과가 들어오는 대로 판정하고 화면에 밀어 준다.

분석기(scripts/live_video.py)가 10초 남짓한 청크를 VLM에 보내고, 결과(점수판·장면·해설)를
서버의 ingest로 보낸다. 서버는 쌓인 판독 전체로 다시 판정한다(수백 개 수준이라 매번 다시 해도 싸다).
그래서 청크가 순서 없이 와도 결과가 같다.

실시간 경기는 LiveRelaySource를 통해 일반 경기와 똑같이 보인다 — 스코어보드·카드·한 줄 요약·
챗봇 엔드포인트를 그대로 쓴다. 새로 생긴 상황은 /api/live/{id}/stream(SSE)으로 알린다.
"""

import threading
import time
from typing import Any, Optional

from app.adapters.relay.base import RelaySource, RelaySourceError
from app.adapters.video.base import EVENT_TYPES
from app.domain.detectors import detect
from app.domain.game_state import replay
from app.domain.models import GameFeed, GameMeta, Situation
from app.domain.video_judge import from_snapshot, judge, to_relay_events


class LiveGame:
    def __init__(self, meta: GameMeta) -> None:
        self.meta = meta
        self.scoreboard: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.speech: list[dict[str, Any]] = []
        self.feed = GameFeed(meta=meta, events=[])
        self.updates: list[dict[str, Any]] = []  # SSE로 보낼 것 (순서대로 쌓기만 한다)
        self.ended = False
        self.started_at = time.time()
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def ingest(self, analysis: dict[str, Any]) -> list[Situation]:
        """청크 하나의 분석 결과를 더하고, 새로 생긴 상황을 돌려준다."""
        with self._lock:
            self.scoreboard += [r for r in analysis.get("scoreboard") or [] if "t" in r]
            self.events += [e for e in analysis.get("events") or []
                            if e.get("event_type") in EVENT_TYPES and "t_start" in e]
            self.speech += [s for s in analysis.get("speech") or [] if "t" in s]
            readings, cues = from_snapshot({"scoreboard": self.scoreboard, "events": self.events,
                                            "speech": self.speech})
            events = to_relay_events(judge(readings, cues))
            last = max((r.t for r in readings), default=0)
            meta = self.meta.model_copy(update={"video_duration_sec": int(last)})
            self.feed = GameFeed(meta=meta, events=events)
            fresh = [s for s in detect(self.feed) if s.id not in self._seen]
            self._seen.update(s.id for s in fresh)
            state = replay(events, meta.away_team, meta.home_team)
            self.updates.append({
                "type": "update", "t": int(last), "scoreboard": state.scoreboard_text(),
                "situations": [
                    {"id": s.id, "t": s.t, "rule_id": s.rule_id, "term_id": s.term_id,
                     "label": s.label, "category": s.category} for s in fresh
                ],
            })
            return fresh

    def end(self) -> None:
        with self._lock:
            self.ended = True
            self.updates.append({"type": "end"})


class LiveRelaySource:
    """실시간 경기를 먼저 보고, 없으면 원래 소스(파일)로 넘긴다."""

    def __init__(self, base: RelaySource) -> None:
        self.base = base
        self.games: dict[str, LiveGame] = {}

    def start(self, meta: GameMeta) -> LiveGame:
        game = LiveGame(meta)
        self.games[meta.id] = game
        return game

    def get(self, game_id: str) -> Optional[LiveGame]:
        return self.games.get(game_id)

    def list_games(self) -> list[GameMeta]:
        live = [g.feed.meta for g in self.games.values()]
        try:
            rest = [m for m in self.base.list_games() if m.id not in self.games]
        except RelaySourceError:
            rest = []
        return live + rest

    def load(self, game_id: str) -> GameFeed:
        game = self.games.get(game_id)
        return game.feed if game is not None else self.base.load(game_id)
