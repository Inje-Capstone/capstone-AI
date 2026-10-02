"""실시간 모드 — 영상 분석 결과와 경기 데이터가 들어오는 대로 판정·보강하고 화면에 밀어 준다.

- 영상: 분석기(scripts/live_video.py)가 몇 초짜리 청크를 VLM에 보내 결과(점수판·장면·해설)를
  ingest로 보낸다. 서버는 쌓인 판독 전체로 다시 판정한다(순서가 섞여도 결과가 같다).
- 데이터: 분석기가 경기 중 네이버 중계를 주기적으로 받아 data로 보낸다. 선수 이름·구종·기록만
  영상 판정에 붙인다(enrich). 데이터의 플레이 결과는 쓰지 않는다.
- 영상↔데이터 시각 차이는 쌓인 장면 단서로 추정하고, 확신이 서면 고정한다.
  그 전까지는 영상 판정만 보인다(선수 이름 없이).

실시간 경기는 LiveRelaySource를 통해 일반 경기와 똑같이 보인다 — 스코어보드·카드·한 줄 요약·
챗봇 엔드포인트를 그대로 쓴다. 새로 생긴 상황은 /api/live/{id}/stream(SSE)으로 알린다.
"""

import threading
import time
from typing import Any, Optional

from app.adapters.relay.base import RelaySource, RelaySourceError
from app.adapters.video.base import EVENT_TYPES
from app.domain.detectors import detect
from app.domain.enrich import enrich
from app.domain.game_state import replay
from app.domain.models import GameFeed, GameMeta, RelayEvent, Situation
from app.domain.scorebug import stabilize
from app.domain.video_judge import from_snapshot, judge, to_relay_events
from app.domain.video_sync import Anchor, estimate_offset


class LiveGame:
    def __init__(self, meta: GameMeta) -> None:
        self.meta = meta
        self.scoreboard: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.speech: list[dict[str, Any]] = []
        self.data_events: list[RelayEvent] = []
        self.offset: Optional[float] = None  # 영상 t − 데이터 t, 확신이 서면 고정
        self.feed = GameFeed(meta=meta, events=[])
        self.updates: list[dict[str, Any]] = []  # SSE로 보낼 것 (순서대로 쌓기만 한다)
        self.ended = False
        self.started_at = time.time()
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    # ── 입력 ────────────────────────────────────────────────────────────
    def ingest(self, analysis: dict[str, Any]) -> list[Situation]:
        """청크 하나의 영상 분석 결과를 더하고, 새로 생긴 상황을 돌려준다."""
        with self._lock:
            self.scoreboard += [r for r in analysis.get("scoreboard") or [] if "t" in r]
            self.events += [e for e in analysis.get("events") or []
                            if e.get("event_type") in EVENT_TYPES and "t_start" in e]
            self.speech += [s for s in analysis.get("speech") or [] if "t" in s]
            return self._rebuild()

    def set_data(
        self, events: list[dict[str, Any]], context: Optional[dict] = None
    ) -> list[Situation]:
        """경기 데이터(네이버 중계를 변환한 이벤트 전체)를 바꿔 끼운다 — 매번 전체를 보낸다."""
        with self._lock:
            self.data_events = [RelayEvent(**e) for e in events]
            if context:
                self.meta = self.meta.model_copy(update={"context": context})
            return self._rebuild()

    # ── 내부 ────────────────────────────────────────────────────────────
    def _estimate_offset(self) -> None:
        if self.offset is not None or not self.data_events:
            return
        anchors = [Anchor(float(e["t_start"]), e["event_type"]) for e in self.events]
        est = estimate_offset(self.data_events, anchors)
        if est is not None and est.confident:
            self.offset = est.offset

    def _rebuild(self) -> list[Situation]:
        readings, cues = from_snapshot({"scoreboard": self.scoreboard, "events": self.events,
                                        "speech": self.speech})
        events = to_relay_events(judge(readings, cues))
        self._estimate_offset()
        if self.offset is not None and self.data_events:
            events = enrich(events, self.data_events, self.offset, stabilize(readings))
        last = max((r.t for r in readings), default=0)
        meta = self.meta.model_copy(update={"video_duration_sec": int(last)})
        self.feed = GameFeed(meta=meta, events=events)
        fresh = [s for s in detect(self.feed) if s.id not in self._seen]
        self._seen.update(s.id for s in fresh)
        state = replay(events, meta.away_team, meta.home_team)
        self.updates.append({
            "type": "update", "t": int(last), "scoreboard": state.scoreboard_text(),
            "batter": state.batter, "pitcher": state.pitcher,
            "data": self.offset is not None and bool(self.data_events),
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
