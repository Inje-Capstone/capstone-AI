"""방금 장면 한 줄 요약 (S4, 조사서 2.1.2 신규 요구 "실시간 분석").

리플레이에서는 "방금 장면"이 타임코드의 결정적 함수다 — t 직전의 결과성 중계 이벤트
하나(타석 결과·도루·투수 교체·보크)를 한 줄로 만든다. 카드와 같은 폴백 순서:
메모리 → 디스크 스냅샷(실제 모델 결과만) → 생성 → **조립 문장**(항상 성공).
그래서 LLM이 죽어도 한 줄은 늘 뜬다(`source="template"`).
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from app.adapters.llm.base import LLMClient, LLMError
from app.adapters.relay.base import RelaySource
from app.domain.game_state import KIND_SUB, states_by_event
from app.domain.models import (
    KIND_CALL,
    KIND_RESULT,
    KIND_STEAL,
    LEVEL_LABELS,
    GameFeed,
    GameState,
    RelayEvent,
)
from app.domain.timeline import Timeline
from app.prompt_templates import system_prompt

log = logging.getLogger(__name__)

MAX_CHARS = 60
_MOMENT_CALLS = frozenset({"balk", "wild_pitch", "passed_ball", "infield_fly"})
_QUOTES = "\"'“”‘’"


class Moment(BaseModel):
    event_id: Optional[str]
    t: Optional[int]  # 그 장면의 영상 타임코드
    text: str
    source: str  # llm | snapshot | template
    scoreboard: str


def is_moment(event: RelayEvent) -> bool:
    if event.kind in (KIND_RESULT, KIND_STEAL):
        return True
    if event.kind == KIND_SUB:
        return event.detail.get("sub_type") == "pitcher"
    return event.kind == KIND_CALL and event.detail.get("call") in _MOMENT_CALLS


def template_text(event: RelayEvent, before: GameState, after: GameState) -> str:
    """LLM 없이 만드는 한 줄: 중계 원문 + 경기에 생긴 변화."""
    changes = []
    runs = (after.away_score + after.home_score) - (before.away_score + before.home_score)
    if runs > 0:
        changes.append(
            f"{runs}점 ({after.away_team} {after.away_score}:{after.home_score} {after.home_team})"
        )
    same_half = after.inning == before.inning and after.half == before.half
    if event.kind == KIND_RESULT and not same_half:
        changes.append("공수 교대")
    elif after.outs > before.outs:
        changes.append(f"{after.outs}아웃")
    if after.game_over and not before.game_over:
        changes.append("경기 종료")
    head = f"{event.inning}회{event.half_label} {event.text}".strip()
    return f"{head} → {', '.join(changes)}" if changes else head


def clean_line(text: str) -> str:
    line = re.sub(r"\s+", " ", text).strip().strip(_QUOTES).strip()
    return line[:MAX_CHARS]


class MomentService:
    def __init__(
        self, relay: RelaySource, llm: LLMClient, snapshot_dir: Optional[Path] = None
    ) -> None:
        self.relay = relay
        self.llm = llm
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self._cache: dict[str, Moment] = {}
        self._loaded: set[str] = set()

    def moment(self, game_id: str, video_t: Optional[int], level: int = 0) -> Moment:
        feed = self.relay.load(game_id)
        timeline = Timeline(feed.meta.relay_video_offset_sec)
        relay_t = timeline.to_relay(int(video_t)) if video_t is not None else None
        states = states_by_event(feed.events, feed.meta.away_team, feed.meta.home_team)

        idx = None
        for i, event in enumerate(feed.events):
            if relay_t is not None and event.t > relay_t:
                break
            if is_moment(event):
                idx = i
        if idx is None:
            board = GameState(
                away_team=feed.meta.away_team, home_team=feed.meta.home_team
            ).scoreboard_text()
            return Moment(event_id=None, t=None, text="아직 경기가 시작되지 않았어요.",
                          source="template", scoreboard=board)
        return self._moment_at(feed, states, idx, level, timeline)

    def all(self, game_id: str, level: int = 0) -> list[Moment]:
        """경기 전체 한 줄 목록 — 사전 생성·타임라인용."""
        feed = self.relay.load(game_id)
        timeline = Timeline(feed.meta.relay_video_offset_sec)
        states = states_by_event(feed.events, feed.meta.away_team, feed.meta.home_team)
        return [
            self._moment_at(feed, states, i, level, timeline)
            for i, e in enumerate(feed.events) if is_moment(e)
        ]

    # ── 내부 ────────────────────────────────────────────────────────────
    def _moment_at(
        self, feed: GameFeed, states: list[GameState], idx: int, level: int, timeline: Timeline
    ) -> Moment:
        event = feed.events[idx]
        after = states[idx]
        before = states[idx - 1] if idx > 0 else GameState(
            away_team=feed.meta.away_team, home_team=feed.meta.home_team)
        key = f"{feed.meta.id}:{event.id}@L{level}"
        self._load_snapshot(feed.meta.id)
        if key in self._cache:
            return self._cache[key]

        text, source = template_text(event, before, after), "template"
        if self.llm.source == "llm":
            generated = self._generate(event, before, after, level)
            if generated:
                text, source = generated, "llm"
        moment = Moment(event_id=event.id, t=timeline.to_video(event.t), text=text,
                        source=source, scoreboard=after.scoreboard_text())
        if source == "llm":  # 조립 문장은 캐시하지 않는다 — 모델이 살아나면 다시 생성
            self._cache[key] = moment
            self._save_snapshot(feed.meta.id)
        return moment

    def _generate(
        self, event: RelayEvent, before: GameState, after: GameState, level: int
    ) -> Optional[str]:
        user = "\n".join([
            f"[난이도] {level} · {LEVEL_LABELS.get(level, '입문')}",
            f"[중계] ({event.inning}회{event.half_label}) {event.text}",
            f"[전] {before.scoreboard_text()} · {before.runners_text()}",
            f"[후] {after.scoreboard_text()} · {after.runners_text()}",
            "",
            "방금 장면을 한 줄로 요약하라.",
        ])
        try:
            result = self.llm.complete(
                system=system_prompt("moment_system"), user=user, max_tokens=120,
                effort="low", thinking=False,
            )
        except LLMError as exc:
            log.info("한 줄 요약 생략(%s): %s", event.id, exc)
            return None
        return clean_line(result.text) or None

    def _snapshot_path(self, game_id: str) -> Optional[Path]:
        return self.snapshot_dir / f"{game_id}.moments.json" if self.snapshot_dir else None

    def _load_snapshot(self, game_id: str) -> None:
        if game_id in self._loaded:
            return
        self._loaded.add(game_id)
        path = self._snapshot_path(game_id)
        if not path or not path.exists() or self.llm.source != "llm":
            return  # 실제 모델이 없는 환경에선 생성물 스냅샷을 쓰지 않는다(카드와 같은 규칙)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("한 줄 요약 스냅샷 무시(%s): %s", path, exc)
            return
        for key, raw in (data.get("items") or {}).items():
            self._cache[key] = Moment(**{**raw, "source": "snapshot"})

    def _save_snapshot(self, game_id: str) -> None:
        path = self._snapshot_path(game_id)
        if not path:
            return
        prefix = f"{game_id}:"
        items = {k: v.model_dump() for k, v in self._cache.items() if k.startswith(prefix)}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"model": self.llm.model, "items": items}, ensure_ascii=False,
                           indent=1),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("한 줄 요약 스냅샷 저장 실패(%s): %s", path, exc)
