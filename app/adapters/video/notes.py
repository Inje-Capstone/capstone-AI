"""영상 분석 스냅샷 읽기 — `scripts/analyze_video.py`가 만든 `data/video/{game}.json`.

서버는 영상을 보지 않는다. 미리 뽑아 둔 VLM 관찰만 읽어 카드 근거·챗봇 맥락에 붙인다.
파일이 없거나 깨졌으면 빈 목록 — 영상 관찰은 있으면 좋은 보조 신호일 뿐이다.
"""

import json
import logging
from pathlib import Path

from app.adapters.video.base import VideoEvent

log = logging.getLogger(__name__)


class VideoNotes:
    def __init__(self, video_dir: Path) -> None:
        self.video_dir = Path(video_dir)
        self._cache: dict[str, tuple[float, list[VideoEvent]]] = {}

    def events(self, game_id: str) -> list[VideoEvent]:
        path = self.video_dir / f"{game_id}.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []
        cached = self._cache.get(game_id)
        if cached and cached[0] == mtime:  # 분석 스크립트가 다시 돌면 자동으로 새로 읽는다
            return cached[1]
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            events = sorted(
                (VideoEvent(**e) for e in raw.get("events", [])), key=lambda e: e.t_start
            )
        except (OSError, ValueError, TypeError) as exc:
            log.warning("영상 관찰 스냅샷 무시(%s): %s", path, exc)
            events = []
        self._cache[game_id] = (mtime, events)
        return events


def near(
    events: list[VideoEvent], video_t: float, before: float = 5.0, after: float = 15.0
) -> list[VideoEvent]:
    """중계 시각(영상 기준) 전후 창 안의 관찰. 중계 기록 시각은 실제 장면보다 조금 늦다."""
    return [e for e in events if video_t - before <= e.t_start <= video_t + after]


def recent(events: list[VideoEvent], video_t: float, window: float = 90.0) -> list[VideoEvent]:
    """챗봇 "방금 저거"용 — 질문 시점 직전 창 안의 관찰."""
    return [e for e in events if video_t - window <= e.t_start <= video_t]


def clock(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def evidence_line(event: VideoEvent) -> str:
    what = event.description or event.event_type
    return f"영상 관찰({clock(event.t_start)}, {event.event_type}): {what} — VLM 관찰이며 판정 아님"
