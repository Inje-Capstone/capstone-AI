"""이닝 시청 분석 — 사용자가 본 영상 구간이 어느 이닝(반 이닝)을 덮었나. 순수 로직(I/O 없음).

포인트(이닝당 1점·하루 상한)는 백엔드가 계산한다. 여기서는 '몇 이닝을 봤나'만 센다.

- 반 이닝 구간: 그 반 이닝 첫 중계 이벤트부터 다음 반 이닝 첫 이벤트 전까지(영상 시각). 마지막 반
  이닝은 영상 끝(없으면 마지막 이벤트 + LAST_HALF_SEC)까지.
- 반 이닝을 '봤다': 그 구간의 절반 이상을 봤을 때(보다 만 반 이닝은 안 센다).
- 이닝 수는 올림: 초·말 중 하나라도 봤으면 그 이닝을 센다. 5회초에 우천 취소된 경기도 5회초를
  봤으면 5회로 인정된다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from app.domain.models import RelayEvent
from app.domain.timeline import Timeline

LAST_HALF_SEC = 900  # 영상 길이를 모를 때 마지막 반 이닝 길이(15분)
WATCHED_RATIO = 0.5  # 반 이닝 구간 중 이만큼 이상 봐야 본 것으로


@dataclass(frozen=True)
class HalfSpan:
    inning: int
    half: str  # "top" | "bot"
    start: int  # 영상 초
    end: int  # 영상 초(포함 안 함)


@dataclass(frozen=True)
class HalfWatch:
    span: HalfSpan
    watched_sec: int
    counted: bool


def half_spans(
    events: Sequence[RelayEvent], timeline: Timeline, video_end: Optional[int] = None
) -> list[HalfSpan]:
    """중계 이벤트로 반 이닝마다 영상 구간을 만든다(이벤트 순서대로)."""
    starts: list[tuple[int, str, int]] = []
    for e in sorted(events, key=lambda e: e.t):
        key = (e.inning, e.half)
        if not starts or (starts[-1][0], starts[-1][1]) != key:
            starts.append((e.inning, e.half, timeline.to_video(e.t)))
    if not starts:
        return []
    last_event = timeline.to_video(max(e.t for e in events))
    tail = video_end if video_end is not None and video_end > last_event else (
        last_event + LAST_HALF_SEC)
    spans = []
    for i, (inning, half, start) in enumerate(starts):
        end = starts[i + 1][2] if i + 1 < len(starts) else tail
        if end > start:
            spans.append(HalfSpan(inning, half, start, end))
    return spans


def _merge(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted((int(a), int(b)) for a, b in ranges if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def watch(spans: Sequence[HalfSpan], ranges: Sequence[tuple[int, int]]) -> list[HalfWatch]:
    """본 구간(영상 초, [시작, 끝))이 반 이닝마다 몇 초를 덮었나. 겹친 구간은 한 번만 센다."""
    merged = _merge(ranges)
    out = []
    for s in spans:
        sec = sum(max(0, min(b, s.end) - max(a, s.start)) for a, b in merged)
        out.append(HalfWatch(s, sec, sec >= WATCHED_RATIO * (s.end - s.start)))
    return out


def innings_watched(halves: Sequence[HalfWatch]) -> list[int]:
    """본 이닝 번호(올림 — 초·말 중 하나라도 봤으면 그 이닝)."""
    return sorted({h.span.inning for h in halves if h.counted})
