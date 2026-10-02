"""영상-중계 시각 정합. 순수 로직(I/O 없음).

중계 이벤트 `t`는 첫 투구 기준 초이고, 영상에서는 방송 시작·광고 때문에 그만큼 밀려
있다. 영상 분석이 찾은 장면(타격·홈런·삼진·투수 교체…)과 같은 종류의 중계 이벤트를
짝지어, 가장 많은 쌍을 설명하는 하나의 오프셋을 고른다(투표).

VLM 관찰은 틀릴 수 있다 — 그래서 한 쌍을 믿지 않고 다수결로 정하고, 표가 모자라면
`confident=False`로 돌려 수동 오프셋을 유지하게 한다.
"""

from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Optional

from app.domain.game_state import KIND_SUB
from app.domain.models import KIND_PITCH, KIND_RESULT, KIND_STEAL, RelayEvent

MIN_MATCHES = 5
MIN_MATCH_RATIO = 0.3


@dataclass(frozen=True)
class Anchor:
    t: float
    kind: str


@dataclass(frozen=True)
class OffsetEstimate:
    offset: float  # video_t = relay_t + offset
    matched: int
    anchors: int  # 짝을 찾을 수 있었던 영상 관찰 수
    residual: float  # 짝들의 |오차| 중앙값(초)
    confident: bool


def relay_anchors(events: Iterable[RelayEvent]) -> list[Anchor]:
    """중계 이벤트 → 영상에서 보일 법한 장면 종류."""
    out = []
    for e in events:
        detail = e.detail
        kind: Optional[str] = None
        # 이름은 영상 단서 어휘(adapters/video/base.EVENT_TYPES)에 맞춘다
        if e.kind == KIND_PITCH and detail.get("result") == "in_play":
            kind = "contact"
        elif e.kind == KIND_PITCH and detail.get("result") == "swing_strike":
            kind = "swing_miss"
        elif e.kind == KIND_RESULT and detail.get("result") == "homerun":
            kind = "ball_over_fence"
        elif e.kind == KIND_STEAL:
            kind = "slide"
        elif e.kind == KIND_SUB and detail.get("sub_type") == "pitcher":
            kind = "pitching_change"
        elif e.kind == "note":
            if "마운드 방문" in e.text:
                kind = "mound_visit"
            elif "비디오 판독" in e.text:
                kind = "replay_review"
        if kind:
            out.append(Anchor(float(e.t), kind))
    return out


def _by_kind(anchors: Iterable[Anchor]) -> dict[str, list[float]]:
    table: dict[str, list[float]] = {}
    for a in anchors:
        table.setdefault(a.kind, []).append(a.t)
    for times in table.values():
        times.sort()
    return table


def _nearest(times: Sequence[float], t: float) -> Optional[float]:
    i = bisect_left(times, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(times) and (best is None or abs(times[j] - t) < abs(best - t)):
            best = times[j]
    return best


def _pairs(
    video: Sequence[Anchor], relay: dict[str, list[float]], offset: float, tol: float
) -> list[float]:
    """오프셋을 적용했을 때 tol 안에 짝이 있는 영상 관찰들의 (영상t - 중계t)."""
    diffs = []
    for v in video:
        times = relay.get(v.kind)
        if not times:
            continue
        near = _nearest(times, v.t - offset)
        if near is not None and abs(v.t - offset - near) <= tol:
            diffs.append(v.t - near)
    return diffs


def estimate_offset(
    relay_events: Sequence[RelayEvent],
    video_anchors: Sequence[Anchor],
    min_offset: float = -600.0,
    max_offset: float = 3 * 3600.0,
    tol: float = 6.0,
) -> Optional[OffsetEstimate]:
    """가장 많은 영상 관찰을 설명하는 오프셋. 후보가 하나도 없으면 None."""
    relay = _by_kind(relay_anchors(relay_events))
    video = [a for a in video_anchors if a.kind in relay]
    if not video:
        return None

    candidates = {
        round(v.t - r)
        for v in video
        for r in relay[v.kind]
        if min_offset <= v.t - r <= max_offset
    }
    if not candidates:
        return None

    best_diffs: list[float] = []
    for cand in sorted(candidates):
        diffs = _pairs(video, relay, cand, tol)
        if len(diffs) > len(best_diffs):
            best_diffs = diffs
    if not best_diffs:
        return None

    refined = median(best_diffs)
    final = _pairs(video, relay, refined, tol) or best_diffs
    residual = median(abs(d - refined) for d in final)
    matched = len(final)
    return OffsetEstimate(
        offset=round(refined, 1),
        matched=matched,
        anchors=len(video),
        residual=round(residual, 2),
        confident=matched >= MIN_MATCHES and matched / len(video) >= MIN_MATCH_RATIO,
    )
