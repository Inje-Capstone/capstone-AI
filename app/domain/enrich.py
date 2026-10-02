"""영상 판정 + 데이터 보강. 순수 로직(I/O 없음).

역할 분담 (2026-10-02 결정):
- **영상**: 플레이 판정 — 보크·도루·주루·아웃·득점 (video_judge)
- **데이터(네이버)**: 선수 이름·라인업·구종·구속·기록 — 영상으로 읽기 어렵고 데이터가 정확한 것
- 데이터의 **플레이 결과**(삼진·볼넷…)는 여기서 쓰지 않는다 — 그건 채점(grading)에만

데이터의 시각은 중계 기준이라 `offset`(영상 t − 중계 t)으로 영상 시각에 맞춘다.
"""

from collections.abc import Sequence
from typing import Optional

from app.domain.game_state import KIND_SUB
from app.domain.models import KIND_ATBAT, KIND_PITCH, KIND_RESULT, RelayEvent
from app.domain.scorebug import Reading

# 같은 초에 겹치면 이 순서로 재생한다 — 타석 시작 → 교체 → 투구 → 주루·콜 → 결과
_RANK = {KIND_ATBAT: 0, KIND_SUB: 1, KIND_PITCH: 2, "call": 3, "steal": 3, KIND_RESULT: 4}
SUB_MATCH_SEC = 90.0


def _half_at(stable: Sequence[Reading], t: float) -> Optional[tuple[int, str]]:
    """그 시각 점수판의 이닝·초말. 데이터 이벤트도 영상 점수판의 이닝을 따른다
    (오프셋이 몇 초 어긋나도 이닝 경계에서 상태가 엉키지 않게)."""
    last = None
    for r in stable:
        if r.t > t:
            break
        last = r
    if last is None and stable:
        last = stable[0]
    return (last.inning, last.half) if last else None


def enrich(
    video_events: Sequence[RelayEvent],
    data_events: Sequence[RelayEvent],
    offset: float,
    stable: Sequence[Reading],
) -> list[RelayEvent]:
    """영상 판정 이벤트에 데이터(타석·선수·구종)를 끼워 넣는다."""
    if not stable:
        return list(video_events)
    start, end = stable[0].t, stable[-1].t

    def vt(e: RelayEvent) -> float:
        return e.t + offset

    atbats = [e for e in data_events if e.kind == KIND_ATBAT and start <= vt(e) <= end]
    pitches = [e for e in data_events if e.kind == KIND_PITCH and e.detail.get("pitch_type")]
    subs = [e for e in data_events
            if e.kind == KIND_SUB and e.detail.get("sub_type") == "pitcher"]

    out: list[RelayEvent] = []
    for a in atbats:
        where = _half_at(stable, vt(a))
        inning, half = where if where else (a.inning, a.half)
        detail = {"source": "data"}
        if a.detail.get("stats"):
            detail["stats"] = a.detail["stats"]
        out.append(RelayEvent(
            id=f"d{a.id}", t=int(vt(a)), inning=inning, half=half, kind=KIND_ATBAT,
            text=f"{a.batter or ''} 타석", batter=a.batter, pitcher=a.pitcher, detail=detail,
        ))

    def who(t: float) -> tuple[Optional[str], Optional[str]]:
        last = None
        for a in atbats:
            if vt(a) > t:
                break
            last = a
        return (last.batter, last.pitcher) if last else (None, None)

    for e in video_events:
        batter, pitcher = who(e.t)
        e2 = e.model_copy(update={"batter": e.batter or batter, "pitcher": e.pitcher or pitcher})
        if e.kind == KIND_SUB:
            near = [s for s in subs if abs(vt(s) - e.t) <= SUB_MATCH_SEC]
            if near:
                best = min(near, key=lambda s: abs(vt(s) - e.t))
                e2.detail = {**e.detail, "pitcher": best.detail.get("pitcher")}
        out.append(e2)
        if e.kind == KIND_RESULT and e.detail.get("result") == "strikeout":
            pitch = _last_pitch(pitches, e.t - offset, atbats)
            if pitch is not None:
                # 결정구: 영상이 삼진으로 판정한 타석의 마지막 공 — 구종·구속만 데이터에서
                out.append(RelayEvent(
                    id=f"{e.id}p", t=e.t, inning=e.inning, half=e.half, kind=KIND_PITCH,
                    text=f"{pitch.detail['pitch_type']} {pitch.detail.get('speed') or ''}km/h",
                    batter=e2.batter, pitcher=e2.pitcher,
                    detail={"result": "unknown", "pitch_type": pitch.detail["pitch_type"],
                            "speed": pitch.detail.get("speed"), "decisive": True,
                            "source": "data"},
                ))
    out.sort(key=lambda e: (e.t, _RANK.get(e.kind, 5)))
    return out


def _last_pitch(
    pitches: Sequence[RelayEvent], relay_t: float, atbats: Sequence[RelayEvent],
    slack: float = 15.0,
) -> Optional[RelayEvent]:
    """중계 시각 relay_t 직전(같은 타석 안)의 마지막 투구."""
    pa_start = max((a.t for a in atbats if a.t <= relay_t + slack), default=float("-inf"))
    cands = [p for p in pitches if pa_start <= p.t <= relay_t + slack]
    return cands[-1] if cands else None
