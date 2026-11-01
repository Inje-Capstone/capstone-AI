"""타자 스윙 — 손이 얼마나 빨리 움직였나로 가른다. 순수 로직(I/O 없음).

관절 30fps에서 두 손목의 위치를 엉덩이 기준으로 재고, 0.2초 동안 좌우로 움직인 거리를 키로
나눈다. 좌우만 보는 이유: 공이 지나간 뒤 배트를 내리는 손은 위아래로 움직인다. 배트를 직접
찾는 방법은 실패했다(허프 선 AUC 0.37 — 긴 선은 배트보다 광고판과 몸 윤곽이었다).

경계에 걸친 값과 손을 끝까지 못 본 장면은 어느 쪽으로도 쓰지 않는다(`clear`).

실측(2026-11-01, KBO 1080p, 사람이 프레임을 보고 매긴 정답). 기준값을 정한 뒤 쓰지 않던 영상
2편 52장면: 스윙이라 한 29개 중 28개, 안 침이라 한 12개 모두 맞음, 11개는 모름. 다만 실제로
안 친 공 23개 중 '안 침'으로 가른 것은 12개뿐이다. 하이라이트만 보면 안 친 공이 거의 없어
'안 침'으로 잘못 매긴 정답이 많았다 — 손목 AUC가 정답을 고치자 0.77에서 0.94로 올랐다.

빠르게 휘두르면 손목이 흐려 점수가 낮다 — 한 손만 잡혀도 쓰고, 짧은 빈칸은 앞뒤를 잇는다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Optional

Point = tuple[float, float, float]
Person = Sequence[Point]
XY = tuple[float, float]

MIN_SCORE = 0.3
MIN_JOINTS = 8  # 관절이 이보다 적게 잡히면 키를 재지 않는다
SWING_MOVE = 0.30  # 이상이면 스윙
NO_SWING_MOVE = 0.15  # 미만이면 안 침. 그 사이는 모른다
PITCH_LIFT = 0.40  # 두 발목 높이 차 / 키 — 넘어야 투구로 본다
SPAN_FRAMES = 6  # 0.2초 (30fps)
GAP_FRAMES = 5  # 이만큼까지 끊긴 손은 앞뒤를 잇는다
WRISTS = (9, 10)
HIPS = (11, 12)
ANKLES = (15, 16)


@dataclass(frozen=True)
class Swing:
    moved: float  # 0.2초에 손이 좌우로 움직인 거리 / 키
    swing: bool
    clear: bool  # 경계에 걸리지 않았다 — 판정 단서로 써도 된다
    frame: Optional[int]  # 가장 빠르게 움직인 구간의 가운데


@dataclass(frozen=True)
class Pitch:
    frame: int  # 두 발목 높이 차가 가장 큰 프레임
    lift: float


def _height(p: Person) -> Optional[float]:
    good = [q for q in p if q[2] >= MIN_SCORE]
    if len(good) < MIN_JOINTS:
        return None
    ys = [q[1] for q in good]
    return max(ys) - min(ys)


def _center(p: Person) -> Optional[XY]:
    hips = [p[i] for i in HIPS if p[i][2] >= MIN_SCORE]
    if hips:
        return sum(q[0] for q in hips) / len(hips), sum(q[1] for q in hips) / len(hips)
    good = [q for q in p if q[2] >= MIN_SCORE]
    if len(good) < MIN_JOINTS:
        return None
    xs, ys = [q[0] for q in good], [q[1] for q in good]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def hand_positions(frames: Sequence[Person]) -> list[Optional[XY]]:
    """프레임마다 두 손 평균 위치 — 엉덩이 기준, 키 단위. 몸이 움직여도 손만 본다."""
    heights = [h for h in map(_height, frames) if h]
    if not heights:
        return [None] * len(frames)
    tall = median(heights)
    out: list[Optional[XY]] = []
    for p in frames:
        hands = [p[i] for i in WRISTS if p[i][2] >= MIN_SCORE]
        center = _center(p)
        if not hands or center is None:
            out.append(None)
            continue
        x = sum(q[0] for q in hands) / len(hands)
        y = sum(q[1] for q in hands) / len(hands)
        out.append(((x - center[0]) / tall, (y - center[1]) / tall))
    return out


def _smooth(points: Sequence[Optional[XY]]) -> list[Optional[XY]]:
    """앞뒤 셋의 중앙값 — 관절이 한 프레임 튀는 것을 없앤다."""
    out = list(points)
    for i in range(1, len(points) - 1):
        near = [p for p in points[i - 1:i + 2] if p is not None]
        if len(near) >= 2:
            out[i] = median(p[0] for p in near), median(p[1] for p in near)
    return out


def _watched(points: Sequence[Optional[XY]]) -> bool:
    """손을 끝까지 봤나. 이어 붙이고도 빈칸이 남으면 그 사이에 휘둘렀을 수 있다."""
    return bool(points) and all(p is not None for p in points)


def _fill(points: Sequence[Optional[XY]]) -> list[Optional[XY]]:
    out = list(points)
    seen = [i for i, p in enumerate(points) if p is not None]
    for a, b in zip(seen, seen[1:]):
        if 1 < b - a <= GAP_FRAMES:
            pa, pb = points[a], points[b]
            assert pa is not None and pb is not None
            for j in range(a + 1, b):
                k = (j - a) / (b - a)
                out[j] = pa[0] + (pb[0] - pa[0]) * k, pa[1] + (pb[1] - pa[1]) * k
    return out


def pitch_moment(frames: Sequence[Person]) -> Optional[Pitch]:
    """투수 관절 열 → 투구 순간. 공을 놓은 뒤 뒷발이 차올라 두 발목 높이 차가 가장 커진다.

    투구가 없는 화면(타자가 가만히 서 있는 화면)을 '안 침'으로 재지 않게 거른다. 시각도 화면
    움직임보다 정확하다: 스윙까지 간격이 5프레임 안팎이고(사분위 4~7), 광류 최대는
    -9~12프레임으로 흩어졌다(2026-11-01, 61장면).
    """
    heights = [h for h in map(_height, frames) if h]
    if not heights:
        return None
    tall = median(heights)
    best, at = 0.0, None
    for i, p in enumerate(frames):
        if min(p[k][2] for k in ANKLES) < MIN_SCORE:
            continue
        lift = abs(p[ANKLES[0]][1] - p[ANKLES[1]][1]) / tall
        if lift > best:
            best, at = lift, i
    return Pitch(at, round(best, 3)) if at is not None and best >= PITCH_LIFT else None


def measure(frames: Sequence[Person]) -> Swing:
    """타자 한 사람의 관절 열(30fps) → 스윙 여부."""
    points = _fill(_smooth(hand_positions(frames)))
    best, at = 0.0, None
    for i in range(len(points) - SPAN_FRAMES):
        a, b = points[i], points[i + SPAN_FRAMES]
        if a is None or b is None:
            continue
        moved = abs(a[0] - b[0])
        if moved > best:
            best, at = moved, i + SPAN_FRAMES // 2
    swing = best >= SWING_MOVE
    return Swing(round(best, 3), swing, swing or (best < NO_SWING_MOVE and _watched(points)), at)
