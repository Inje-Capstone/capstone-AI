"""투수 동작 → 보크 의심 동작. 순수 로직(I/O 없음) — 입력은 프레임별 투수 관절 좌표(COCO 17점).

보크는 유형이 여럿이다(규칙 6.02(a) 13가지). 실측 클립(2026-10-26)에서도 정지 없이 투구,
세트 후 상체를 움직임(린드블럼), 투구 중 넘어짐(폰세), 빈 베이스로 던짐(배영수)이 모두 달랐다.
그래서 '세트 이후 무엇을 했나'를 세 갈래로 나눠 본다.

1. 세트: 두 손목이 몸 앞에서 모이고 손목이 거의 멈춘다
2. 세트 뒤 첫 큰 움직임의 결과
   - 던지기 완료: 다리를 들었거나(투구), 손을 머리 위로 들었거나, 팔을 옆으로 크게 뻗었다(견제)
     · 중계 카메라는 투수 뒤(중견수 쪽)에서 찍어 홈 쪽 보폭은 화면 깊이 방향이라 안 보인다
       (실측 2026-10-26) — 보폭 대신 다리·팔 높이를 본다
   - 중단: 움직였다가 내딛기·팔 뻗기 없이 다시 멈춘다 → 6.02(a)(1) '투구 동작 중단' 의심
3. 던지기 완료인데 세트 정지가 STOP_MIN초보다 짧다 → 6.02(a)(13) '정지 없음' 의심

판정은 심판 몫이다. 결과는 '의심 동작'과 그 근거(정지 시간·보폭·팔 뻗음)다(core-belief 4).
세트 포지션(주자 있음)일 때만 의미가 있다 — 주자 여부는 점수판이 준다.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional

# COCO 키포인트 번호
NOSE = 0
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

HANDS_TOGETHER = 0.6  # 손목 거리 / 어깨 너비 — 이보다 작으면 손이 모였다
STILL_SPEED = 0.35  # 몸통 길이/초 — 손목 중점이 이보다 느리면 멈춘 것
MOVE_SPEED = 1.0  # 몸통 길이/초 — 이보다 빠르면 '움직였다'
STOP_MIN = 0.3  # 세트 정지로 인정할 최소 시간(초)
KNEE_LIFT = 0.35  # 두 무릎 높이 차 / 몸통 길이 — 이보다 크면 다리를 들었다
ARM_UP = 0.2  # 손목이 코보다 이만큼(몸통 길이) 위면 팔을 들었다
REACH = 1.3  # 손목-어깨 거리 / 몸통 길이 — 이보다 멀면 팔을 옆으로 뻗었다(견제 송구)
OUTCOME_WINDOW = 1.5  # 움직임 시작 뒤 이 안에 던지기를 마쳐야 한다(초)
MIN_SCORE = 0.3


@dataclass(frozen=True)
class PoseFrame:
    """한 프레임의 투수 관절. points[i] = (x, y, score)."""

    t: float
    points: tuple[tuple[float, float, float], ...]

    def ok(self, *idx: int) -> bool:
        return all(self.points[i][2] >= MIN_SCORE for i in idx)

    def xy(self, i: int) -> tuple[float, float]:
        return self.points[i][0], self.points[i][1]


@dataclass
class PitchMotion:
    """세트 한 번과 그 뒤 동작."""

    t_set: float
    stop_sec: float
    t_move: Optional[float]
    outcome: str  # throw | aborted | unknown | none
    suspect: Optional[str]  # no_stop | interrupted | None
    evidence: list[str] = field(default_factory=list)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def body_scale(f: PoseFrame) -> Optional[tuple[float, float]]:
    """(어깨 너비, 몸통 길이) — 못 재면 None."""
    if not f.ok(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        return None
    shoulders = _dist(f.xy(L_SHOULDER), f.xy(R_SHOULDER))
    torso = _dist(_mid(f.xy(L_SHOULDER), f.xy(R_SHOULDER)), _mid(f.xy(L_HIP), f.xy(R_HIP)))
    if shoulders <= 1 or torso <= 1:
        return None
    return shoulders, torso


def hands_together(f: PoseFrame) -> bool:
    scale = body_scale(f)
    if scale is None or not f.ok(L_WRIST, R_WRIST):
        return False
    return _dist(f.xy(L_WRIST), f.xy(R_WRIST)) / max(scale[0], scale[1] * 0.5) < HANDS_TOGETHER


def knee_lift(f: PoseFrame) -> Optional[float]:
    """두 무릎 높이 차 / 몸통 길이 (화면 y는 아래로 증가)."""
    scale = body_scale(f)
    if scale is None or not f.ok(L_KNEE, R_KNEE):
        return None
    return abs(f.points[L_KNEE][1] - f.points[R_KNEE][1]) / scale[1]


def arm_up(f: PoseFrame) -> Optional[float]:
    """더 높이 든 손목이 코보다 얼마나 위인가 / 몸통 길이."""
    scale = body_scale(f)
    if scale is None or not f.ok(NOSE, L_WRIST, R_WRIST):
        return None
    top = min(f.points[L_WRIST][1], f.points[R_WRIST][1])
    return (f.points[NOSE][1] - top) / scale[1]


def reach(f: PoseFrame) -> Optional[float]:
    """더 멀리 뻗은 팔의 손목-어깨 거리 / 몸통 길이."""
    scale = body_scale(f)
    if scale is None or not f.ok(L_WRIST, R_WRIST):
        return None
    return (
        max(_dist(f.xy(L_WRIST), f.xy(L_SHOULDER)), _dist(f.xy(R_WRIST), f.xy(R_SHOULDER)))
        / scale[1]
    )


def wrist_speeds(frames: Sequence[PoseFrame]) -> list[Optional[float]]:
    """프레임마다 손목 중점 속도 (몸통 길이/초). 첫 프레임·못 재는 프레임·장면 끊김은 None."""
    out: list[Optional[float]] = [None]
    for a, b in zip(frames, frames[1:]):
        sa, sb = body_scale(a), body_scale(b)
        dt = b.t - a.t
        if (
            sa is None
            or sb is None
            or not a.ok(L_WRIST, R_WRIST)
            or not b.ok(L_WRIST, R_WRIST)
            or dt <= 0
            or dt > 0.5
        ):
            out.append(None)
            continue
        ma, mb = _mid(a.xy(L_WRIST), a.xy(R_WRIST)), _mid(b.xy(L_WRIST), b.xy(R_WRIST))
        v = _dist(ma, mb) / ((sa[1] + sb[1]) / 2) / dt
        out.append(v if v < 30 else None)  # 이보다 빠르면 사람이 바뀐 것(추적 실수)
    return out


def _still_run(
    frames: Sequence[PoseFrame], speeds: Sequence[Optional[float]], start: int, end: int
) -> float:
    """[start, end) 안에서 손이 모인 채 멈춘 가장 긴 시간."""
    longest, run_start = 0.0, None
    for k in range(start, end):
        v = speeds[k]
        if hands_together(frames[k]) and v is not None and v < STILL_SPEED:
            run_start = frames[k - 1].t if run_start is None and k > 0 else run_start
            if run_start is not None:
                longest = max(longest, frames[k].t - run_start)
        else:
            run_start = None
    return longest


def _peak(fn: Callable[[PoseFrame], Optional[float]], frames: Sequence[PoseFrame]) -> float:
    return max((v for v in (fn(f) for f in frames) if v is not None), default=0.0)


def analyze_motions(frames: Sequence[PoseFrame]) -> list[PitchMotion]:
    """프레임 열 → 세트마다 그 뒤 동작과 보크 의심 여부."""
    speeds = wrist_speeds(frames)
    motions: list[PitchMotion] = []
    i, n = 0, len(frames)
    while i < n:
        if not hands_together(frames[i]):
            i += 1
            continue
        start = i
        j = i
        while (
            j < n
            and (speeds[j] is None or speeds[j] < MOVE_SPEED)
            and frames[j].t - frames[max(j - 1, 0)].t <= 0.5
        ):
            j += 1
        t_set = frames[start].t
        stop = _still_run(frames, speeds, start, j)
        if j >= n or frames[j - 1].t - t_set < 0.2:  # 끝까지 안 움직였거나 한순간 스침
            if j < n:
                i = j + 1
                continue
            motions.append(
                PitchMotion(
                    round(t_set, 2),
                    round(stop, 2),
                    None,
                    "none",
                    None,
                    [f"세트 {t_set:.1f}s 뒤 큰 움직임 없음"],
                )
            )
            break
        t_move = frames[j].t
        window = [f for f in frames[j:] if f.t - t_move <= OUTCOME_WINDOW]
        lift, up, far = (_peak(fn, window) for fn in (knee_lift, arm_up, reach))
        threw = lift >= KNEE_LIFT or up >= ARM_UP or far >= REACH
        lost = len(window) < max(3, int(OUTCOME_WINDOW * 4))  # 움직임 뒤 투수를 놓침
        if threw:
            outcome, suspect = "throw", ("no_stop" if stop < STOP_MIN else None)
        elif lost:
            outcome, suspect = "unknown", None
        else:
            outcome, suspect = "aborted", "interrupted"
        motions.append(
            PitchMotion(
                round(t_set, 2),
                round(stop, 2),
                round(t_move, 2),
                outcome,
                suspect,
                [
                    f"세트 {t_set:.1f}s · 정지 {stop:.2f}s(기준 {STOP_MIN}s) · "
                    f"움직임 {t_move:.1f}s · 다리 {lift:.2f}(기준 {KNEE_LIFT}) · "
                    f"팔 높이 {up:.2f}(기준 {ARM_UP}) · 팔 뻗음 {far:.1f}(기준 {REACH})"
                ],
            )
        )
        k = j
        while k < n and frames[k].t - t_move <= OUTCOME_WINDOW:
            k += 1
        i = k
    return motions
