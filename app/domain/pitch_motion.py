"""투수 동작 → 보크 의심 동작. 순수 로직(I/O 없음) — 입력은 프레임별 투수 관절 좌표(COCO 17점).

판정 기준은 저장된 규칙(data/rules_catalog.json의 6.02(a)·5.07, 용어 사전 balk_*)에서 온다.
모든 보크의 **성립 조건**: ① 축발이 투수판에 닿아 있다 ② 주자가 있다.
- ①: 축발을 먼저 뒤로 빼면(이탈) 야수와 같아져 그 뒤 동작은 보크가 아니다 → 이탈은 의심에서 뺀다
- ②: 주자 여부는 동작으로 모른다 → 점수판·해설이 정한다(balk_judge)

자세로 재는 조항 (중견수 쪽 카메라 — 투수 등이 보인다)
- 6.02(a)(13) 정지 없음: 세트에 들어가는 움직임 뒤 멈춤 없이 바로 다리를 든다
- 6.02(a)(1) 투구 동작 중단: 세트 들어가다 짧게 멈췄다 다시 감(이중 동작) / 다리를 들다 팔을
  들지 않고 다시 섬 / 세트 후 자유발을 옮김 (KBO 해설은 모두 '투수 보크'로 설명)
- 6.02(a)(10) 손 분리: 세트 정지 뒤 투구·견제 없이 팔(팔꿈치)이 내려온다
- 6.02(a)(3) 발 안 내딛고 견제: 팔을 옆으로 뻗어 던지는데 자유발이 먼저 안 나간다
나머지 조항(공·포수·타자·시간이 필요)은 자세로 못 잰다 — 해설로만 판정한다(카탈로그 motion 칸).

실측(2026-10-26 KBO 보크 클립): 세트 때 손목은 몸에 가리고(신뢰도 0.4~0.6), 홈 쪽 보폭은 화면
깊이 방향이라 안 보인다 — 어깨·팔꿈치·무릎·발목으로 잰다. 판정은 심판 몫이다(근거 수치를 남긴다).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

# COCO 키포인트 번호
NOSE = 0
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
UPPER = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW)  # 투수 뒤에서 찍어도 보인다

STILL_SPEED = 0.35  # 몸통 길이/초 — 상체가 이보다 느리면 멈춘 것
STOP_MIN = 0.3  # 세트 정지로 인정할 최소 시간(초) — '분명한 정지'의 하한
SHORT_PAUSE = 0.5  # 세트 들어가는 중간 멈춤이 이보다 짧으면 이중 동작 의심
KNEE_LIFT = 0.35  # 두 무릎 높이 차 / 몸통 길이 — 이보다 크면 다리를 들었다(투구 시작)
KNEE_START = 0.2  # 다리를 들기 시작했다고 볼 높이 차
ARM_UP = 0.2  # 손목이 코보다 이만큼 위 — 팔을 들었다(투구로 이어짐)
REACH = 1.3  # 손목-어깨 거리 / 몸통 길이 — 팔을 옆으로 크게 뻗음(견제 송구)
FOOT_MOVE = 0.3  # 발목 이동 / 몸통 길이 — 이보다 움직이면 발을 옮긴 것
ARM_DROP = 0.25  # 팔꿈치가 세트 위치보다 이만큼 내려오면 손을 푼 것
DELIVERY_START = 0.5  # 정지 뒤 다리가 기준선까지 오르는 동안의 움직임(초) — 실측 0.1~0.5초
MIN_SCORE = 0.3

# 자세로 재는 의심 종류 → 카탈로그 id (data/rules_catalog.json)
PITCH_ONLY_RULES = {
    "no_stop": "balk_13_no_stop",
    "double_motion": "balk_1_stop",
    "kick_aborted": "balk_1_stop",
    "foot_off": "balk_1_stop",
    "hands_separated": "balk_10_hands",
    "pickoff_no_step": "balk_3_step",
}


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
    """세트(또는 투구 시작) 하나와 그 판단."""

    t_set: float
    stop_sec: float
    t_move: Optional[float]
    outcome: str  # throw | pickoff | aborted | step_off
    suspect: Optional[str]  # PITCH_ONLY_RULES 키 | None
    evidence: list[str] = field(default_factory=list)

    @property
    def rule(self) -> Optional[str]:
        return PITCH_ONLY_RULES.get(self.suspect or "")


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


def _hip(f: PoseFrame) -> tuple[float, float]:
    return _mid(f.xy(L_HIP), f.xy(R_HIP))


def _rel(f: PoseFrame, i: int, base: PoseFrame) -> Optional[tuple[float, float]]:
    """관절 i 위치 — `base` 프레임의 엉덩이 기준, 몸통 길이 단위."""
    scale = body_scale(base)
    if scale is None or not f.ok(i):
        return None
    hx, hy = _hip(base)
    return (f.points[i][0] - hx) / scale[1], (f.points[i][1] - hy) / scale[1]


def knee_lift(f: PoseFrame) -> Optional[float]:
    """두 무릎 높이 차 / 몸통 길이 (화면 y는 아래로 증가)."""
    scale = body_scale(f)
    if scale is None or not f.ok(L_KNEE, R_KNEE):
        return None
    return abs(f.points[L_KNEE][1] - f.points[R_KNEE][1]) / scale[1]


def lifted_side(f: PoseFrame) -> Optional[str]:
    """든 다리 쪽 'L'|'R' (= 자유발) — 다리를 들지 않았으면 None."""
    lift = knee_lift(f)
    if lift is None or lift < KNEE_START:
        return None
    return "L" if f.points[L_KNEE][1] < f.points[R_KNEE][1] else "R"


def arm_up(f: PoseFrame) -> Optional[float]:
    scale = body_scale(f)
    if scale is None or not f.ok(NOSE) or not (f.ok(L_WRIST) or f.ok(R_WRIST)):
        return None
    tops = [f.points[i][1] for i in (L_WRIST, R_WRIST) if f.ok(i)]
    return (f.points[NOSE][1] - min(tops)) / scale[1]


def reach(f: PoseFrame) -> Optional[float]:
    scale = body_scale(f)
    if scale is None:
        return None
    arms = [
        _dist(f.xy(w), f.xy(s))
        for w, s in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER))
        if f.ok(w, s)
    ]
    return max(arms) / scale[1] if arms else None


def upper_body_speeds(frames: Sequence[PoseFrame]) -> list[Optional[float]]:
    """프레임마다 상체(어깨·팔꿈치) 속도 (몸통 길이/초).

    엉덩이 기준 좌표라 카메라 이동이 상쇄되고, 3프레임 중앙값으로 흔들림을 줄인다.
    """
    raw: list[Optional[float]] = [None]
    for a, b in zip(frames, frames[1:]):
        sa, sb = body_scale(a), body_scale(b)
        dt = b.t - a.t
        if sa is None or sb is None or dt <= 0 or dt > 0.5:
            raw.append(None)
            continue
        ha, hb = _hip(a), _hip(b)
        moves = [
            _dist(
                (a.points[j][0] - ha[0], a.points[j][1] - ha[1]),
                (b.points[j][0] - hb[0], b.points[j][1] - hb[1]),
            )
            for j in UPPER
            if a.ok(j) and b.ok(j)
        ]
        if len(moves) < 3:
            raw.append(None)
            continue
        v = sum(moves) / len(moves) / ((sa[1] + sb[1]) / 2) / dt
        raw.append(v if v < 30 else None)  # 이보다 빠르면 사람이 바뀐 것(추적 실수)
    out: list[Optional[float]] = []
    for k in range(len(raw)):
        near = [v for v in raw[max(0, k - 1) : k + 2] if v is not None]
        out.append(sorted(near)[len(near) // 2] if near else None)
    return out


def segments(frames: Sequence[PoseFrame]) -> list[tuple[str, int, int]]:
    """still(상체 멈춤) / move(움직임) / kick(다리 듦) 구간 [(상태, 시작, 끝(포함))].

    0.5초 넘게 끊기면(장면 전환) 구간을 끊는다.
    """
    speeds = upper_body_speeds(frames)
    out: list[tuple[str, int, int]] = []
    for k, (f, v) in enumerate(zip(frames, speeds)):
        lift = knee_lift(f)
        if lift is not None and lift >= KNEE_LIFT:
            st = "kick"
        elif v is None:
            continue
        else:
            st = "still" if v < STILL_SPEED else "move"
        gap = k > 0 and frames[k].t - frames[k - 1].t > 0.5
        if out and out[-1][0] == st and not gap and out[-1][2] == k - 1:
            out[-1] = (st, out[-1][1], k)
        else:
            out.append((st, k, k))
    return out


def _dur(frames: Sequence[PoseFrame], seg: tuple[str, int, int]) -> float:
    return frames[seg[2]].t - frames[seg[1]].t + 0.1  # 구간은 프레임 하나만큼 짧게 잡힌다


def free_side(frames: Sequence[PoseFrame]) -> Optional[str]:
    """이 투수의 자유발 쪽 — 다리를 크게 든 프레임에서 다수결. 축발은 반대쪽."""
    votes = [lifted_side(f) for f in frames if (knee_lift(f) or 0) >= KNEE_LIFT]
    votes = [v for v in votes if v]
    return max(set(votes), key=votes.count) if votes else None


def _ankles(side: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """(자유발 발목, 축발 발목)"""
    if side == "L":
        return L_ANKLE, R_ANKLE
    if side == "R":
        return R_ANKLE, L_ANKLE
    return None, None


def _moved(frames: Sequence[PoseFrame], i: int, j: int, joint: Optional[int]) -> float:
    """프레임 i → j 사이 관절 이동(몸통 길이, i 프레임 엉덩이 기준)."""
    if joint is None:
        return 0.0
    a, b = _rel(frames[i], joint, frames[i]), _rel(frames[j], joint, frames[i])
    return _dist(a, b) if a and b else 0.0


def _double_motion(frames, segs, k_still) -> Optional[float]:
    """세트 정지(segs[k_still]) 전 2.5초 안 '움직임–짧은 멈춤–움직임'의 멈춤 길이."""
    t_end = frames[segs[k_still][1]].t
    before = [sg for sg in segs[:k_still] if t_end - frames[sg[2]].t <= 2.5]
    for a in range(len(before) - 2):
        kinds = (before[a][0], before[a + 1][0], before[a + 2][0])
        if kinds == ("move", "still", "move") and _dur(frames, before[a + 1]) < SHORT_PAUSE:
            return _dur(frames, before[a + 1])
    return None


def _leads_to_pitch(frames: Sequence[PoseFrame], segs: list, k: int) -> bool:
    """segs[k] 멈춤 뒤 3초 안에 '멈춤 → 다리 듦'이 오면 사인 보기 → 세트 → 투구 흐름이다."""
    t0 = frames[segs[k][2]].t
    kinds = [sg[0] for sg in segs[k + 1 :] if frames[sg[1]].t - t0 <= 3.0]
    return any(kinds[a : a + 2] == ["still", "kick"] for a in range(len(kinds) - 1))


def analyze_motions(frames: Sequence[PoseFrame]) -> list[PitchMotion]:
    """세트 정지 앞(세트 들어가기)과 뒤(투구·견제·이탈·중단)를 조항별로 본다."""
    if not frames:
        return []
    segs = segments(frames)
    free_ankle, pivot_ankle = _ankles(free_side(frames))
    motions: list[PitchMotion] = []

    def add(
        t_set: float,
        stop: float,
        t_move: float,
        outcome: str,
        suspect: Optional[str],
        why: list[str],
    ) -> None:
        motions.append(
            PitchMotion(round(t_set, 2), round(stop, 2), round(t_move, 2), outcome, suspect, why)
        )

    for k, (state, s0, s1) in enumerate(segs):
        prev = segs[k - 1] if k > 0 else None
        nxt = segs[k + 1] if k + 1 < len(segs) else None

        # ── 다리를 든 순간(투구 시작) — 6.02(a)(13) 정지 없음 · (a)(1) 이중 동작·중단
        if state == "kick" and (prev is None or prev[0] != "kick"):
            t_kick = frames[s0].t
            if prev is None or t_kick - frames[prev[2]].t > 0.5:
                continue  # 다리 들기 전을 못 봤다(장면 전환)
            # 축발 = 지금 든 다리의 반대쪽(좌우 판별이 영상 전체 다수결과 어긋날 수 있다).
            # 다리를 들기 전까지만 잰다 — 든 순간엔 몸 전체가 움직인다
            side = lifted_side(frames[s0])
            pivot = _ankles(side)[1] if side else pivot_ankle
            if _moved(frames, prev[1], prev[2], pivot) >= FOOT_MOVE:
                add(
                    frames[prev[1]].t,
                    0.0,
                    t_kick,
                    "step_off",
                    None,
                    ["축발이 먼저 빠짐 — 투수판 이탈, 보크 아님"],
                )
                continue
            k_still = k - 1
            p2 = segs[k - 2] if k >= 2 else None
            seen = p2 is not None and frames[prev[1]].t - frames[p2[2]].t <= 0.5
            if seen and p2 is not None and p2[0] == "kick" and _dur(frames, prev) <= DELIVERY_START:
                continue  # 같은 투구 — 무릎이 기준선을 잠깐 오르내렸다
            if prev[0] == "move" and seen and p2 is not None:
                if p2[0] == "still" and _dur(frames, prev) <= DELIVERY_START:
                    k_still = k - 2  # 멈춘 뒤 투구 시작의 짧은 움직임 — 정지는 그 앞 구간
            elif prev[0] == "move":
                continue  # 다리 들기 전 정지 여부를 못 봤다(장면 전환 직후)
            stop = _dur(frames, segs[k_still]) if segs[k_still][0] == "still" else 0.0
            why = [f"투구 시작(다리 듦) {t_kick:.1f}s · 직전 정지 {stop:.2f}s(기준 {STOP_MIN}s)"]
            suspect: Optional[str] = None
            if stop < STOP_MIN:
                suspect = "no_stop"
            else:
                pause = _double_motion(frames, segs, k_still)
                if pause is not None:
                    suspect = "double_motion"
                    why.append(
                        f"세트 들어가다 {pause:.2f}s 멈췄다 다시 움직임(기준 {SHORT_PAUSE}s)"
                    )
            after = [f for f in frames[s1 + 1 :] if f.t - frames[s1].t <= 1.2]
            if (
                after
                and nxt is not None
                and nxt[0] == "still"
                and _dur(frames, nxt) >= 0.6
                and max((arm_up(f) or -9.0) for f in after) < ARM_UP
            ):
                suspect = "kick_aborted"
                why.append("다리를 들었다가 팔을 들지 않고 다시 멈춤")
            add(frames[prev[1]].t, stop, t_kick, "throw", suspect, why)
            continue

        # ── 세트 정지 뒤 다리를 안 들었다 — 이탈 · 견제(6.02(a)(3)) · 자유발 · 손 분리(a)(10)
        if state != "still" or _dur(frames, (state, s0, s1)) < STOP_MIN:
            continue
        if nxt is None or nxt[0] != "move":
            continue
        end = next((sg for sg in segs[k + 1 :] if sg[0] != "move"), None)
        if end is not None and end[0] == "kick":
            continue  # 투구로 이어짐 — 위에서 본다
        if _leads_to_pitch(frames, segs, k):
            continue  # 이 멈춤은 사인 보기, 다음 움직임은 세트 들어가기 — 정상 흐름
        j = nxt[2] if end is None else end[1]
        stop, t_move = _dur(frames, (state, s0, s1)), frames[nxt[1]].t
        if _moved(frames, s1, j, pivot_ankle) >= FOOT_MOVE:
            add(
                frames[s0].t,
                stop,
                t_move,
                "step_off",
                None,
                ["축발을 먼저 뺌 — 투수판 이탈, 그 뒤 동작은 보크 아님"],
            )
            continue
        far = max((reach(f) or 0.0) for f in frames[nxt[1] : j + 1])
        free_move = _moved(frames, s1, j, free_ankle)
        why = [
            f"세트 정지 {stop:.2f}s 뒤 {t_move:.1f}s 움직임 · 팔 뻗음 {far:.1f}(기준 {REACH})"
            f" · 자유발 이동 {free_move:.2f}(기준 {FOOT_MOVE})"
        ]
        if far >= REACH:  # 견제 송구 — 자유발이 베이스 쪽으로 나갔어야 한다
            no_step = free_ankle is not None and free_move < FOOT_MOVE
            add(frames[s0].t, stop, t_move, "pickoff", "pickoff_no_step" if no_step else None, why)
        elif free_move >= FOOT_MOVE:
            add(frames[s0].t, stop, t_move, "aborted", "foot_off", why + ["세트 후 자유발을 옮김"])
        else:
            drop = 0.0
            for i in (L_ELBOW, R_ELBOW):
                a, b = _rel(frames[s1], i, frames[s1]), _rel(frames[j], i, frames[s1])
                if a and b:
                    drop = max(drop, b[1] - a[1])
            if drop >= ARM_DROP:
                add(
                    frames[s0].t,
                    stop,
                    t_move,
                    "aborted",
                    "hands_separated",
                    why + [f"팔꿈치가 {drop:.2f} 내려옴(기준 {ARM_DROP}) — 투구·견제 없이 손을 풂"],
                )
    return motions
