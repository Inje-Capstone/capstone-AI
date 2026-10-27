"""선수 접촉·충돌·슬라이딩 — 화면 속 모든 사람의 관절(COCO 17점)로 잰다. 순수 로직(I/O 없음).

규칙 판단의 신호(카탈로그 신호 어휘 pose.contact · pose.slide_path)를 해설·VLM 말 없이 직접 잰다.
VLM 장면 v4로 '충돌'을 받아 보려 했더니 세리머니·물 뿌리기까지 충돌로 적었다(2026-10-26 폐기).

중계 화면은 2D라 겹쳐 보인다고 닿은 게 아니다. 그래서
- 같은 깊이: 두 사람 키가 비슷하고(원근) 발 높이가 비슷해야(같은 땅 위) 닿을 수 있다
- 오래 붙어 있으면 겹쳐 선 것: 타석의 포수 뒤 심판처럼 몇 초씩 겹친 건 접촉이 아니다
- 충돌은 '빠르게 다가와 닿고 누가 넘어짐', 슬라이딩 태그는 '미끄러져 들어와 누운 채 닿음'

중계 카메라는 공·주자를 따라 빠르게 돌고 당긴다(실측 0.1초에 화면 너비의 1/4, 3초에 4배 줌).
화면 속 위치만 보면 가만히 선 덕아웃 선수도 달리는 것처럼 보이고, 카메라가 따라가는 주자는
멈춘 것처럼 보인다. 그래서 위치·키는 카메라 변환(`PeopleFrame.cam`)을 되돌린 '세상 좌표'로
잰다. 다가온 속도(두 사람 사이 거리 / 키)는 카메라가 움직여도 변하지 않는다.

키(scale)는 관절 범위의 긴 변 — 누운 사람도 키가 줄지 않게.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Optional

Point = tuple[float, float, float]
Person = tuple[Point, ...]
XY = tuple[float, float]
Camera = tuple[float, float, float, float]  # 화면 = [[a,-b],[b,a]]·세상 + (tx, ty)
STILL: Camera = (1.0, 0.0, 0.0, 0.0)

MIN_SCORE = 0.3
MIN_JOINTS = 6
SAME = 0.1  # 같은 번호 관절이 키의 이 비율 안에 절반 넘게 겹치면 한 사람을 두 번 잡은 것
# 한 사람 관절 범위가 다른 사람 안에 이만큼 넘게 들면 접촉으로 안 본다 — 클로즈업에서 머리·상체만
# 따로 한 사람으로 잡힌 부분 검출(실측 2026-10-27, 다이빙 뒤 일어나는 수비).
SAME_BOX = 0.8
# 이어진 슬라이딩인데 몸 크기(세상 좌표)가 이보다 튀면 다른 화면(실측: 같은 화면 1.15, 다른 1.39)
SIZE_JUMP = 1.25
# 같은 때 몸 중심이 키의 SAME_PLACE배 안이거나 관절 범위가 PART 넘게 겹쳐 들면 한 사람(부분 검출).
# 같은 베이스에 함께 미끄러진 두 선수는 떨어져 있고 범위도 절반쯤만 겹친다
SAME_PLACE = 0.3
PART = 0.6
# 팔·다리·몸통 뼈(COCO 번호). 뼈 하나가 키의 BONE_MAX를 넘으면 여러 사람 관절이 섞인 것 —
# 실측(2026-10-27): 머리는 왼쪽 끝, 발목은 오른쪽 끝에 잡힌 뼈대가 '누운 사람'으로 읽혔다.
BONES = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
         (5, 11), (6, 12), (5, 6), (11, 12))
BONE_MAX = 0.75  # 웅크린 포수 허벅지·하체가 가린 덕아웃 선수 어깨 폭이 키의 절반을 넘는다
TOUCH = 0.15  # 두 사람 관절 사이 최소 거리 / 작은 사람 키 — 이하면 닿음
DEPTH_RATIO = 1.5  # 큰 사람 키 / 작은 사람 키 — 넘으면 다른 깊이(앞뒤로 떨어져 있다)
FOOT_GAP = 0.35  # 두 사람 발 높이 차 / 작은 사람 키 — 넘으면 다른 깊이
FOOT_REACH = 0.6  # 한 사람만 누웠으면 — 수비가 손을 뻗어 앞에 누운 주자에 닿는다(실측: 0.44)
STRETCH = 1.25  # 누워 팔을 뻗으면 관절 범위가 키의 이만큼 — 누운 사람 키의 아래 한계
TAG_HAND = 0.3  # 서 있는 사람 손목이 누운 주자에게 이만큼(작은 사람 키 배) 안 — 글러브 태그
HAND_AFTER = 0.5  # 닿은 뒤 이 시간까지 손이 닿는지 본다
# 손은 닿기 시작한 뒤 이 시간 안에서만 — 오래 누워 있는 동안 심판이 팔을 내려도 태그가 아니다
HAND_WINDOW = 1.0
STATIC = 2.5  # 이보다 오래 붙어 있으면 겹쳐 선 것(심판 뒤 포수, 수비 위치 겹침)
JOIN = 0.35  # 접촉·누운 구간 사이 이만큼 끊겨도 한 구간(관절 놓침, 10fps에 세 프레임)
LOOK_BACK = 0.5  # 닿기 이만큼 전과 비교해 다가온 속도를 잰다
CLOSING = 1.5  # 다가온 속도(키/초) — 이상이면 '빠르게 부딪힘'
LYING = 1.2  # 관절 범위 가로/세로 — 이상이면 누움(넘어짐·슬라이딩)
TORSO_FLAT = 1.0  # 어깨→엉덩이 가로/세로 — 이상이면 상체가 눕혀짐(45°). 허리 숙인 수비는 서 있다
# 슬라이딩·다이빙 = 누운 채 몸길이의 SLIDE_DIST배 넘게 미끄러짐(세상 좌표). 속도로 가르면 느린
# 화면(다시보기)·클로즈업 슬라이딩을 놓친다 — 실측(2026-10-27, 6클립 누운 후보 20개):
# 슬라이딩 0.41~3.4배, 넘어져 있음·포수 몸 날림·일어나기 0.11~0.37배.
SLIDE_DIST = 0.4
SLIDE_MIN = 0.2  # 이 시간 이상 누워 있어야(10fps 세 프레임)
# 달려오다 몸을 던짐 — 눕기 ENTRY_LOOK초 전 자리에서 몸길이 ENTRY배 넘게. 누운 뒤엔 거의 안
# 움직이는 다이빙·헤드퍼스트를 잡는다. 실측(기준·검증 18클립): 슬라이딩 1.8~3.3배,
# 아님 최대 1.08배(쓰러진 타자 0.82).
ENTRY = 1.2
ENTRY_LOOK = 1.0
RUN_SPEED = 1.0  # 몸을 던지기 전 달린 속도(키/초) — 걷다가(0.75) 넘어진 건 던진 게 아니다
STITCH = 0.3  # 전력질주 다이빙은 눕는 순간 자세가 확 바뀌어 추적이 끊긴다 — 이만큼 앞에 끝난 사람을
STITCH_NEAR = 1.5  # 누운 시작 자리에서 몸길이의 이만큼 안이면 같은 사람으로 이어 본다
# 느린 화면 다시보기 — 1.5초 넘게 누운 채 몸길이 0.25배 넘게.
# 실측(기준·검증): 슬라이딩 0.35, 누워 있음 최대 0.18.
SLOW_MIN = 1.5
SLOW_DIST = 0.25
SLIDE_INTO = 0.5  # 슬라이딩이 끝나고 이 시간 안에 닿아야 '미끄러져 들어와 닿음'
TRACK_JUMP = 0.6  # 추적: 0.1초에 키의 이 비율 넘게 움직이면 다른 사람(세상 좌표)
CUT = 0.5  # 프레임 사이가 이보다 비면 장면 전환 — 추적을 새로 시작
EDGE = 3  # 화면 아래 끝에서 이 px 안에 발(몸 아래)이 걸리면 잘린 사람 — 앞줄 관중(실측 2026-10-27)


@dataclass(frozen=True)
class PeopleFrame:
    t: float
    people: tuple[Person, ...]
    cam: Camera = STILL  # 장면 시작 화면 → 이 화면의 카메라 변환(패닝·줌)
    cut: bool = False  # 바로 앞 프레임과 다른 장면(화면 전환)
    height: int = 0  # 화면 높이(px) — 0이면 모름


@dataclass(frozen=True)
class Body:
    """한 사람의 관절 범위. pts는 COCO 번호 순서(못 잡은 관절은 None)."""

    x0: float
    y0: float
    x1: float
    y1: float
    pts: tuple[Optional[XY], ...]
    cam: Camera = STILL
    edge: bool = False  # 화면 아래 끝에 잘림 — 자세를 다 못 본다

    @property
    def joints(self) -> tuple[XY, ...]:
        return tuple(p for p in self.pts if p is not None)

    @property
    def scale(self) -> float:
        return max(self.x1 - self.x0, self.y1 - self.y0, 1.0)

    @property
    def center(self) -> XY:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def zoom(self) -> float:
        return max((self.cam[0] ** 2 + self.cam[1] ** 2) ** 0.5, 1e-6)

    @property
    def world(self) -> XY:
        """카메라 변환을 되돌린 몸 중심."""
        a, b, tx, ty = self.cam
        x, y = self.center[0] - tx, self.center[1] - ty
        z2 = self.zoom ** 2
        return (a * x + b * y) / z2, (-b * x + a * y) / z2

    @property
    def world_scale(self) -> float:
        return self.scale / self.zoom

    @property
    def lying(self) -> bool:
        if self.edge or (self.x1 - self.x0) < LYING * max(self.y1 - self.y0, 1.0):
            return False
        shoulders = [p for p in self.pts[5:7] if p is not None]
        hips = [p for p in self.pts[11:13] if p is not None]
        if not shoulders or not hips:  # 상체를 못 봤으면 눕혔는지 모른다 — 화면 끝에 걸린 사람
            return False
        sx, sy = _mean(shoulders)
        hx, hy = _mean(hips)
        return abs(sx - hx) >= TORSO_FLAT * abs(sy - hy)


def _mean(points: Sequence[XY]) -> XY:
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def body(person: Person, cam: Camera = STILL) -> Optional[Body]:
    """관절이 모자라거나 뼈대가 말이 안 되면(여러 사람 관절이 섞임) None."""
    pts = tuple((x, y) if s >= MIN_SCORE else None for x, y, s in person)
    good = [p for p in pts if p is not None]
    if len(good) < MIN_JOINTS:
        return None
    xs, ys = [p[0] for p in good], [p[1] for p in good]
    b = Body(min(xs), min(ys), max(xs), max(ys), pts, cam)
    for i, j in BONES:
        pi, pj = pts[i], pts[j]
        if pi is not None and pj is not None and _dist(pi, pj) > BONE_MAX * b.scale:
            return None
    return b


def _dist(a: XY, b: XY) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def duplicate(a: Body, b: Body) -> bool:
    """검출기가 한 사람을 두 번 잡았다 — 같은 번호 관절이 절반 넘게 겹친다."""
    both = [(p, q) for p, q in zip(a.pts, b.pts) if p is not None and q is not None]
    if len(both) < 4:
        return False
    limit = SAME * min(a.scale, b.scale)
    return sum(_dist(p, q) <= limit for p, q in both) * 2 > len(both)


def bodies(frame: PeopleFrame, min_scale: float = 0.0) -> list[Optional[Body]]:
    """사람 번호 순서의 몸. 관절이 모자라거나, 너무 작거나, 앞 사람과 겹친 중복이면 None."""
    out: list[Optional[Body]] = []
    for person in frame.people:
        b = body(person, frame.cam)
        if b is not None and (b.scale < min_scale or any(
                o is not None and duplicate(b, o) for o in out)):
            b = None
        if b is not None and frame.height and b.y1 >= frame.height - EDGE:
            b = replace(b, edge=True)
        out.append(b)
    return out


def _head_feet(b: Body) -> Optional[float]:
    """머리(얼굴 관절)에서 발목까지 가장 먼 거리 — 팔을 뻗었는지와 상관없는 몸길이."""
    head = [p for p in b.pts[0:5] if p is not None]
    feet = [p for p in b.pts[15:17] if p is not None]
    if not head or not feet:
        return None
    return max(_dist(p, q) for p in head for q in feet)


def _height(b: Body) -> float:
    """키 — 누운 사람은 머리~발 길이와 (팔 뻗은 몸길이 / STRETCH) 중 큰 쪽.

    머리~발만 쓰면 카메라 쪽으로 누운 주자가 짧아 보여 진짜 태그를 놓치고(실측 몸길이의 0.5배),
    관절 범위만 쓰면 팔을 안 뻗은 발 먼저 슬라이딩이 뒤쪽 작은 사람과 같은 깊이로 묶인다.
    """
    if not b.lying:
        return b.scale
    return max(_head_feet(b) or 0.0, b.scale / STRETCH)


def _size(b: Body) -> float:
    """세상 좌표 키 — 줌을 되돌린 크기. 이어진 슬라이딩이 같은 사람인지 견줄 때."""
    return _height(b) / b.zoom


def same_depth(a: Body, b: Body) -> bool:
    small, big = sorted((_height(a), _height(b)))
    reach = FOOT_REACH if a.lying != b.lying else FOOT_GAP
    return big <= DEPTH_RATIO * small and abs(a.y1 - b.y1) <= reach * small


def gap(a: Body, b: Body) -> float:
    """두 사람 관절 사이 가장 가까운 거리 / 작은 사람 키."""
    near = min(_dist(p, q) for p in a.joints for q in b.joints)
    return near / min(a.scale, b.scale)


def inside(a: Body, b: Body) -> float:
    """작은 쪽 관절 범위가 큰 쪽 안에 든 비율 — 0.8을 넘으면 한 사람을 쪼개 잡았거나 가려진 것."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    return w * h / max(small, 1.0)


def touching(a: Body, b: Body) -> bool:
    if not same_depth(a, b) or inside(a, b) > SAME_BOX:
        return False
    limit = TOUCH * min(a.scale, b.scale)
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    if (dx * dx + dy * dy) ** 0.5 > limit:  # 관절 범위끼리도 멀다 — 17×17 거리 계산 생략
        return False
    return gap(a, b) <= TOUCH


# ── 추적 ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Scene:
    """프레임별 몸과 추적 번호 — 접촉·슬라이딩이 함께 쓴다."""

    frames: Sequence[PeopleFrame]
    bodies: list[list[Optional[Body]]]
    ids: list[list[Optional[int]]]

    def paths(self) -> dict[int, list[tuple[float, Body]]]:
        """추적 번호 → [(시각, 몸)] 시간순."""
        out: dict[int, list[tuple[float, Body]]] = {}
        for f, row, ids in zip(self.frames, self.bodies, self.ids):
            for b, tid in zip(row, ids):
                if tid is not None and b is not None:
                    out.setdefault(tid, []).append((f.t, b))
        return out


def scene(frames: Sequence[PeopleFrame], min_scale: float = 0.0) -> Scene:
    rows = [bodies(f, min_scale) for f in frames]
    return Scene(frames, rows, _track(frames, rows))


def track(frames: Sequence[PeopleFrame], min_scale: float = 0.0) -> list[list[Optional[int]]]:
    """프레임마다 사람 번호 → 추적 번호. 못 쓰는 사람은 None."""
    return scene(frames, min_scale).ids


def _track(frames: Sequence[PeopleFrame],
           rows: list[list[Optional[Body]]]) -> list[list[Optional[int]]]:
    """가장 가까운 짝부터 이어 붙인다(욕심 매칭, 세상 좌표). 장면이 바뀌면 새로 시작한다."""
    out: list[list[Optional[int]]] = []
    last: dict[int, tuple[float, Body]] = {}  # 추적 번호 → (마지막 시각, 몸)
    next_id = 0
    prev_t: Optional[float] = None
    for f, row in zip(frames, rows):
        if f.cut or (prev_t is not None and f.t - prev_t > CUT):
            last = {}
        prev_t = f.t
        pairs = []
        for k, b in enumerate(row):
            if b is None:
                continue
            for tid, (t0, ob) in last.items():
                dt = max(f.t - t0, 0.05)
                d = _dist(b.world, ob.world) / ob.world_scale
                if d <= TRACK_JUMP * dt / 0.1:
                    pairs.append((d, k, tid))
        ids: list[Optional[int]] = [None] * len(row)
        used: set[int] = set()
        for _d, k, tid in sorted(pairs):
            if ids[k] is None and tid not in used:
                ids[k] = tid
                used.add(tid)
        for k, b in enumerate(row):
            if b is not None and ids[k] is None:
                ids[k], next_id = next_id, next_id + 1
        for k, b in enumerate(row):
            tid = ids[k]
            if b is not None and tid is not None:
                last[tid] = (f.t, b)
        last = {tid: v for tid, v in last.items() if f.t - v[0] <= CUT}
        out.append(ids)
    return out


def _at(path: list[tuple[float, Body]], t: float, tol: float = 0.15) -> Optional[Body]:
    best = min(path, key=lambda tb: abs(tb[0] - t), default=None)
    return best[1] if best is not None and abs(best[0] - t) <= tol else None


# ── 슬라이딩 ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Slide:
    t0: float
    t1: float
    who: int
    distance: float  # 이동 거리(몸길이 배, 세상 좌표) — 누운 채, 또는 던졌으면 던지기 전 자리부터
    thrown: bool = False  # 달려오다 몸을 던짐(누운 뒤엔 거의 안 움직임)
    members: tuple[int, ...] = ()  # 합쳐진 추적 번호들(who 포함) — 비었으면 (who,)


Run = list[tuple[float, Body]]


def _slides(sc: Scene) -> list[tuple[Slide, Run]]:
    """추적 번호마다의 슬라이딩(합치기 전)과 그 누운 구간 — 접촉은 번호로 찾으니 이걸 쓴다."""
    out = []
    paths = sc.paths()
    for tid, path in paths.items():
        runs: list[Run] = []
        for t, b in path:
            if not b.lying:
                continue
            if runs and t - runs[-1][-1][0] <= JOIN:
                runs[-1].append((t, b))
            else:
                runs.append([(t, b)])
        for run in runs:
            dur = run[-1][0] - run[0][0] + 1e-6  # 10fps 시각의 소수 오차
            if dur < SLIDE_MIN:
                continue
            scale = sum(b.world_scale for _t, b in run) / len(run)
            far = max(_dist(run[0][1].world, b.world) for _t, b in run) / scale
            slow = dur >= SLOW_MIN and far >= SLOW_DIST
            t0 = run[0][0]
            before = [(t, b) for t, b in path if t0 - ENTRY_LOOK <= t < t0]
            if not before:
                before = _stitched(paths, tid, run[0][1], t0)
            entry = _thrown(before, run, scale)
            if far >= SLIDE_DIST or slow:
                found = Slide(round(t0, 2), round(run[-1][0], 2), tid, round(far, 2))
            elif entry is not None:
                found = Slide(round(t0, 2), round(run[-1][0], 2), tid, round(entry, 2), True)
            else:
                continue
            out.append((found, run))
    return sorted(out, key=lambda sr: sr[0].t0)


def _stitched(paths: dict[int, Run], tid: int, start: Body, t0: float) -> Run:
    """누운 구간 앞을 못 봤다 — STITCH초 안에 끝난 다른 추적 중 누운 시작 자리 가까운 것."""
    best: Optional[tuple[float, int]] = None
    for other, path in paths.items():
        t_end, last = path[-1]
        if other == tid or not t0 - STITCH <= t_end < t0:
            continue
        d = _dist(last.world, start.world) / max(last.world_scale, start.world_scale)
        if d <= STITCH_NEAR and (best is None or d < best[0]):
            best = (d, other)
    if best is None:
        return []
    return [(t, b) for t, b in paths[best[1]] if t0 - ENTRY_LOOK <= t < t0]


def _thrown(before: Run, run: Run, scale: float) -> Optional[float]:
    """달려오다 몸을 던졌으면 던지기 전 자리부터 이동한 거리, 아니면 None.

    선 몸 중심과 누운 몸 중심은 자세만 바꿔도 몸길이 0.3~0.5배 차이가 난다 — 그래서 눕기 전에
    실제로 달리고 있었는지(RUN_SPEED)도 본다. 선 모습을 한 번밖에 못 봤으면 거리를 두 배로 요구한다.
    """
    if not before:
        return None
    entry = max(_dist(before[0][1].world, b.world) for _t, b in run) / scale
    standing = [(t, b) for t, b in before if not b.lying]
    if len(standing) >= 2 and standing[-1][0] - standing[0][0] >= 0.2:
        (ta, a), (tb, b) = standing[0], standing[-1]
        size = (a.world_scale + b.world_scale) / 2
        running = _dist(a.world, b.world) / size / (tb - ta) >= RUN_SPEED
        return entry if running and entry >= ENTRY else None
    return entry if entry >= 2 * ENTRY else None


def _cuts(frames: Sequence[PeopleFrame]) -> list[float]:
    """화면 전환 시각 — 표시된 전환과 CUT초 넘는 공백(경기장이 아닌 화면)."""
    return [f.t for i, f in enumerate(frames)
            if f.cut or (i and f.t - frames[i - 1].t > CUT)]


def _merge(found: list[tuple[Slide, Run]], cuts: Sequence[float] = ()) -> list[Slide]:
    """한 슬라이딩이 여러 번 잡힌 것을 하나로 — 1초 안, 화면이 안 바뀌었을 때(다른 각도는 따로):
    - 같은 추적 번호
    - 시간이 겹치면 같은 때 같은 자리(SAME_PLACE) — 한 사람을 둘로 잡은 부분 검출. 같은 베이스에
      동시에 미끄러진 두 선수는 떨어져 있다
    - 이어지면 앞이 끝난 자리에서 시작하고 키가 이어짐(SIZE_JUMP) — 크기가 튀면 디졸브 전환
    """
    out: list[tuple[Slide, Run]] = []
    for s, run in found:
        for k in range(len(out) - 1, -1, -1):
            p, prev = out[k]
            if s.t0 > p.t1 + 1.0 or any(p.t1 < c <= s.t0 for c in cuts):
                continue
            if s.who == p.who or (_together(prev, run) if s.t0 <= p.t1 else _continues(prev, run)):
                thrown = p.thrown if p.distance >= s.distance else s.thrown
                ids = tuple(dict.fromkeys((p.members or (p.who,)) + (s.who,)))
                merged = Slide(p.t0, max(p.t1, s.t1), p.who, max(p.distance, s.distance), thrown,
                               ids)
                out[k] = (merged, sorted(prev + run, key=lambda tb: tb[0]))
                break
        else:
            out.append((s, run))
    return [s for s, _r in out]


def _together(a: Run, b: Run) -> bool:
    """같은 때 같은 자리 — 한 사람을 두 번(부분 검출) 잡았다."""
    for t, x in a:
        y = _at(b, t, tol=0.05)
        if y is None:
            continue
        if inside(x, y) >= PART or _dist(x.world, y.world) <= SAME_PLACE * max(_size(x), _size(y)):
            return True
    return False


def _continues(a: Run, b: Run) -> bool:
    """앞 슬라이딩이 끝난 자리에서 키가 이어진 몸으로 다시 시작했다."""
    last, start = a[-1][1], b[0][1]
    near = _dist(last.world, start.world) <= 2 * max(last.world_scale, start.world_scale)
    small, big = sorted((_size(last), _size(start)))
    return near and big <= SIZE_JUMP * small


def slides(frames: Sequence[PeopleFrame], min_scale: float = 0.0) -> list[Slide]:
    """누운 채(상체가 눕혀진 자세) 미끄러진 구간 — 슬라이딩·다이빙.

    셋 중 하나: 누운 채 몸길이 0.4배 넘게 / 달려오다(1키/초 넘게) 몸을 던져 1초 전 자리에서
    1.2배 넘게 / 느린 화면으로 1.5초 넘게 누운 채 0.25배 넘게. 넘어져 제자리에 누워 있는 건 뺀다.
    """
    return _merge(_slides(scene(frames, min_scale)), _cuts(frames))


# ── 접촉 ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Contact:
    t0: float
    t1: float
    a: int  # 추적 번호
    b: int
    closing: Optional[float]  # 닿기 LOOK_BACK초 전부터 다가온 속도(키/초) — 그 전을 못 봤으면 None
    fell: bool  # 닿은 뒤 1초 안에 서 있던 사람이 누움
    slid: bool  # 한 사람이 미끄러져 들어와 누운 채 닿음
    kind: str  # collision | slide_tag | contact
    hand: Optional[float] = None  # 서 있는 사람 손목 ↔ 누운 사람(작은 사람 키 배) — 못 봤으면 None

    @property
    def why(self) -> str:
        speed = "못 봄" if self.closing is None else f"{self.closing:.1f}키/초"
        return (f"{self.t0:.1f}~{self.t1:.1f}s 사람 {self.a}·{self.b} 접촉 · 다가온 속도 {speed}"
                f"(기준 {CLOSING}) · 넘어짐 {'예' if self.fell else '아니오'}"
                f" · 미끄러져 닿음 {'예' if self.slid else '아니오'}"
                + ("" if self.hand is None else f" · 손목 {self.hand:.2f}(기준 {TAG_HAND})"))


def _contacts(sc: Scene, slid: Sequence[Slide], merged: Sequence[Slide] = ()) -> list[Contact]:
    paths = sc.paths()
    spans: dict[tuple[int, int], list[list[float]]] = {}
    for f, row, ids in zip(sc.frames, sc.bodies, sc.ids):
        here = [(tid, b) for b, tid in zip(row, ids) if tid is not None and b is not None]
        for i in range(len(here)):
            for j in range(i + 1, len(here)):
                (ta, ba), (tb, bb) = here[i], here[j]
                if not touching(ba, bb):
                    continue
                key = (min(ta, tb), max(ta, tb))
                runs = spans.setdefault(key, [])
                if runs and f.t - runs[-1][1] <= JOIN:
                    runs[-1][1] = f.t
                else:
                    runs.append([f.t, f.t])
    out = []
    for (a, b), runs in spans.items():
        for t0, t1 in runs:
            lying = any(body.lying for path in (paths[a], paths[b])
                        for t, body in path if t0 <= t <= t1)
            if t1 - t0 > STATIC and not lying:  # 둘 다 서서 오래 겹친 것만 뺀다(태그 유지는 남김)
                continue
            out.append(_describe(paths[a], paths[b], a, b, t0, t1, slid))
    return _one_tag_per_slide(sorted(out, key=lambda c: (c.t0, c.a, c.b)), merged or slid)


def _one_tag_per_slide(found: list[Contact], slid: Sequence[Slide]) -> list[Contact]:
    """미끄러져 들어온 주자는 포수·심판과 함께, 또 몇 번 닿기도 한다 — 한 슬라이딩에 태그 한 번.

    slid는 합친 슬라이딩 — 추적이 끊겨 두 번호로 잡힌 한 주자도 한 번으로 센다.
    """
    out: list[Contact] = []
    seen: set[tuple[int, float]] = set()
    for c in found:
        if c.kind == "slide_tag":
            s = next((s for s in slid if {c.a, c.b} & set(s.members or (s.who,))
                      and s.t0 - JOIN <= c.t0 <= s.t1 + SLIDE_INTO), None)
            key = (s.who, s.t0) if s is not None else (min(c.a, c.b), round(c.t0))
            if key in seen:
                continue
            seen.add(key)
        out.append(c)
    return out


def _legs(b: Optional[Body]) -> bool:
    """발목이 보인다 — 다리까지 화면에 든 사람. 상체만 보이는 클로즈업은 넘어짐을 못 믿는다."""
    return b is not None and (b.pts[15] is not None or b.pts[16] is not None)


def contacts(frames: Sequence[PeopleFrame], min_scale: float = 0.0) -> list[Contact]:
    """접촉 구간 — 같은 깊이에서 닿았고, 짧게(STATIC초 안) 닿은 것만."""
    sc = scene(frames, min_scale)
    raw = _slides(sc)
    return _contacts(sc, [s for s, _r in raw], _merge(raw, _cuts(frames)))


def _describe(pa, pb, a: int, b: int, t0: float, t1: float, slid: Sequence[Slide]) -> Contact:
    before_a, before_b = _at(pa, t0 - LOOK_BACK), _at(pb, t0 - LOOK_BACK)
    now_a, now_b = _at(pa, t0), _at(pb, t0)
    closing = None
    if before_a and before_b and now_a and now_b:  # 그때그때 키로 나눠 줌에도 같은 값
        d0 = _dist(before_a.center, before_b.center) / min(before_a.scale, before_b.scale)
        d1 = _dist(now_a.center, now_b.center) / min(now_a.scale, now_b.scale)
        closing = (d0 - d1) / LOOK_BACK
    slid_in = any(s.who in (a, b) and s.t0 - JOIN <= t0 <= s.t1 + SLIDE_INTO for s in slid)
    # 몸을 던지는 중에 먼저 닿고 0.1초 뒤에 눕기도 한다 — 닿아 있는 동안 누웠는지 본다
    lying = any(b.lying for path in (pa, pb) for t, b in path if t0 <= t <= t1)
    hand = _hand(pa, pb, t0, t1) if slid_in and lying else None
    fell = False
    for path in (pa, pb):
        start = _at(path, t0)
        if start is not None and not start.lying:
            fell = fell or any(b.lying for t, b in path if t0 < t <= t1 + 1.0)
    lying_at_start = bool((now_a and now_a.lying) or (now_b and now_b.lying))
    knocked = (closing is not None and closing >= CLOSING and fell
               and _legs(now_a) and _legs(now_b))
    if knocked and not lying_at_start:  # 서서 빠르게 부딪힌 뒤에 쓰러졌다 — 미끄러져 온 게 아니다
        kind = "collision"
    elif slid_in and lying and hand is not None and hand <= TAG_HAND:  # 손목을 봐야 태그
        kind = "slide_tag"
    else:
        kind = "contact"
    return Contact(round(t0, 2), round(t1, 2), a, b,
                   None if closing is None else round(closing, 2), fell, slid_in and lying, kind,
                   None if hand is None else round(hand, 2))


def _hand(pa, pb, t0: float, t1: float) -> Optional[float]:
    """닿아 있는 동안(+HAND_AFTER) 서 있는 사람 손목 ↔ 누운 사람 최소 거리 / 작은 사람 키.

    태그는 글러브로 한다. 주자 뒤에 선 심판은 화면에서 겹쳐도 손이 주자에게 가지 않는다
    (실측: 태그 0.03~0.21, 심판 0.44~0.45).
    """
    best: Optional[float] = None
    for t, a in pa:
        if not t0 - 0.05 <= t <= min(t1 + HAND_AFTER, t0 + HAND_WINDOW):
            continue
        b = _at(pb, t, tol=0.05)
        if b is None or a.lying == b.lying:
            continue
        lying, standing = (a, b) if a.lying else (b, a)
        wrists = [p for p in standing.pts[9:11] if p is not None]
        if not wrists:
            continue
        d = min(_dist(p, q) for p in wrists for q in lying.joints) / min(a.scale, b.scale)
        best = d if best is None else min(best, d)
    return best


def analyze(frames: Sequence[PeopleFrame],
            min_scale: float = 0.0) -> tuple[Scene, list[Contact], list[Slide]]:
    """한 번 추적해 접촉과 슬라이딩을 함께 낸다."""
    sc = scene(frames, min_scale)
    slid = _slides(sc)
    merged = _merge(slid, _cuts(frames))
    return sc, _contacts(sc, [s for s, _r in slid], merged), merged
