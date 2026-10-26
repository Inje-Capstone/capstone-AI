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
# 팔·다리·몸통 뼈(COCO 번호). 뼈 하나가 키의 BONE_MAX를 넘으면 여러 사람 관절이 섞인 것 —
# 실측(2026-10-27): 머리는 왼쪽 끝, 발목은 오른쪽 끝에 잡힌 뼈대가 '누운 사람'으로 읽혔다.
BONES = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
         (5, 11), (6, 12), (5, 6), (11, 12))
BONE_MAX = 0.75  # 웅크린 포수 허벅지·하체가 가린 덕아웃 선수 어깨 폭이 키의 절반을 넘는다
TOUCH = 0.15  # 두 사람 관절 사이 최소 거리 / 작은 사람 키 — 이하면 닿음
DEPTH_RATIO = 1.5  # 큰 사람 키 / 작은 사람 키 — 넘으면 다른 깊이(앞뒤로 떨어져 있다)
FOOT_GAP = 0.35  # 두 사람 발 높이 차 / 작은 사람 키 — 넘으면 다른 깊이
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


def same_depth(a: Body, b: Body) -> bool:
    small, big = sorted((a.scale, b.scale))
    return big <= DEPTH_RATIO * small and abs(a.y1 - b.y1) <= FOOT_GAP * small


def gap(a: Body, b: Body) -> float:
    """두 사람 관절 사이 가장 가까운 거리 / 작은 사람 키."""
    near = min(_dist(p, q) for p in a.joints for q in b.joints)
    return near / min(a.scale, b.scale)


def touching(a: Body, b: Body) -> bool:
    if not same_depth(a, b):
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
    distance: float  # 누운 채 미끄러진 거리(몸길이 배, 세상 좌표)


def _slides(sc: Scene) -> list[tuple[Slide, Body]]:
    """추적 번호마다의 슬라이딩(합치기 전)과 그 시작 몸 — 접촉은 번호로 찾으니 이걸 쓴다."""
    out = []
    for tid, path in sc.paths().items():
        runs: list[list[tuple[float, Body]]] = []
        for t, b in path:
            if not b.lying:
                continue
            if runs and t - runs[-1][-1][0] <= JOIN:
                runs[-1].append((t, b))
            else:
                runs.append([(t, b)])
        for run in runs:
            if run[-1][0] - run[0][0] + 1e-6 < SLIDE_MIN:  # 10fps 시각의 소수 오차
                continue
            scale = sum(b.world_scale for _t, b in run) / len(run)
            far = max(_dist(run[0][1].world, b.world) for _t, b in run) / scale
            if far >= SLIDE_DIST:
                out.append((Slide(round(run[0][0], 2), round(run[-1][0], 2), tid, round(far, 2)),
                            run[0][1]))
    return sorted(out, key=lambda sb: sb[0].t0)


def _merge(found: list[tuple[Slide, Body]]) -> list[Slide]:
    """한 슬라이딩이 여러 번 잡힌 것을 하나로 — 1초 안에 이어지고 같은 사람이거나 같은 자리."""
    out: list[tuple[Slide, Body]] = []
    for s, b in found:
        for k in range(len(out) - 1, -1, -1):
            p, pb = out[k]
            near = _dist(pb.world, b.world) <= 2 * max(pb.world_scale, b.world_scale)
            if (s.who == p.who or near) and s.t0 <= p.t1 + 1.0:
                out[k] = (Slide(p.t0, max(p.t1, s.t1), p.who, max(p.distance, s.distance)), pb)
                break
        else:
            out.append((s, b))
    return [s for s, _b in out]


def slides(frames: Sequence[PeopleFrame], min_scale: float = 0.0) -> list[Slide]:
    """누운 채(상체가 눕혀진 자세) 미끄러진 구간 — 슬라이딩·다이빙.

    넘어져 제자리에 누워 있는 건 뺀다.
    """
    return _merge(_slides(scene(frames, min_scale)))


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

    @property
    def why(self) -> str:
        speed = "못 봄" if self.closing is None else f"{self.closing:.1f}키/초"
        return (f"{self.t0:.1f}~{self.t1:.1f}s 사람 {self.a}·{self.b} 접촉 · 다가온 속도 {speed}"
                f"(기준 {CLOSING}) · 넘어짐 {'예' if self.fell else '아니오'}"
                f" · 미끄러져 닿음 {'예' if self.slid else '아니오'}")


def _contacts(sc: Scene, slid: Sequence[Slide]) -> list[Contact]:
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
    return _one_tag_per_slide(sorted(out, key=lambda c: (c.t0, c.a, c.b)))


def _one_tag_per_slide(found: list[Contact]) -> list[Contact]:
    """미끄러져 들어온 주자는 포수·심판과 함께 닿기도 한다 — 같은 주자의 태그는 한 번만."""
    out: list[Contact] = []
    for c in found:
        if c.kind == "slide_tag" and any(
            p.kind == "slide_tag" and {p.a, p.b} & {c.a, c.b} and c.t0 - p.t0 <= 1.0
            for p in out
        ):
            continue
        out.append(c)
    return out


def _legs(b: Optional[Body]) -> bool:
    """발목이 보인다 — 다리까지 화면에 든 사람. 상체만 보이는 클로즈업은 넘어짐을 못 믿는다."""
    return b is not None and (b.pts[15] is not None or b.pts[16] is not None)


def contacts(frames: Sequence[PeopleFrame], min_scale: float = 0.0) -> list[Contact]:
    """접촉 구간 — 같은 깊이에서 닿았고, 짧게(STATIC초 안) 닿은 것만."""
    sc = scene(frames, min_scale)
    return _contacts(sc, [s for s, _b in _slides(sc)])


def _describe(pa, pb, a: int, b: int, t0: float, t1: float, slid: Sequence[Slide]) -> Contact:
    before_a, before_b = _at(pa, t0 - LOOK_BACK), _at(pb, t0 - LOOK_BACK)
    now_a, now_b = _at(pa, t0), _at(pb, t0)
    closing = None
    if before_a and before_b and now_a and now_b:  # 그때그때 키로 나눠 줌에도 같은 값
        d0 = _dist(before_a.center, before_b.center) / min(before_a.scale, before_b.scale)
        d1 = _dist(now_a.center, now_b.center) / min(now_a.scale, now_b.scale)
        closing = (d0 - d1) / LOOK_BACK
    slid_in = any(s.who in (a, b) and s.t0 - JOIN <= t0 <= s.t1 + SLIDE_INTO for s in slid)
    lying = bool((now_a and now_a.lying) or (now_b and now_b.lying))
    fell = False
    for path in (pa, pb):
        start = _at(path, t0)
        if start is not None and not start.lying:
            fell = fell or any(b.lying for t, b in path if t0 < t <= t1 + 1.0)
    if slid_in and lying:
        kind = "slide_tag"
    elif closing is not None and closing >= CLOSING and fell and _legs(now_a) and _legs(now_b):
        kind = "collision"
    else:
        kind = "contact"
    return Contact(round(t0, 2), round(t1, 2), a, b,
                   None if closing is None else round(closing, 2), fell, slid_in and lying, kind)


def analyze(frames: Sequence[PeopleFrame],
            min_scale: float = 0.0) -> tuple[Scene, list[Contact], list[Slide]]:
    """한 번 추적해 접촉과 슬라이딩을 함께 낸다."""
    sc = scene(frames, min_scale)
    slid = _slides(sc)
    return sc, _contacts(sc, [s for s, _b in slid]), _merge(slid)
