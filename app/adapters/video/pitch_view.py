"""투구 화면 찾기와 타자 스윙 재기 — 해설도 점수판도 보지 않는다.

1. 작은 그림(64×36, 4fps)으로 화면 전환을 자르고 비슷한 화면끼리 묶는다. 움직임이 가장 적은
   큰 묶음이 중계의 고정 투구 화면이다. 영상을 다 풀어도 영상 1초당 0.08초다.
2. 투구 화면마다 한 장을 사람 검출에 넣어 포수(웅크린 사람)를 찾고, 그 옆에 서 있는 사람을
   타자로 본다. 포수가 기준점이다 — 사람 위치만으로 고르면 3분의 1이 관중을 잘랐다(실측).
3. 투수 관절로 투구가 있었는지 보고, 있으면 타자 손을 잰다. 투구가 없는 화면은 판정하지 않는다.
4. 관절은 검출을 빼고 자리 상자만 모델에 넣는다. 엉덩이를 따라 상자를 옮긴다. 검출까지
   돌리면 프레임당 약 290ms, 빼면 14ms다(실측 2026-11-01).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.adapters.video.frames import open_frames
from app.adapters.video.pose import _PoseRunner
from app.domain.swing import Pitch, Swing, measure, pitch_moment

SCAN_FPS = 4
THUMB = (64, 36)
CUT = 22.0  # 프레임 사이 평균 밝기 차 — 넘으면 화면 전환
GROUPS = 10
MIN_SHOT = 1.0  # 이보다 짧은 화면은 투구를 담지 못한다
MIN_GROUP_SEC = 3.0
MIN_GROUP_SHARE = 0.05  # 투구 화면 묶음은 영상 길이의 이 비율은 넘는다
SIMILAR_GROUP = 1.5  # 가장 잔잔한 묶음의 이 배까지 함께 본다 — 광고판이 바뀌면 묶음이 갈린다
ROLE_FRACS = (0.05, 0.2, 0.5)  # 화면에서 역할을 찾아 볼 위치
MIN_POINTS = 8
GOOD = 0.3
CROUCH = 0.55  # 엉덩이~무릎 세로 / 몸통 — 이보다 짧으면 웅크린 자세
FOLLOW_LIMIT = 0.7  # 처음 자리에서 키의 이만큼 넘게 벗어난 관절은 따라가지 않는다
CROP = 256
FRAME_FPS = 30
MAX_SHOT_SEC = 30.0
PITCHER_STEP = 2  # 투수는 15fps면 충분하다
MIN_STANDING = 0.6  # 타자 키 / 투수 키 — 이보다 작으면 멀리 있는 수비·덕아웃이다
FEET_MARGIN = 0.02  # 타자 발은 포수 발과 같거나 앞이다(포수 키 비율). 뒤에 선 사람은 심판이다

Box = list[float]


@dataclass(frozen=True)
class Shot:
    t0: float
    t1: float
    group: int
    motion: float


@dataclass(frozen=True)
class Roles:
    pitcher: Box
    catcher: Box
    batter: Box


def split_shots(tiny: Any, times: Sequence[float]) -> list[Shot]:
    """작은 그림 열 → 화면들. 묶음은 아직 정하지 않는다(group=-1)."""
    import numpy as np

    if len(tiny) < 2:
        return []
    gray = tiny.mean(axis=3)
    diff = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    cuts = [0] + [i + 1 for i, d in enumerate(diff) if d > CUT] + [len(tiny)]
    return [Shot(float(times[a]), float(times[b - 1]), -1, float(diff[a:b - 1].mean()))
            for a, b in zip(cuts, cuts[1:]) if b - a >= 2]


def group_shots(tiny: Any, times: Sequence[float], shots: list[Shot]) -> list[Shot]:
    """화면 평균 그림을 k-means로 묶는다."""
    import cv2
    import numpy as np

    if len(shots) < 2:
        return [Shot(s.t0, s.t1, 0, s.motion) for s in shots]
    stamps = np.asarray(times)
    means = np.array([tiny[(stamps >= s.t0) & (stamps <= s.t1)].mean(axis=0).reshape(-1)
                      for s in shots], np.float32)
    cv2.setRNGSeed(0)
    stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, _ = cv2.kmeans(means, min(GROUPS, len(shots)), None, stop, 5,
                              cv2.KMEANS_PP_CENTERS)
    return [Shot(s.t0, s.t1, int(k), s.motion) for s, k in zip(shots, labels.ravel())]


def pitch_groups(shots: Sequence[Shot]) -> set[int]:
    """충분히 긴 묶음 중 움직임이 적은 것들. 진짜 투구 화면인지는 포수를 찾아 가린다."""
    total = sum(s.t1 - s.t0 for s in shots)
    need = max(MIN_GROUP_SEC, total * MIN_GROUP_SHARE)
    groups: dict[int, list[Shot]] = {}
    for s in shots:
        groups.setdefault(s.group, []).append(s)
    big = {k: sum(s.motion for s in rows) / len(rows) for k, rows in groups.items()
           if sum(s.t1 - s.t0 for s in rows) >= need}
    if not big:
        return set()
    still = min(big.values())
    return {k for k, motion in big.items() if motion <= still * SIMILAR_GROUP}


def _box(person: Sequence[tuple[float, float, float]]) -> Optional[Box]:
    good = [q for q in person if q[2] >= GOOD]
    if len(good) < MIN_POINTS:
        return None
    xs, ys = [q[0] for q in good], [q[1] for q in good]
    return [min(xs), min(ys), max(xs), max(ys)]


def _crouch(p: Sequence[tuple[float, float, float]]) -> float:
    if min(p[i][2] for i in (5, 6, 11, 12, 13, 14)) < GOOD:
        return 9.0
    shoulder = (p[5][1] + p[6][1]) / 2
    hip = (p[11][1] + p[12][1]) / 2
    knee = (p[13][1] + p[14][1]) / 2
    return abs(knee - hip) / max(abs(hip - shoulder), 5.0)


def find_roles(people: Sequence[Sequence[tuple[float, float, float]]]) -> Optional[Roles]:
    """투구 화면 한 장 → 투수·포수·타자. 포수를 못 찾으면 투구 화면이 아니다."""
    rows = [(b, _crouch(p)) for b, p in ((_box(p), p) for p in people) if b is not None]
    if len(rows) < 3:
        return None
    pitcher = max(rows, key=lambda r: r[0][3] - r[0][1])  # 앞쪽이라 가장 크다
    tall = pitcher[0][3] - pitcher[0][1]
    rest = [r for r in rows if r is not pitcher and r[0][3] < pitcher[0][3]
            and r[0][3] - r[0][1] >= 0.3 * tall]
    if len(rest) < 2:
        return None
    for catcher in sorted((r for r in rest if r[1] <= CROUCH), key=lambda r: r[1]):
        box = catcher[0]
        width, middle = box[2] - box[0], (box[0] + box[2]) / 2
        standing = [r for r in rest if r is not catcher and r[1] > CROUCH
                    and r[0][3] - r[0][1] >= MIN_STANDING * tall
                    and r[0][3] >= box[3] - FEET_MARGIN * (box[3] - box[1])
                    and abs((r[0][0] + r[0][2]) / 2 - middle) < 4.5 * width]
        if standing:
            batter = max(standing, key=lambda r: (r[0][3], abs((r[0][0] + r[0][2]) / 2 - middle)))
            return Roles(pitcher[0], box, batter[0])
    return None


def crop_square(box: Box, scale: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(box[3] - box[1], box[2] - box[0]) * scale
    return (int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)),
            int(min(cx + side / 2, size[0])), int(min(cy + side / 2, size[1])))


def pitch_shots(video: Path, start: float = 0.0, end: Optional[float] = None) -> list[Shot]:
    """영상 → 투구 화면들."""
    import numpy as np

    rows = list(open_frames(video, start, end, SCAN_FPS, THUMB).frames)
    if not rows:
        return []
    times = [t for t, _ in rows]
    tiny = np.array([img for _, img in rows])
    shots = group_shots(tiny, times, split_shots(tiny, times))
    groups = pitch_groups(shots)
    return [s for s in shots if s.group in groups and s.t1 - s.t0 >= MIN_SHOT]


class SwingExtractor(_PoseRunner):
    """투구 화면마다 타자 스윙을 잰다. 결과 시각은 영상 기준 초."""

    def __init__(self, mode: str = "balanced") -> None:
        super().__init__(fps=FRAME_FPS, mode=mode)

    def _roles(self, video: Path, shot: Shot) -> Optional[Roles]:
        for frac in ROLE_FRACS:
            t = shot.t0 + (shot.t1 - shot.t0) * frac
            source = open_frames(video, t, t + 0.5)
            first = next(source.frames, None)
            source.frames.close()
            if first is None:
                continue
            self.size = source.size
            keypoints, scores = self._body(first[1])
            people = [[(float(x), float(y), float(s)) for (x, y), s in zip(kp, sc)]
                      for kp, sc in zip(keypoints, scores)]
            found = find_roles(people)
            if found:
                return found
        return None

    def _crops(
        self, video: Path, shot: Shot, roles: Roles
    ) -> tuple[list[float], list[Any], list[Any], tuple[int, ...], tuple[int, ...]]:
        """(시각들, 타자 그림들, 투수 그림들, 타자 자리, 투수 자리). 크기를 맞춰 잘라 둔다."""
        import cv2

        bat = crop_square(roles.batter, 2.0, self.size)
        pit = crop_square(roles.pitcher, 1.6, self.size)
        end = min(shot.t1, shot.t0 + MAX_SHOT_SEC)
        times, bats, pits = [], [], []
        for t, img in open_frames(video, shot.t0, end, FRAME_FPS).frames:
            times.append(t)
            bats.append(cv2.resize(img[bat[1]:bat[3], bat[0]:bat[2]], (CROP, CROP),
                                   interpolation=cv2.INTER_AREA))
            pits.append(cv2.resize(img[pit[1]:pit[3], pit[0]:pit[2]], (CROP, CROP),
                                   interpolation=cv2.INTER_AREA))
        return times, bats, pits, bat, pit

    def _follow(
        self, crops: Sequence[Any], box: Box, area: tuple[int, ...]
    ) -> list[list[tuple[float, float, float]]]:
        """검출 없이 한 사람 관절을 잇는다. box는 원본 좌표, area는 잘라낸 자리."""
        import numpy as np

        fx, fy = CROP / max(area[2] - area[0], 1), CROP / max(area[3] - area[1], 1)
        home = np.array([((box[0] + box[2]) / 2 - area[0]) * fx,
                         ((box[1] + box[3]) / 2 - area[1]) * fy])
        tall = (box[3] - box[1]) * fy
        half_w, half_h = tall * 0.75, tall * 0.72
        at = home.copy()
        rows = []
        for crop in crops:
            place = np.array([[max(at[0] - half_w, 0), max(at[1] - half_h, 0),
                               min(at[0] + half_w, CROP - 1), min(at[1] + half_h, CROP - 1)]])
            keypoints, scores = self._body.pose_model(crop, bboxes=place)
            rows.append([(float(x), float(y), float(s))
                         for (x, y), s in zip(keypoints[0], scores[0])])
            hips = keypoints[0][[11, 12]][scores[0][[11, 12]] >= GOOD]
            if len(hips):  # 손은 휘둘러도 몸통 자리는 천천히 움직인다
                seen = np.array([hips[:, 0].mean(), hips[:, 1].mean()])
                if np.hypot(*(seen - home)) < tall * FOLLOW_LIMIT:
                    at = at * 0.5 + seen * 0.5
        return rows

    def measure_shot(self, video: Path, shot: Shot) -> Optional[tuple[float, Pitch, Swing]]:
        """(시각, 투구, 스윙). 역할이나 투구를 못 찾은 화면은 None — 안 쳤다는 뜻이 아니다."""
        roles = self._roles(video, shot)
        if roles is None:
            return None
        times, bats, pits, bat_area, pit_area = self._crops(video, shot, roles)
        pitch = pitch_moment(self._follow(pits[::PITCHER_STEP], roles.pitcher, pit_area))
        if pitch is None:
            return None
        found = measure(self._follow(bats, roles.batter, bat_area))
        k = found.frame if found.swing else pitch.frame * PITCHER_STEP
        return round(times[min(k, len(times) - 1)], 3), pitch, found

    def extract(
        self, video: Path, start: float = 0.0, end: Optional[float] = None
    ) -> list[dict[str, Any]]:
        out = []
        for shot in pitch_shots(video, start, end):
            found = self.measure_shot(video, shot)
            if found:
                t, pitch, swing = found
                out.append({"t": t, "t0": shot.t0, "t1": shot.t1, "lift": pitch.lift,
                            "moved": swing.moved, "swing": swing.swing, "clear": swing.clear})
        return out
