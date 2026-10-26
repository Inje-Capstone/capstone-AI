"""영상 → 프레임별 투수 관절 좌표. 선택 설치: pip install -e ".[pose]" (rtmlib + onnxruntime, CPU)

rtmlib(Apache-2.0)의 RTMDet + RTMPose로 사람마다 COCO 17점을 뽑고, 그중 투수를 고른다.
투수 고르기 — 국내 중계의 기본 화면(중견수 쪽 카메라)에서 투수는 화면 가운데 아래쪽에
가장 크게 잡힌다. 타자·포수·심판은 위쪽에 작게 모여 있다. 다른 화면(느린 화면·클로즈업)은
이 조건에 안 맞아 저절로 빠진다.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

from app.adapters.video.base import VideoAnalyzerError
from app.domain.contact import PeopleFrame
from app.domain.pitch_motion import PoseFrame

Person = list[tuple[float, float, float]]

CENTER_BAND = (0.2, 0.8)  # 투수 몸 중심 x (화면 비율)
LOWER_MIN = 0.4  # 투수 몸 중심 y가 이보다 아래
MIN_HEIGHT = 0.12  # 투수 키(관절 범위)가 화면 높이의 이 비율 이상
TRACK_JUMP = 0.12  # 앞 프레임 투수와 이만큼(화면 너비) 이상 떨어지면 다른 사람
FIELD_MIN = 0.15  # 화면에서 잔디·흙 색 비율 — 이보다 적으면 경기장 화면이 아니다
CAM_WIDTH = 320  # 카메라 움직임은 이 너비로 줄여 잰다
CAM_POINTS = 150  # 따라갈 배경 특징점 수
CAM_INLIERS = 12  # 같은 움직임에 맞는 특징점이 이보다 적으면(흐림) 앞 움직임을 이어 쓴다
CAM_HOLD = 6  # 앞 움직임은 이 프레임까지만 이어 쓴다 — 밋밋한 잔디 화면에서 밀려가지 않게
CUT_HIST = 0.5  # 앞 프레임과 색 분포 상관 — 이보다 낮으면 화면 전환
# 한 프레임(1/30초)에 화면이 이 배 넘게 커지거나 작아지면 줌이 아니라 화면 전환(와이프·로고 넘김).
# 실측(2026-10-27): 중계 줌은 3초에 4배 = 프레임당 1.016배, 와이프는 한 번에 1.3배 튀었다.
ZOOM_JUMP = 1.1


def field_ratio(img: Any) -> float:
    """잔디(초록)·흙(주황 갈색) 화소 비율. 점수판 유무와 상관없이 '경기장 화면'을 가른다.

    실측(2026-10-26): 보크 해설 영상은 화면 대부분이 해설자라 해설자의 모은 손이 투수 세트로
    잡혔다. 점수판으로 거르면 직캠·해설 속 다시보기를 잃는다 — 색으로 거른다.
    """
    import cv2

    small = cv2.resize(img, (64, 36))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (h >= 30) & (h <= 90) & (s_ >= 50) & (v >= 40)
    dirt = (h >= 5) & (h <= 25) & (s_ >= 60) & (v >= 60)
    return float((grass | dirt).mean())


def _center_y(pts: list[tuple[float, float, float]], height: int) -> Optional[float]:
    ys = [y for _, y, s in pts if s >= 0.3]
    return sum(ys) / len(ys) / height if len(ys) >= 5 else None


def pick_pitcher(
    people: list[list[tuple[float, float, float]]],
    width: int,
    height: int,
    previous: Optional[tuple[float, float]] = None,
) -> Optional[int]:
    """사람들 중 투수 번호. 조건에 맞는 사람이 없으면 None."""
    best, best_key = None, None
    for k, pts in enumerate(people):
        good = [(x, y) for x, y, s in pts if s >= 0.3]
        if len(good) < 8:
            continue
        xs, ys = [p[0] for p in good], [p[1] for p in good]
        cx, cy = sum(xs) / len(xs) / width, sum(ys) / len(ys) / height
        h = (max(ys) - min(ys)) / height
        w = (max(xs) - min(xs)) / width * width / height  # 화면 높이 기준 너비
        if not (CENTER_BAND[0] <= cx <= CENTER_BAND[1] and cy >= LOWER_MIN and h >= MIN_HEIGHT):
            continue
        if w > h * 1.6:  # 누운 사람(슬라이딩 주자) — 다른 장면
            continue
        above = sum(
            1
            for other in people
            if other is not pts
            and _center_y(other, height) is not None
            and _center_y(other, height) < cy - 0.1
        )
        if above < 2:  # 투수 위쪽에 타자·포수·심판이 없으면 중견수 쪽 화면이 아니다
            continue
        near = previous is not None and abs(cx - previous[0]) < TRACK_JUMP
        key = (near, h)
        if best_key is None or key > best_key:
            best, best_key = k, key
    return best


class _PoseRunner:
    """rtmlib 모델을 한 번 올리고, 영상을 fps로 솎아 경기장 화면마다 사람들의 관절을 낸다."""

    def __init__(self, fps: float = 15.0, mode: str = "balanced") -> None:
        try:
            from rtmlib import Body
        except ImportError as exc:
            raise VideoAnalyzerError('자세 분석은 pip install -e ".[pose]"가 필요하다') from exc
        self.fps = fps
        self._body = Body(mode=mode, backend="onnxruntime", device="cpu")
        self.size = (0, 0)  # 마지막 영상의 (너비, 높이)

    def _people(
        self, video: Path, start: float, end: Optional[float]
    ) -> Iterator[tuple[float, Optional[list[Person]]]]:
        """(시각, 사람들) — 경기장이 아닌 화면(스튜디오·해설자)은 사람들 대신 None."""
        import cv2

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise VideoAnalyzerError(f"영상을 열 수 없다: {video}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / self.fps))
        self.size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        idx = 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if end is not None and t > end:
                    break
                idx += 1
                if idx % step:
                    continue
                if field_ratio(img) < FIELD_MIN:
                    yield t, None
                    continue
                keypoints, scores = self._body(img)
                yield t, [
                    [(float(x), float(y), float(s)) for (x, y), s in zip(kp, sc)]
                    for kp, sc in zip(keypoints, scores)
                ]
        finally:
            cap.release()


class PitcherPoseExtractor(_PoseRunner):
    def extract(
        self, video: Path, start: float = 0.0, end: Optional[float] = None
    ) -> list[PoseFrame]:
        frames: list[PoseFrame] = []
        previous: Optional[tuple[float, float]] = None
        for t, people in self._people(video, start, end):
            if people is None:  # 스튜디오·해설자 화면 — 경기장이 아니다
                previous = None
                continue
            width, height = self.size
            k = pick_pitcher(people, width, height, previous)
            if k is None:
                previous = None
                continue
            pts = people[k]
            good = [(x, y) for x, y, s in pts if s >= 0.3]
            previous = (
                sum(p[0] for p in good) / len(good) / width,
                sum(p[1] for p in good) / len(good) / height,
            )
            frames.append(PoseFrame(round(t, 3), tuple(pts)))
        return frames


class PeoplePoseExtractor(_PoseRunner):
    """화면 속 모든 사람 — 선수 접촉·충돌·슬라이딩(app/domain/contact.py)용.

    경기장이 아닌 화면은 건너뛴다. 그 공백이 contact.track에서 장면 전환으로 읽힌다.
    """

    def extract(
        self, video: Path, start: float = 0.0, end: Optional[float] = None
    ) -> list[PeopleFrame]:
        return [
            PeopleFrame(round(t, 3), tuple(tuple(p) for p in people), height=self.size[1])
            for t, people in self._people(video, start, end)
            if people is not None
        ]


Camera = tuple[float, float, float, float]  # 화면 = [[a,-b],[b,a]]·세상 + (tx, ty)


def camera_motion(
    video: Path, start: float = 0.0, end: Optional[float] = None
) -> list[tuple[float, Camera, bool]]:
    """프레임마다 (시각, 장면 시작 화면 → 지금 화면의 닮음 변환, 화면 전환 여부).

    중계 카메라는 돌고(패닝) 당긴다(줌). 실측(2026-10-27 홈 송구 클립): 0.1초에 화면 너비의
    1/4을 돌면서 3초 동안 4배 당겼다. 배경 특징점을 광류로 따라가 이동·확대·회전을 RANSAC으로
    맞추면 움직이는 선수는 어긋난 점으로 빠진다. 흐려서 점을 놓치면 앞 움직임을 이어 쓴다.
    화면 전환은 색 분포로 가르고 거기서 처음으로 되돌린다(장면이 다르면 좌표도 다르다).
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise VideoAnalyzerError(f"영상을 열 수 없다: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CAM_WIDTH
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAM_WIDTH
    small = (CAM_WIDTH, max(16, round(CAM_WIDTH * height / width)))
    ratio = width / CAM_WIDTH
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    out: list[tuple[float, Camera, bool]] = []
    prev_gray, prev_hist = None, None
    total, step = np.eye(3), np.eye(3)
    blind = 0  # 특징점이 모자라 앞 움직임을 이어 쓴 프레임 수
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if end is not None and t > end:
                break
            tiny = cv2.resize(img, small)
            gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)], [0, 1], None,
                                [16, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            cut = prev_gray is None
            if prev_gray is not None:
                move, inliers = _similarity(cv2, np, prev_gray, gray)
                same = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                zoom = (move[0, 0] ** 2 + move[1, 0] ** 2) ** 0.5
                jump = inliers >= CAM_INLIERS and not 1 / ZOOM_JUMP <= zoom <= ZOOM_JUMP
                cut = same < CUT_HIST or (inliers < CAM_INLIERS and same < 0.9) or jump
                if cut:
                    total, step, blind = np.eye(3), np.eye(3), 0
                else:
                    if inliers >= CAM_INLIERS:
                        step, blind = move, 0
                        step[:2, 2] *= ratio
                    else:
                        blind += 1
                        if blind > CAM_HOLD:
                            step = np.eye(3)
                    total = step @ total
            a, b, tx, ty = total[0, 0], total[1, 0], total[0, 2], total[1, 2]
            cam = (round(a, 4), round(b, 4), round(tx, 1), round(ty, 1))
            out.append((round(t, 3), cam, cut))
            prev_gray, prev_hist = gray, hist
    finally:
        cap.release()
    return out


def _similarity(cv2: Any, np: Any, prev: Any, cur: Any) -> tuple[Any, int]:
    """두 프레임 사이 배경의 닮음 변환(3×3)과 그에 맞는 특징점 수."""
    p0 = cv2.goodFeaturesToTrack(prev, CAM_POINTS, 0.01, 6)
    if p0 is None:
        return np.eye(3), 0
    p1, status, _err = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    good = status.ravel() == 1
    if good.sum() < CAM_INLIERS:
        return np.eye(3), 0
    m, mask = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                          ransacReprojThreshold=2.0)
    if m is None or mask is None:
        return np.eye(3), 0
    return np.vstack([m, [0.0, 0.0, 1.0]]), int(mask.sum())


def attach_camera(
    frames: list[PeopleFrame], motion: list[tuple[float, Camera, bool]]
) -> list[PeopleFrame]:
    """사람 관절 프레임에 그 시각의 카메라 변환과, 앞 프레임 뒤 화면 전환이 있었는지를 붙인다."""
    if not motion:
        return frames
    out, k = [], 0
    for n, f in enumerate(frames):
        cut = False
        while k + 1 < len(motion) and motion[k + 1][0] <= f.t + 1e-3:
            k += 1
            cut = cut or (n > 0 and motion[k][2])
        a, b, tx, ty = motion[k][1]
        out.append(PeopleFrame(f.t, f.people, (a, b, tx, ty), cut, f.height))
    return out


def frames_to_json(frames: list[PoseFrame]) -> list[dict[str, Any]]:
    return [
        {"t": f.t, "p": [[round(x, 1), round(y, 1), round(s, 2)] for x, y, s in f.points]}
        for f in frames
    ]


def frames_from_json(rows: list[dict[str, Any]]) -> list[PoseFrame]:
    return [PoseFrame(float(r["t"]), tuple(tuple(p) for p in r["p"])) for r in rows]


def people_to_json(frames: list[PeopleFrame], size: tuple[int, int]) -> dict[str, Any]:
    return {
        "size": list(size),
        "frames": [
            {
                "t": f.t,
                "people": [
                    [[round(x, 1), round(y, 1), round(s, 2)] for x, y, s in person]
                    for person in f.people
                ],
            }
            for f in frames
        ],
    }


def people_from_json(doc: dict[str, Any]) -> list[PeopleFrame]:
    height = int((doc.get("size") or [0, 0])[1])
    return [
        PeopleFrame(
            float(f["t"]),
            tuple(tuple(tuple(p) for p in person) for person in f["people"]),
            height=height,
        )
        for f in doc["frames"]
    ]
