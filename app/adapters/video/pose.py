"""영상 → 프레임별 투수 관절 좌표. 선택 설치: pip install -e ".[pose]" (rtmlib + onnxruntime, CPU)

rtmlib(Apache-2.0)의 RTMDet + RTMPose로 사람마다 COCO 17점을 뽑고, 그중 투수를 고른다.
투수 고르기 — 국내 중계의 기본 화면(중견수 쪽 카메라)에서 투수는 화면 가운데 아래쪽에
가장 크게 잡힌다. 타자·포수·심판은 위쪽에 작게 모여 있다. 다른 화면(느린 화면·클로즈업)은
이 조건에 안 맞아 저절로 빠진다.
"""

from pathlib import Path
from typing import Any, Optional

from app.adapters.video.base import VideoAnalyzerError
from app.domain.pitch_motion import PoseFrame

CENTER_BAND = (0.2, 0.8)  # 투수 몸 중심 x (화면 비율)
LOWER_MIN = 0.4  # 투수 몸 중심 y가 이보다 아래
MIN_HEIGHT = 0.12  # 투수 키(관절 범위)가 화면 높이의 이 비율 이상
TRACK_JUMP = 0.12  # 앞 프레임 투수와 이만큼(화면 너비) 이상 떨어지면 다른 사람


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


class PitcherPoseExtractor:
    def __init__(self, fps: float = 15.0, mode: str = "balanced") -> None:
        try:
            from rtmlib import Body
        except ImportError as exc:
            raise VideoAnalyzerError('자세 분석은 pip install -e ".[pose]"가 필요하다') from exc
        self.fps = fps
        self._body = Body(mode=mode, backend="onnxruntime", device="cpu")

    def extract(
        self, video: Path, start: float = 0.0, end: Optional[float] = None
    ) -> list[PoseFrame]:
        import cv2

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise VideoAnalyzerError(f"영상을 열 수 없다: {video}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / self.fps))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        frames: list[PoseFrame] = []
        previous: Optional[tuple[float, float]] = None
        idx = 0
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
            keypoints, scores = self._body(img)
            people = [
                [(float(x), float(y), float(s)) for (x, y), s in zip(kp, sc)]
                for kp, sc in zip(keypoints, scores)
            ]
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
        cap.release()
        return frames


def frames_to_json(frames: list[PoseFrame]) -> list[dict[str, Any]]:
    return [
        {"t": f.t, "p": [[round(x, 1), round(y, 1), round(s, 2)] for x, y, s in f.points]}
        for f in frames
    ]


def frames_from_json(rows: list[dict[str, Any]]) -> list[PoseFrame]:
    return [PoseFrame(float(r["t"]), tuple(tuple(p) for p in r["p"])) for r in rows]
