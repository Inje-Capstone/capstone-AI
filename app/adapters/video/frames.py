"""영상 프레임 읽기. PyAV가 있으면 여러 스레드로 풀고, 없으면 OpenCV로 읽는다.

프레임 읽기는 분석 시간의 큰 몫이다. 실측(2026-11-01, 1080p60 AV1, 다른 분석이 도는 중,
영상 1초당): OpenCV 0.81초, OpenCV 하드웨어 가속 0.18초, PyAV 4스레드 0.12초.
H.264 클립에선 차이가 작다(0.08초 → 0.07초).

어느 쪽으로 읽어도 같은 그림·같은 시각이어야 한다 — 기준값(경기장 색 비율 등)이 OpenCV로 맞춰졌고
팀원마다 PyAV 유무가 다르다. 그래서 PyAV 쪽은 색 변환을 OpenCV와 맞추고, 시각은 0부터 센다.
회전 정보가 있는 영상(세로 직캠)은 OpenCV만 돌려 주므로 그쪽으로 넘긴다.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.adapters.video.base import VideoAnalyzerError

THREADS = 4


@dataclass
class Frames:
    fps: float
    size: tuple[int, int]  # 원본 (너비, 높이)
    frames: Iterator[tuple[float, Any]]  # (시각, BGR 배열)


def open_frames(
    video: Path,
    start: float = 0.0,
    end: Optional[float] = None,
    target_fps: Optional[float] = None,
    size: Optional[tuple[int, int]] = None,
) -> Frames:
    """start~end를 읽는다. `target_fps`면 그만큼 솎고, `size`면 그 크기로 줄여서 준다."""
    try:
        import av
    except ImportError:
        return _open_cv2(video, start, end, target_fps, size)
    return _open_av(av, video, start, end, target_fps, size)


def video_size(video: Path) -> tuple[int, int]:
    """(너비, 높이). 프레임을 풀지 않는다."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise VideoAnalyzerError(f"영상을 열 수 없다: {video}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def _step(fps: float, target_fps: Optional[float]) -> int:
    return max(1, round(fps / target_fps)) if target_fps else 1


def _open_av(
    av: Any,
    video: Path,
    start: float,
    end: Optional[float],
    target_fps: Optional[float],
    size: Optional[tuple[int, int]],
) -> Frames:
    try:
        box = av.open(str(video))
    except (av.error.FFmpegError, OSError) as exc:
        raise VideoAnalyzerError(f"영상을 열 수 없다: {video}") from exc
    if not box.streams.video:
        box.close()
        raise VideoAnalyzerError(f"영상 트랙이 없다: {video}")
    stream = box.streams.video[0]
    width, height = stream.codec_context.width, stream.codec_context.height
    if video_size(video) != (width, height):
        box.close()
        return _open_cv2(video, start, end, target_fps, size)
    stream.thread_type = "AUTO"
    stream.thread_count = THREADS
    fps = float(stream.average_rate or 30.0)
    step = _step(fps, target_fps)
    origin = float((stream.start_time or 0) * stream.time_base)  # TS·mkv는 0에서 시작하지 않는다

    def frames() -> Iterator[tuple[float, Any]]:
        try:
            if start > 0:
                box.seek(int((start + origin) / stream.time_base), stream=stream)
            idx = 0
            for frame in box.decode(stream):
                if frame.time is None or frame.time - origin < start:  # 앞 키프레임부터 풀었다
                    continue
                t = frame.time - origin
                if end is not None and t > end:
                    break
                idx += 1
                if idx % step:
                    continue
                width_height = size or (width, height)
                shown = frame.reformat(*width_height, format="bgr24", src_colorspace="ITU601")
                yield t, shown.to_ndarray(format="bgr24")
        except av.error.FFmpegError:  # 뒤가 잘린 파일 — OpenCV처럼 읽은 데까지만 쓴다
            return
        finally:
            box.close()

    return Frames(fps, (width, height), frames())


def _open_cv2(
    video: Path,
    start: float,
    end: Optional[float],
    target_fps: Optional[float],
    size: Optional[tuple[int, int]],
) -> Frames:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise VideoAnalyzerError(f"영상을 열 수 없다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = _step(fps, target_fps)
    shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    def frames() -> Iterator[tuple[float, Any]]:
        try:
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
                yield t, cv2.resize(img, size) if size else img
        finally:
            cap.release()

    return Frames(fps, shape, frames())
