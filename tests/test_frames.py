"""영상 프레임 읽기 — PyAV와 OpenCV가 같은 구간·같은 간격을 준다."""

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from app.adapters.video import frames as frames_module  # noqa: E402
from app.adapters.video.base import VideoAnalyzerError  # noqa: E402
from app.adapters.video.frames import open_frames, video_size  # noqa: E402

FPS = 30
SECONDS = 2


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    assert out.isOpened()
    for i in range(FPS * SECONDS):
        out.write(np.full((48, 64, 3), i % 255, np.uint8))
    out.release()
    return path


@pytest.fixture(params=["av", "cv2"])
def reader(request, monkeypatch):
    if request.param == "av":
        pytest.importorskip("av")
    else:
        real = frames_module._open_cv2
        monkeypatch.setattr(frames_module, "_open_av",
                            lambda _av, *args: real(*args))
    return open_frames


def test_reads_every_frame_with_its_size_and_rate(reader, video):
    source = reader(video)
    rows = list(source.frames)
    assert source.size == (64, 48)
    assert source.fps == pytest.approx(FPS)
    assert len(rows) == FPS * SECONDS
    assert rows[0][1].shape == (48, 64, 3)
    times = [t for t, _ in rows]
    assert times == sorted(times)


def test_target_fps_thins_and_range_cuts(reader, video):
    rows = list(reader(video, start=0.5, end=1.5, target_fps=10).frames)
    assert 9 <= len(rows) <= 11
    assert all(0.4 <= t <= 1.55 for t, _ in rows)
    gaps = np.diff([t for t, _ in rows])
    assert gaps == pytest.approx(0.1, abs=0.04)


def test_size_shrinks_frames(reader, video):
    _, img = next(reader(video, size=(32, 24)).frames)
    assert img.shape == (24, 32, 3)


def test_missing_file_raises_our_error(reader, tmp_path):
    with pytest.raises(VideoAnalyzerError):
        reader(tmp_path / "none.mp4")


def test_stream_that_does_not_start_at_zero_is_counted_from_zero(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "clip.ts"
    box = av.open(str(path), "w", format="mpegts")
    stream = box.add_stream("mpeg2video", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for i in range(FPS * SECONDS):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i % 255, np.uint8), format="bgr24")
        box.mux(stream.encode(frame))
    box.mux(stream.encode())
    box.close()
    assert av.open(str(path)).streams.video[0].start_time  # 이 테스트의 전제

    rows = list(open_frames(path).frames)
    assert rows[0][0] == pytest.approx(0.0, abs=0.1)
    part = list(open_frames(path, start=0.5, end=1.0).frames)
    assert part and all(0.5 <= t <= 1.0 for t, _ in part)


def test_video_size_reads_without_decoding(video, tmp_path):
    assert video_size(video) == (64, 48)
    with pytest.raises(VideoAnalyzerError):
        video_size(tmp_path / "none.mp4")


def test_stopping_early_releases_the_file(reader, video, tmp_path):
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(video.read_bytes())
    source = reader(copy)
    next(source.frames)
    source.frames.close()
    copy.unlink()  # Windows는 열려 있으면 지울 수 없다
    assert not copy.exists()
