"""점수판 전용 판독 — 읽을 순간 고르기(로컬)와 응답 정리. API·영상 없이."""
import pytest

from app.adapters.video.scorebug_reader import (
    THUMB_H,
    THUMB_W,
    crop_filter,
    normalize_reading,
    plan_reads,
)

SIZE = THUMB_W * THUMB_H


def _thumb(graphic: int, background: int, digit: int = 0) -> bytes:
    """왼쪽 절반 = 점수판 그래픽(고정), 오른쪽 절반 = 배경. digit = 숫자 변화."""
    px = bytearray(SIZE)
    for i in range(SIZE):
        px[i] = graphic if (i % THUMB_W) < THUMB_W // 2 else background
    for i in range(digit * 200):  # 그래픽 안 일부 화소가 바뀜 (카운트 숫자)
        px[(i // (THUMB_W // 2)) * THUMB_W + i % (THUMB_W // 2)] = 255 - graphic
    return bytes(px)


def test_reads_only_moments_with_score_bug_and_splits_on_change():
    live = [_thumb(50, 120)] * 4  # 점수판 보임
    replay = [_thumb(200, 30)] * 5  # 느린 화면 — 점수판 없음, 화면 전체가 다름
    after = [_thumb(50, 120, digit=1)] * 3  # 점수판 다시, 숫자 바뀜
    plan = plan_reads(live + replay + after, ref=1)
    assert [start for start, _ in plan] == [0, 9]


def test_plan_respects_max_reads_keeping_time_order():
    frames = []
    for k in range(6):  # 점수판이 6번 나타났다 사라짐
        frames += [_thumb(50, 120, digit=k)] * 2 + [_thumb(200, 30)] * 2
    plan = plan_reads(frames, ref=0, max_reads=3)
    assert len(plan) == 3 and plan == sorted(plan)


def test_normalize_reading_accepts_korean_halves_and_drops_invisible():
    raw = {"visible": True, "inning": "6", "half": "bottom", "balls": 3, "strikes": 1,
           "outs": 0, "bases": [3, 1, 2, 4], "away": 1, "home": 2}
    r = normalize_reading(raw, 3.0)
    assert r is not None and (r["half"], r["bases"], r["inning"]) == ("bot", [1, 2, 3], 6)
    assert normalize_reading({"visible": False}, 1.0) is None
    assert normalize_reading({"inning": 1, "half": "??"}, 1.0) is None


def test_crop_filter_scales_corner():
    assert crop_filter("top_left").startswith("crop=iw*0.45:ih*0.32:iw*0.0:ih*0.0,scale=iw*3")


def test_read_span_reports_video_time():
    """경기 영상은 VLM이 본 구간만 읽는다 — 판독 시각·프레임 시각은 구간 시작만큼 밀린다."""
    from app.adapters.video.scorebug_reader import ScorebugReader

    reader = ScorebugReader("K", "http://x", "m")
    asked = {}
    reader.pick_corner = lambda video, duration, start: ("top_left", start + 1.0)

    def thumbs(video, corner, start, duration):
        asked["thumbs"] = (start, duration)
        return [_thumb(50, 120)] * 3

    frames = []
    reader._thumbs = thumbs
    reader._frame = lambda video, t, vf: frames.append(t) or b"jpg"
    reader._ask = lambda images, text, max_tokens: {
        "visible": True, "inning": 2, "half": "top", "balls": 0, "strikes": 1,
        "outs": 1, "bases": [], "away": 0, "home": 0}
    out = reader.read("g.mp4", 30.0, start=600.0)
    assert asked["thumbs"] == (600.0, 30.0)
    assert [r["t"] for r in out["scoreboard"]] == [600.0]
    assert frames == [600.5]


def test_periodic_plan_reads_every_n_seconds_and_on_reappearance():
    """경기 영상: 보이는 동안 N초마다, 2초 넘게 가렸다 다시 보이면 바로 — 1초마다 읽지 않는다."""
    from app.adapters.video.scorebug_reader import plan_periodic

    live = [_thumb(50, 120)] * 20  # 20초 동안 보임
    replay = [_thumb(200, 30)] * 3  # 3초 가림 (느린 화면)
    blink = [_thumb(200, 30)]  # 1초 깜빡임은 새 장면이 아니다
    frames = live + replay + live[:10] + blink + live[:10]
    plan = plan_periodic(frames, ref=0, every=8)
    assert [s for s, _ in plan] == [0, 8, 16, 23, 31, 39]
    assert len(plan_periodic(frames, ref=0, every=8, max_reads=3)) == 3


def test_pick_corner_skips_a_corner_that_does_not_read_as_a_score_bug():
    """영상 첫 장면(팀 소개 그래픽)에서 고른 모서리가 점수판이 아니면 다음 시각에서 다시 고른다."""
    from app.adapters.video.scorebug_reader import ScorebugReader

    reader = ScorebugReader("K", "http://x", "m")
    reader._frame = lambda video, t, vf: f"{t:.0f}".encode()
    answers = iter([{"image": 3}, {"visible": False},  # 1초: 왼쪽 아래를 골랐지만 안 읽힘
                    {"image": 1}, {"visible": True, "inning": 1, "half": "top", "balls": 0,
                                   "strikes": 0, "outs": 0, "bases": [], "away": 0,
                                   "home": 0}])
    reader._ask = lambda images, text, max_tokens: next(answers)
    corner, t = reader.pick_corner("g.mp4", 100.0)
    assert corner == "top_left" and t == 30.0


def test_combined_plan_keeps_both_periodic_and_change_reads():
    from app.adapters.video.scorebug_reader import plan_both, plan_periodic

    # 점수판이 1초만 가렸다 다시 보임 — 간격 읽기는 2초 넘게 가려져야 다시 읽는다
    frames = [_thumb(50, 120)] * 5 + [_thumb(200, 30)] + [_thumb(50, 120, digit=1)] * 4
    periodic = plan_periodic(frames, ref=0, every=8)
    both = plan_both(frames, ref=0, every=8)
    assert set(periodic) <= set(both) and len(both) > len(periodic)
    assert [at for _s, at in both] == sorted(at for _s, at in both)


def test_requests_keep_a_minimum_gap(monkeypatch):
    """분당 한도(429)를 넘지 않게 미리 천천히 보낸다 — 다시 보내지 않는다."""
    from app.adapters.video import scorebug_reader as sr

    clock = [100.0]
    slept = []
    monkeypatch.setattr(sr.time, "monotonic", lambda: clock[0])
    def sleep(sec):
        slept.append(sec)
        clock[0] += sec

    monkeypatch.setattr(sr.time, "sleep", sleep)
    monkeypatch.setattr(sr, "post_json",
                        lambda *a, **k: {"choices": [{"message": {"content": "{}"}}]})
    reader = sr.ScorebugReader("K", "http://x", "m", min_gap=1.6)
    reader._ask([b"a"], "p", 8)
    clock[0] += 0.5
    reader._ask([b"a"], "p", 8)
    assert slept == [pytest.approx(1.1)] and reader.requests == 2


def test_request_gap_holds_across_threads(monkeypatch):
    """실시간은 조각을 여러 줄로 읽는다 — 동시에 들어와도 간격을 지킨다(독립 리뷰 재현)."""
    import threading

    from app.adapters.video import scorebug_reader as sr

    sent = []
    monkeypatch.setattr(sr, "post_json", lambda *a, **k: sent.append(sr.time.monotonic()) or {
        "choices": [{"message": {"content": "{}"}}]})
    reader = sr.ScorebugReader("K", "http://x", "m", min_gap=0.2)
    threads = [threading.Thread(target=reader._ask, args=([b"a"], "p", 8)) for _ in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    gaps = [b - a for a, b in zip(sorted(sent), sorted(sent)[1:])]
    assert len(sent) == 3 and min(gaps) >= 0.17  # 윈도우 시계 해상도(약 15ms)만큼 여유


def test_single_frame_past_the_end_of_a_short_segment_is_a_read_error(tmp_path):
    from app.adapters.video import scorebug_reader as sr
    from app.adapters.video.base import VideoAnalyzerError
    from app.adapters.video.scorebug_layout import Layout

    path = tmp_path / "bar.json"
    path.write_text('{"name": "b", "crop": [0.2, 0.9, 0.3, 0.06], "bases": {"1": [0.5, 0.5]},'
                    ' "outs": []}', encoding="utf-8")
    reader = sr.ScorebugReader("K", "http://x", "m", layout=Layout.load(path))

    def missing(*_args):
        raise FileNotFoundError("f.jpg")

    reader._frame = missing
    with pytest.raises(VideoAnalyzerError):
        reader.read_frame("seg.mp4", 2.5)
    reader._frame = lambda video, t, vf: b"jpg"
    reader._ask = lambda images, text, max_tokens: {
        "visible": True, "inning": 1, "half": "top", "balls": 0, "strikes": 0, "outs": 0,
        "bases": [], "away": 0, "home": 0}
    reader._rgb = lambda video, t: b""
    with pytest.raises(VideoAnalyzerError):
        reader.read_frame("seg.mp4", 2.5)
