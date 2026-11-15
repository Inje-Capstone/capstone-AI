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


def _bar_layout(tmp_path, extra=""):
    from app.adapters.video.scorebug_layout import Layout

    path = tmp_path / "box.json"
    path.write_text('{"name": "box", "crop": [0.04, 0.07, 0.14, 0.09], "bases": {"1": [0.5, 0.5]},'
                    ' "outs": [[0.6, 0.6]]' + extra + '}', encoding="utf-8")
    return Layout.load(path)


def _token_pixels(monkeypatch, sr, states):
    """_rgb가 시각마다 글자 하나를 주고, 그 글자를 (주자, 아웃)으로 읽게 바꾼다 — 실제 화소 없이."""
    size = sr.RGB_W * sr.RGB_H * 3
    monkeypatch.setattr(sr, "rgb_sampler", lambda px, w, h: px[:1])
    monkeypatch.setattr(sr, "decode", lambda layout, s: states[s])
    return lambda token: token * size


def test_layout_prompt_note_is_added_to_the_read_prompt(tmp_path):
    """볼카운트를 'X-Y'로 그리는 판(MBC 상자)은 VLM이 그것을 점수로도 읽었다 —
    그 판만 안내를 덧붙인다."""
    from app.adapters.video.scorebug_reader import READ_PROMPT_V2, ScorebugReader

    plain = ScorebugReader("K", "http://x", "m", layout=_bar_layout(tmp_path),
                           prompt=READ_PROMPT_V2)
    assert plain.prompt == READ_PROMPT_V2
    noted = ScorebugReader("K", "http://x", "m", prompt=READ_PROMPT_V2, layout=_bar_layout(
        tmp_path, ', "prompt_note": "  The middle X-Y is the count.  "'))
    assert noted.prompt == READ_PROMPT_V2 + " The middle X-Y is the count."


def test_settled_moves_a_run_read_to_when_its_pixels_first_match(tmp_path, monkeypatch):
    """덩어리 시작(10초)으로 찍던 판독 — 주자 화소가 15초에 바뀌었으면 15초. 끝까지 같으면 10초."""
    import app.adapters.video.scorebug_reader as sr

    px = _token_pixels(monkeypatch, sr, {b"A": ([], 0), b"B": ([1], 0)})
    reader = sr.ScorebugReader("K", "http://x", "m", layout=_bar_layout(tmp_path))
    reader._rgb = lambda video, t: px(b"A" if t < 15 else b"B")
    assert reader._settled("g.mp4", 0.0, 10, 20, ([1], 0)) == 15.0
    reader._rgb = lambda video, t: px(b"B")
    assert reader._settled("g.mp4", 0.0, 10, 20, ([1], 0)) == 10.0
    reader._rgb = lambda video, t: b"" if t < 17 else px(b"B")  # 프레임을 못 읽으면 거기서 멈춘다
    assert reader._settled("g.mp4", 0.0, 10, 20, ([1], 0)) == 17.0


def test_change_read_is_stamped_after_the_state_settles(tmp_path, monkeypatch):
    """한국시리즈 5차전: 220초부터 이어 보인 점수판을 247초 뒤에 읽고 220초로 찍었다(1·2루)
    — 247초로."""
    import app.adapters.video.scorebug_reader as sr

    px = _token_pixels(monkeypatch, sr, {b"A": ([1], 0), b"B": ([1, 2], 0)})
    reader = sr.ScorebugReader("K", "http://x", "m", every=8, changes=True,
                               layout=_bar_layout(tmp_path))
    reader.find_in_layout = lambda video, duration, start: (reader.layout.crop, 0.0)
    reader._thumbs = lambda video, corner, start, duration: [b"t"] * 40
    monkeypatch.setattr(sr, "plan_both", lambda thumbs, ref, every, max_reads: [(20, 20), (20, 30)])
    reader.pixel_states = lambda video, start, duration, thumbs, ref: []  # 다른 시험에서 본다
    reader._frame = lambda video, t, vf: b"jpg"
    reader._ask = lambda images, text, max_tokens: {
        "visible": True, "inning": 3, "half": "top", "balls": 0, "strikes": 0, "outs": 0,
        "bases": [], "away": 0, "home": 0}
    reader._rgb = lambda video, t: px(b"A" if t < 227 else b"B")
    out = reader.read("g.mp4", 40.0, start=200.0)
    assert [(r["t"], r["bases"]) for r in out["scoreboard"]] == [(220.0, [1]), (227.0, [1, 2])]


def test_prompt_note_null_and_unreadable_frame_while_settling(tmp_path, monkeypatch):
    """독립 리뷰 — prompt_note: null은 'None'을 붙이지 않고, 되짚다 프레임을 못 읽으면
    거기서 멈춘다."""
    import subprocess

    import app.adapters.video.scorebug_reader as sr

    assert _bar_layout(tmp_path, ', "prompt_note": null').prompt_note == ""
    px = _token_pixels(monkeypatch, sr, {b"B": ([1], 0)})
    reader = sr.ScorebugReader("K", "http://x", "m", layout=_bar_layout(tmp_path))

    def rgb(video, t):
        if t < 16:
            raise subprocess.CalledProcessError(1, "ffmpeg")
        return px(b"B")

    reader._rgb = rgb
    assert reader._settled("g.mp4", 0.0, 10, 20, ([1], 0)) == 16.0


def test_layout_reference_tries_more_moments_when_the_first_six_show_no_board(tmp_path):
    """PO 3차전 — 5·20·35·50·65·80% 여섯 장면이 모두 점수판 없는 장면이라 판독을 아예 못 했다.
    여섯이 안 되면 사이 장면을 더 본다(앞 여섯은 그대로라 예전 기준 화면은 안 바뀐다)."""
    import app.adapters.video.scorebug_reader as sr

    reader = sr.ScorebugReader("K", "http://x", "m", layout=_bar_layout(tmp_path))
    asked = []
    reader._frame = lambda video, t, vf: asked.append(t) or b"jpg"

    def ask(images, text, max_tokens):
        if len(asked) < 8:
            return {"visible": False}
        return {"visible": True, "inning": 1, "half": "top", "balls": 0, "strikes": 0, "outs": 0,
                "bases": [], "away": 0, "home": 0}

    reader._ask = ask
    corner, t = reader.find_in_layout("g.mp4", 100.0)
    assert corner == reader.layout.crop and t == 15.0  # 여섯 실패 → 10%, 15%
    assert asked[:6] == [5.0, 20.0, 35.0, 50.0, 65.0, 80.0]


def test_state_segments_without_a_read_get_one_more_read():
    """PO 3차전 — 안타 뒤 '무사 1루'(3초)·삼진 뒤 '1사 1루'를 8초 간격 판독이 못 읽어 두 플레이가
    한 전이로 합쳐졌다. 화소 상태가 2초 넘게 이어진 새 구간에 읽어 낸 판독이 없으면 한 장 더."""
    from app.adapters.video.scorebug_reader import plan_state_changes

    a, b, c = ((), 0, (1, 1)), ((1,), 0, (0, 0)), ((1,), 1, (3, 2))
    states = [a] * 5 + [None] * 2 + [b] * 3 + [None] + [c] * 4 + [a]  # 마지막 a는 1초뿐
    assert plan_state_changes(states, [(0.0, (), 0)]) == [(7, 8), (11, 12)]
    assert plan_state_changes(states, [(0.0, (), 0), (8.0, (1,), 0), (13.0, (1,), 1)]) == []
    # 바로 앞 옛 상태 판독은 새 구간을 읽은 것이 아니다(KS5)
    assert plan_state_changes(states, [(0.0, (), 0), (6.5, (), 0)]) == [(7, 8), (11, 12)]
    # 가렸다 다시 보인 같은 상태는 다시 읽지 않는다
    assert plan_state_changes([a] * 3 + [None] * 3 + [a] * 3, [(1.0, (), 0)]) == []


def test_reader_reads_state_segments_it_missed(tmp_path, monkeypatch):
    """판독기 — 계획대로 읽은 뒤 읽어 낸 판독이 없는 화소 상태 구간을 한 장씩 더 읽는다."""
    import app.adapters.video.scorebug_reader as sr

    reader = sr.ScorebugReader("K", "http://x", "m", every=8, changes=True,
                               layout=_bar_layout(tmp_path))
    reader.find_in_layout = lambda video, duration, start: (reader.layout.crop, 0.0)
    reader._thumbs = lambda video, corner, start, duration: [b"t"] * 20
    monkeypatch.setattr(sr, "plan_both", lambda thumbs, ref, every, max_reads: [(0, 0)])
    one = ((), 0, None)
    two = ((1,), 0, None)
    reader.pixel_states = lambda video, start, duration, thumbs, ref: [one] * 10 + [two] * 10
    asked = []
    reader._frame = lambda video, t, vf: asked.append(t) or b"jpg"
    reader._ask = lambda images, text, max_tokens: {
        "visible": True, "inning": 1, "half": "top", "balls": 0, "strikes": 0, "outs": 0,
        "bases": [], "away": 0, "home": 0}
    px = _token_pixels(monkeypatch, sr, {b"A": ([], 0), b"B": ([1], 0)})
    reader._rgb = lambda video, t: px(b"A" if t < 10 else b"B")
    out = reader.read("g.mp4", 20.0)
    assert asked == [0.5, 11.5]
    assert [(r["t"], r["bases"]) for r in out["scoreboard"]] == [(0.0, []), (10.0, [1])]


def test_state_reads_keep_runner_and_out_changes_when_the_budget_is_short():
    """독립 리뷰 — 예산이 모자라면 볼카운트만 바뀐 구간보다 주자·아웃이 바뀐 구간을 먼저 읽는다."""
    from app.adapters.video.scorebug_reader import plan_state_changes

    count1, count2 = ((), 0, (1, 0)), ((), 0, (2, 0))
    runner, out = ((1,), 0, (0, 0)), ((1,), 1, (0, 0))
    states = [((), 0, (0, 0))] * 3 + [count1] * 3 + [count2] * 3 + [runner] * 3 + [out] * 3
    reads = [(0.0, (), 0), (4.0, (), 0), (7.0, (), 0)]  # 볼카운트만 바뀐 두 구간은 읽었다
    assert plan_state_changes(states, reads) == [(9, 10), (12, 13)]
    hidden = [((), 0, (0, 0))] * 3 + [None] * 2 + [count1] * 3 + [None] * 2 + [runner] * 3 + [
        None] * 2 + [out] * 3
    assert plan_state_changes(hidden, []) == [(0, 1), (5, 6), (10, 11), (15, 16)]
    assert plan_state_changes(hidden, [], limit=2) == [(0, 1), (10, 11)]
    assert plan_state_changes(hidden, [], limit=3) == [(0, 1), (10, 11), (15, 16)]


def test_pixel_states_skip_hidden_seconds_and_pad_a_short_stream(tmp_path, monkeypatch):
    """화소 상태 — 점수판이 안 보일 법한 초는 None, 스트림이 일찍 끝나면 나머지도 None."""
    import io

    import app.adapters.video.scorebug_reader as sr

    px = _token_pixels(monkeypatch, sr, {b"A": ([], 0), b"B": ([1], 1)})
    stream = px(b"A") + px(b"A") + px(b"B")  # 3초만 나오는 스트림

    class FakeProc:
        def __init__(self, *args, **kwargs):
            self.stdout = io.BytesIO(stream)

        def wait(self):
            return 0

    monkeypatch.setattr(sr.subprocess, "Popen", FakeProc)
    reader = sr.ScorebugReader("K", "http://x", "m", layout=_bar_layout(tmp_path))
    shown, hidden = _thumb(50, 120), _thumb(200, 30)
    states = reader.pixel_states("g.mp4", 0.0, 4.0, [shown, hidden, shown, shown], 0)
    assert states == [((), 0, None), None, ((1,), 1, None), None]

    def no_stream(*args, **kwargs):  # 점수판이 보일 법한 초가 없으면 영상을 열지 않는다
        raise AssertionError("영상을 흘려 읽으면 안 된다")

    monkeypatch.setattr(sr.subprocess, "Popen", no_stream)
    assert reader.pixel_states("g.mp4", 0.0, 2.0, [shown, hidden], 0) == [None, None]
