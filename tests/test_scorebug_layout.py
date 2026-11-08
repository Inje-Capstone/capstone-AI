"""점수판 주자·아웃을 화소 색으로 — 위치 파일 + 색 판정. 영상·API 없이."""

import json
from pathlib import Path

from app.adapters.video.scorebug_layout import Layout, decode, is_filled, rgb_sampler

ROOT = Path(__file__).resolve().parents[1]
RED, WHITE, PALE = (0.82, 0.08, 0.17), (0.97, 0.97, 0.97), (0.86, 0.62, 0.66)


def test_filled_means_saturated_color_not_white_or_pale_dot():
    assert is_filled(RED)
    assert not is_filled(WHITE)
    assert not is_filled(PALE)  # 0아웃의 흐린 점 (리허설 채도 0.33~0.36)
    assert not is_filled((0.05, 0.0, 0.0))  # 검은 화면


def test_decode_reads_each_diamond_by_position():
    bases = {1: (0.3, 0.1), 2: (0.29, 0.07), 3: (0.28, 0.1)}
    layout = Layout("t", bases, ((0.33, 0.11), (0.34, 0.11)))
    colors = {(0.3, 0.1): RED, (0.29, 0.07): WHITE, (0.28, 0.1): RED,
              (0.33, 0.11): RED, (0.34, 0.11): PALE}
    assert decode(layout, lambda p: colors[p]) == ([1, 3], 1)


def test_rgb_sampler_averages_a_patch_in_ratio_coordinates():
    w, h = 40, 20
    px = bytearray([255, 255, 255] * (w * h))
    for y in range(8, 13):  # (0.5, 0.5) 둘레를 빨갛게
        for x in range(18, 23):
            px[(y * w + x) * 3:(y * w + x) * 3 + 3] = bytes([210, 20, 44])
    sample = rgb_sampler(bytes(px), w, h)
    assert is_filled(sample((0.5, 0.5)))
    assert not is_filled(sample((0.1, 0.1)))


def test_shipped_layout_loads_with_three_bases_and_out_dots():
    layout = Layout.load(ROOT / "data" / "scorebug_layouts" / "sbs_sports.json")
    assert sorted(layout.bases) == [1, 2, 3] and len(layout.outs) == 2
    assert all(0 < x < 1 and 0 < y < 1 for x, y in [*layout.bases.values(), *layout.outs])


def test_reader_takes_bases_and_outs_from_pixels_when_layout_given(tmp_path):
    """VLM이 주자·아웃을 틀려도(리허설 0~1/6) 위치 파일이 있으면 화소 값이 이긴다."""
    from app.adapters.video import scorebug_reader as sr

    path = tmp_path / "l.json"
    path.write_text(json.dumps({"name": "t", "bases": {"1": [0.5, 0.5], "2": [0.1, 0.1],
                                                       "3": [0.9, 0.9]},
                                "outs": [[0.5, 0.5], [0.1, 0.9]]}), encoding="utf-8")
    reader = sr.ScorebugReader("K", "http://x", "m", layout=Layout.load(path))
    w, h = sr.RGB_W, sr.RGB_H
    px = bytearray([250, 250, 250] * (w * h))
    cx, cy = w // 2, h // 2
    for y in range(cy - 3, cy + 4):
        for x in range(cx - 3, cx + 4):
            px[(y * w + x) * 3:(y * w + x) * 3 + 3] = bytes([210, 20, 44])
    reader.pick_corner = lambda video, duration, start: ("top_left", start + 1.0)
    thumb = bytes([50] * sr.THUMB_W * sr.THUMB_H)
    reader._thumbs = lambda video, corner, start, duration: [thumb] * 2
    reader._frame = lambda video, t, vf: b"jpg"
    reader._rgb = lambda video, t: bytes(px)
    reader._ask = lambda images, text, max_tokens: {
        "visible": True, "inning": 3, "half": "bot", "balls": 1, "strikes": 2,
        "outs": 0, "bases": [2, 3], "away": 1, "home": 4}
    (r,) = reader.read("g.mp4", 2.0)["scoreboard"]
    assert (r["bases"], r["outs"]) == ([1], 1)  # 화소: 1루만 빨강, 아웃 점 하나
    assert (r["balls"], r["strikes"], r["home"]) == (1, 2, 4)  # 숫자는 VLM 그대로


def test_bright_mode_counts_white_squares_not_red_or_dark():
    """MBC 아래 띠는 주자 칸을 흰색으로 채운다(실측 밝기 0.92~1.00, 빈 칸 0.11~0.14)."""
    assert is_filled(WHITE, "bright")
    assert not is_filled(RED, "bright")
    assert not is_filled((0.12, 0.12, 0.13), "bright")


def test_bar_layout_reads_white_bases_but_red_out_dots():
    layout = Layout.load(ROOT / "data" / "scorebug_layouts" / "mbc_sports_plus.json")
    assert layout.crop is not None and layout.scale == 1.5 and layout.base_fill == "bright"
    b1, b2, b3 = layout.bases[1], layout.bases[2], layout.bases[3]
    o1, o2 = layout.outs
    colors = {b1: WHITE, b2: (0.12, 0.12, 0.13), b3: WHITE, o1: RED, o2: WHITE}
    assert decode(layout, lambda p: colors[p]) == ([1, 3], 1)


def test_reader_with_a_crop_layout_skips_corner_picking(tmp_path):
    """띠 점수판은 위치 파일의 영역을 그대로 자른다 — 모서리 고르기 요청을 쓰지 않는다."""
    from app.adapters.video import scorebug_reader as sr

    path = tmp_path / "bar.json"
    path.write_text(json.dumps({"name": "bar", "crop": [0.2, 0.9, 0.3, 0.06], "scale": 1.5,
                                "bases": {"1": [0.3, 0.95], "2": [0.29, 0.94],
                                          "3": [0.28, 0.95]},
                                "outs": [[0.4, 0.95], [0.41, 0.95]]}), encoding="utf-8")
    reader = sr.ScorebugReader("K", "http://x", "m", layout=Layout.load(path))

    def no_pick(*_args):
        raise AssertionError("모서리를 고르면 안 된다")

    reader.pick_corner = no_pick
    filters = []
    thumb = bytes([50] * sr.THUMB_W * sr.THUMB_H)
    reader._thumbs = lambda video, corner, start, duration: [thumb] * 2
    reader._frame = lambda video, t, vf: filters.append(vf) or b"jpg"
    reader._rgb = lambda video, t: bytes([20, 20, 20] * sr.RGB_W * sr.RGB_H)
    reader._ask = lambda images, text, max_tokens: {
        "visible": True, "inning": 1, "half": "top", "balls": 0, "strikes": 0,
        "outs": 2, "bases": [1], "away": 0, "home": 0}
    out = reader.read("g.mp4", 2.0)
    assert out["corner"] == "layout" and len(out["scoreboard"]) == 1
    assert all(f.startswith("crop=iw*0.3:ih*0.06:iw*0.2:ih*0.9,scale=iw*1.5") for f in filters)
    r = out["scoreboard"][0]
    assert (r["bases"], r["outs"]) == ([], 0)  # 화소(검정)가 VLM 답을 이긴다


def _triangle(up: bool, color=(220, 20, 60), bg=(245, 245, 245)) -> bytes:
    """1280×720 화면 (600, 300) 둘레에 ▲(up) 또는 ▼."""
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W

    px = bytearray(bytes(bg) * (RGB_W * RGB_H))
    for row in range(10):
        half = row if up else 9 - row  # ▲는 아래로 갈수록 넓다
        for x in range(600 - half, 601 + half):
            i = ((300 + row) * RGB_W + x) * 3
            px[i:i + 3] = bytes(color)
    return bytes(px)


def test_arrow_pixels_decide_top_or_bottom_of_the_inning():
    from app.adapters.video.scorebug_layout import decode_half
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W

    box = (590 / RGB_W, 295 / RGB_H, 612 / RGB_W, 315 / RGB_H)
    layout = Layout("t", {}, (), half=box)
    assert decode_half(layout, _triangle(up=True), RGB_W, RGB_H) == "top"
    assert decode_half(layout, _triangle(up=False), RGB_W, RGB_H) == "bot"
    blank = bytes((245, 245, 245) * (RGB_W * RGB_H))
    assert decode_half(layout, blank, RGB_W, RGB_H) is None  # 화살표가 없으면 VLM 답을 둔다
    white = Layout("t", {}, (), half=box, half_fill="bright")
    dark_up = _triangle(up=True, color=(250, 250, 250), bg=(15, 15, 15))
    assert decode_half(white, dark_up, RGB_W, RGB_H) == "top"
    assert decode_half(Layout("t", {}, ()), dark_up, RGB_W, RGB_H) is None  # 칸이 없는 위치 파일


def test_shipped_layouts_have_arrow_boxes():
    for name in ("sbs_sports_2026", "mbc_sports_plus", "tving_postseason", "kbo_topleft_box"):
        layout = Layout.load(ROOT / "data" / "scorebug_layouts" / f"{name}.json")
        x0, y0, x1, y1 = layout.half
        assert 0 < x0 < x1 < 1 and 0 < y0 < y1 < 1


def test_single_frame_read_for_live_segments_uses_pixels_for_bases_outs_and_half(tmp_path):
    """실시간 조각은 한 장만 읽는다 — 숫자는 VLM, 주자·아웃·초말은 화소."""
    from app.adapters.video import scorebug_reader as sr

    path = tmp_path / "bar.json"
    path.write_text(json.dumps({"name": "bar", "crop": [0.2, 0.9, 0.3, 0.06],
                                "bases": {"1": [0.9, 0.9], "2": [0.91, 0.9], "3": [0.92, 0.9]},
                                "outs": [[0.93, 0.9], [0.94, 0.9]],
                                "half": [590 / sr.RGB_W, 295 / sr.RGB_H, 612 / sr.RGB_W,
                                         315 / sr.RGB_H]}), encoding="utf-8")
    reader = sr.ScorebugReader("K", "http://x", "m", layout=Layout.load(path))
    reader._frame = lambda video, t, vf: b"jpg"
    reader._rgb = lambda video, t: _triangle(up=False)  # ▼ = 말, 주자·아웃 칸은 흰 배경
    reader._ask = lambda images, text, max_tokens: {
        "visible": True, "inning": 7, "half": "top", "balls": 1, "strikes": 2,
        "outs": 2, "bases": [1, 2], "away": 3, "home": 4}
    r = reader.read_frame("seg.ts", 2.5)
    assert (r["t"], r["half"], r["bases"], r["outs"], r["balls"]) == (2.5, "bot", [], 0, 1)
    reader._ask = lambda images, text, max_tokens: {"visible": False}
    assert reader.read_frame("seg.ts", 2.5) is None


def test_lit_arrow_position_decides_half_when_both_arrows_are_drawn():
    """▲·▼가 늘 그려지고 켜진 쪽만 색칠 — 색칠된 화소가 칸 위쪽이면 초, 아래쪽이면 말."""
    from app.adapters.video.scorebug_layout import decode_half
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W

    box = (590 / RGB_W, 290 / RGB_H, 612 / RGB_W, 340 / RGB_H)  # 칸 가운데 y=315
    layout = Layout("t", {}, (), half=box, half_by="position")
    upper = _triangle(up=True)  # y 300~309 — 위쪽
    assert decode_half(layout, upper, RGB_W, RGB_H) == "top"
    lower = bytearray(bytes((245, 245, 245)) * (RGB_W * RGB_H))
    for y in range(325, 333):  # 아래쪽에 작은 빨간 덩어리(모양은 따지지 않는다)
        for x in range(598, 604):
            i = (y * RGB_W + x) * 3
            lower[i:i + 3] = bytes((220, 20, 60))
    assert decode_half(layout, bytes(lower), RGB_W, RGB_H) == "bot"
    boxed = Layout.load(ROOT / "data" / "scorebug_layouts" / "kbo_topleft_box.json")
    assert boxed.half_by == "position"
    assert boxed.mark is not None and boxed.base_hue == (20.0, 60.0)


def test_strikeout_mark_over_the_bases_is_found_and_red_text_is_not_a_runner():
    """삼진 표시('nK' 빨간 글자)가 주자 칸을 덮으면 표시로 읽고, 노란 주자 칸으로 읽지 않는다."""
    from app.adapters.video.scorebug_layout import decode_mark, hue_in
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W

    box = (590 / RGB_W, 295 / RGB_H, 612 / RGB_W, 315 / RGB_H)
    layout = Layout("t", {3: (600 / RGB_W, 305 / RGB_H)}, (), base_hue=(20.0, 60.0), mark=box)
    red = _triangle(up=True)  # 빨간 화소 100개
    assert decode_mark(layout, red, RGB_W, RGB_H) == "strikeout"
    assert decode(layout, rgb_sampler(red, RGB_W, RGB_H))[0] == []
    yellow = _triangle(up=True, color=(240, 190, 30))  # 노란 주자 칸
    assert decode_mark(layout, yellow, RGB_W, RGB_H) is None
    assert decode(layout, rgb_sampler(yellow, RGB_W, RGB_H))[0] == [3]
    assert decode_mark(Layout("t", {}, ()), red, RGB_W, RGB_H) is None  # 칸이 없는 위치 파일
    assert hue_in((1.0, 0.0, 0.05), (340.0, 20.0)) and not hue_in((0.94, 0.75, 0.12), (340.0, 20.0))


def test_position_half_needs_a_clear_side_and_layout_boxes_are_checked(tmp_path):
    """켜진 화소가 칸 가운데면(둘 다 켜짐) 모른다. 칸은 숫자 4개, half_by는 둘 중 하나."""
    import pytest

    from app.adapters.video.scorebug_layout import decode_half
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W

    box = (590 / RGB_W, 280 / RGB_H, 612 / RGB_W, 330 / RGB_H)  # 칸 가운데 y=305 = ▲ 가운데
    assert decode_half(Layout("t", {}, (), half=box, half_by="position"),
                       _triangle(up=True), RGB_W, RGB_H) is None
    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "x", "bases": {}, "outs": [], "mark": [0.1, 0.1, 0.2]}',
                   encoding="utf-8")
    with pytest.raises(ValueError):
        Layout.load(bad)
    bad.write_text('{"name": "x", "bases": {}, "outs": [], "half_by": "postion"}', encoding="utf-8")
    with pytest.raises(ValueError):
        Layout.load(bad)


def test_dark_strikeout_mark_counts_black_text():
    """검은 'nK'가 볼카운트·아웃 칸을 덮는 중계 — mark_fill dark면 어두운 화소를 센다."""
    from app.adapters.video.scorebug_layout import decode_mark
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W

    box = (590 / RGB_W, 295 / RGB_H, 612 / RGB_W, 315 / RGB_H)
    dark = Layout("t", {}, (), mark=box, mark_fill="dark")
    black = _triangle(up=True, color=(10, 10, 10))
    assert decode_mark(dark, black, RGB_W, RGB_H) == "strikeout"
    assert decode_mark(dark, _triangle(up=True), RGB_W, RGB_H) is None  # 빨간 점은 표시가 아니다
    red_text = Layout("t", {}, (), mark=box)  # 기본은 빨간 글자
    assert decode_mark(red_text, black, RGB_W, RGB_H) is None
    caster = Layout.load(ROOT / "data" / "scorebug_layouts" / "kbo_caster_box.json")
    assert caster.mark_fill == "dark" and caster.half is None


def test_white_out_dots_mean_the_strikeout_mark_covers_them():
    """표시가 나타나는 중엔 글자가 모자라도 아웃 점 자리가 흰 바탕이다 — cover_val로 잡는다."""
    from app.adapters.video.scorebug_layout import decode_mark
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W

    outs = ((100 / RGB_W, 100 / RGB_H), (120 / RGB_W, 100 / RGB_H))
    box = (0.9, 0.9, 0.95, 0.95)  # 글자 칸은 비었다
    layout = Layout("t", {}, outs, mark=box, cover_val=0.85)
    white = bytes((250, 250, 250)) * (RGB_W * RGB_H)
    gray = bytes((185, 185, 185)) * (RGB_W * RGB_H)  # 꺼진 아웃 점(밝기 0.73)
    assert decode_mark(layout, white, RGB_W, RGB_H) == "strikeout"
    assert decode_mark(layout, gray, RGB_W, RGB_H) is None
    plain = Layout("t", {}, outs, mark=box)  # 옵션이 없으면 아웃 점은 안 본다
    assert decode_mark(plain, white, RGB_W, RGB_H) is None


def test_dark_lit_arrow_and_new_layouts_load():
    """SPOTV2는 켜진 화살표를 검게 칠한다(half_fill dark). 새 위치 파일 넷은 모두 읽힌다."""
    from app.adapters.video.scorebug_layout import decode_half
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W

    box = (590 / RGB_W, 290 / RGB_H, 612 / RGB_W, 340 / RGB_H)
    dark = Layout("t", {}, (), half=box, half_fill="dark", half_by="position")
    assert decode_half(dark, _triangle(up=True, color=(15, 15, 15)), RGB_W, RGB_H) == "top"
    for name in ("spotv2_box", "kbo_caster_box_up", "kbo_box_bottomright", "bar_two_line"):
        layout = Layout.load(ROOT / "data" / "scorebug_layouts" / f"{name}.json")
        assert layout.mark is not None and len(layout.outs) == 2 and len(layout.bases) == 3


def test_dot_count_is_read_from_pixels_only_when_the_layout_has_dots():
    """B●●● S●● 점수판 — 켜진 점 수가 볼·스트라이크. 점 위치가 없으면 None(VLM 숫자를 쓴다)."""
    from app.adapters.video.scorebug_layout import decode_count

    lit, off = (0.95, 0.45, 0.1), (0.75, 0.75, 0.75)
    balls = ((0.1, 0.5), (0.2, 0.5), (0.3, 0.5))
    strikes = ((0.5, 0.5), (0.6, 0.5))
    colors = {balls[0]: lit, balls[1]: off, balls[2]: off, strikes[0]: lit, strikes[1]: lit}
    layout = Layout("t", {}, (), balls=balls, strikes=strikes)
    assert decode_count(layout, lambda p: colors[p]) == (1, 2)
    assert decode_count(Layout("t", {}, ()), lambda p: colors[p]) is None
    spotv2 = Layout.load(ROOT / "data" / "scorebug_layouts" / "spotv2_box.json")
    assert len(spotv2.balls) == 3 and len(spotv2.strikes) == 2


def test_mark_scan_counts_only_frames_with_the_board_and_merges_one_mark(monkeypatch):
    """화소 훑기 — 바탕 색(panel)이 맞는 장면에서만 표시를 세고, 붙어 있는 표시는 하나로."""
    import pytest

    np = pytest.importorskip("numpy")  # 영상 의존성(numpy·cv2)이 없는 CI에서는 건너뛴다
    pytest.importorskip("cv2")

    import app.adapters.video.frames as frames_mod
    from app.adapters.video.scorebug_layout import board_shown, rgb_sampler
    from app.adapters.video.scorebug_reader import RGB_H, RGB_W, scan_marks

    panel = ((0.1, 0.1, 0.9, 0.9, 0.9),)
    box = (0.5, 0.5, 0.52, 0.53)
    layout = Layout("t", {}, (), mark=box, panel=panel)

    def frame(board: bool, mark: bool):
        img = np.zeros((RGB_H, RGB_W, 3), np.uint8)  # BGR
        if board:
            img[60:84, 120:140] = (230, 230, 230)
        if mark:
            img[362:380, 642:662] = (40, 20, 220)  # 빨간 글자
        return img

    seq = [(0.0, frame(True, False)), (0.5, frame(True, True)), (1.0, frame(True, True)),
           (1.5, frame(False, True)), (5.0, frame(True, True))]

    class Fake:
        frames = iter(seq)

    monkeypatch.setattr(frames_mod, "open_frames", lambda *a, **k: Fake())
    assert scan_marks(layout, ROOT / "x.mp4") == [0.5, 5.0]
    assert scan_marks(Layout("t", {}, (), mark=box), ROOT / "x.mp4") == []  # panel 없으면 안 훑음
    import cv2
    shown = cv2.cvtColor(frame(True, False), cv2.COLOR_BGR2RGB).tobytes()
    assert board_shown(layout, rgb_sampler(shown, RGB_W, RGB_H))
    assert not board_shown(Layout("t", {}, ()), rgb_sampler(shown, RGB_W, RGB_H))
    for name in ("kbo_topleft_box", "kbo_caster_box", "kbo_caster_box_up", "spotv2_box",
                 "kbo_box_bottomright", "bar_two_line"):
        assert Layout.load(ROOT / "data" / "scorebug_layouts" / f"{name}.json").panel
