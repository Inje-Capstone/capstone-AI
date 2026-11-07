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
    for name in ("sbs_sports_2026", "mbc_sports_plus", "tving_postseason"):
        layout = Layout.load(ROOT / "data" / "scorebug_layouts" / f"{name}.json")
        x0, y0, x1, y1 = layout.half
        assert 0 < x0 < x1 < 1 and 0 < y0 < y1 < 1
