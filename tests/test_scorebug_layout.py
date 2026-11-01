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
