"""점수판 전용 판독 — 읽을 순간 고르기(로컬)와 응답 정리. API·영상 없이."""

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
