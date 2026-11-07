"""점수판의 주자(다이아몬드)·아웃(점)을 화소 색으로 읽는다 — 중계사마다 위치를 한 번 잰다.

리허설 실측(2026-10-28, 1080p KBO 중계 6장): VLM은 볼카운트·점수·이닝은 맞혔지만 아웃 점과
주자 다이아몬드는 지시문을 바꿔도 0~1/6만 맞혔다. 같은 6장(+0아웃 1장)을 정해 둔 위치의 색으로
읽자 7/7 — 채워진 칸은 채도 0.84~0.98, 빈 칸 0.00~0.01, 흐린 아웃 점 0.33~0.35.

위치는 화면 비율(0~1)로 저장한다(`data/scorebug_layouts/*.json`) — 해상도가 달라도 쓴다.
새 중계사는 `scripts/scorebug_layout.py`로 한 프레임에서 재고, 정답을 아는 몇 장으로 확인한다.

점수판이 화면 아래 띠인 중계(MBC 스포츠+, 포스트시즌 TVING)는 모서리 자르기(가로 45%)로는
주자·아웃 칸이 잘린다 — 위치 파일의 `crop`(x, y, 너비, 높이 비율)으로 띠를 통째 자른다.
MBC는 주자 칸을 흰색으로 채운다 — `base_fill: "bright"`면 밝고 색 없는 칸을 채운 것으로 본다.

초·말(▲/▼)도 VLM이 자주 뒤집어 읽었다(2026-11-07 MBC 하이라이트: 1회말을 1회초로) — 위치 파일에
`half`(화살표 칸 x0, y0, x1, y1 비율)가 있으면 칸 안 화살표 화소의 무게중심으로 가른다. ▼는 넓은
쪽이 위라 무게중심이 위로, ▲는 아래로 쏠린다.
▲·▼를 둘 다 그려 두고 켜진 쪽만 색칠하는 점수판(왼쪽 위 상자형)은 화살표가 720p에서 8줄뿐이라
모양으로는 5장 중 2장만 갈렸다 — `half_by: "position"`이면 색칠된 화소가 칸 위쪽이면 초,
아래쪽이면 말.

같은 점수판은 삼진이 나오면 주자·아웃 칸 자리에 빨간 'nK'(그 투수의 n번째 삼진)를 2~3초 덮는다.
그 판독은 주자·아웃이 엉터리다(K 글자가 3루 칸에 걸려 '3루 주자', 흰 바탕이라 '0아웃') — 처음 본
시험 경기(zRqpvJdTybA)에서 가짜 출루 판정이 나왔다. `mark`(칸 x0, y0, x1, y1)에 `mark_hue` 색
화소가 많으면 그 판독에 `mark: "strikeout"`을 달아 상태 판독에서 빼고 삼진 단서로 쓴다.
`base_hue`가 있으면 그 색상 범위(도)의 칸만 주자로 본다 — 노란 주자 칸과 빨간 K 글자를 가른다.
다른 중계(캐스터 이름 띠가 위에 붙은 상자형, YiiGl9UUV2E)는 검은 'nK'가 볼카운트·아웃 칸을 덮는다 —
`mark_fill: "dark"`면 표시 칸의 어두운 화소를 센다(평소 그 칸은 흰 바탕에 빨강·회색 점뿐).
표시가 나타나는 중이면 글자 화소가 모자라 놓친다(처음 본 경기 4장 중 3장) — 대신 두 중계 모두 표시가
아웃 점 자리를 흰 바탕으로 덮었다(채도 0.02 이하, 밝기 0.9 이상;
평소 꺼진 점은 회색 밝기 0.68~0.74).
`cover_val`이 있으면 아웃 점이 모두 그보다 밝고 색이 없을 때도 덮인 것(삼진 표시)으로 본다.
"""

import colorsys
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

FILLED_SAT = 0.5  # 이보다 채도가 높으면 채워진 칸(빨강·파랑 등). 흰 칸·흐린 점은 0.35 이하
FILLED_VAL = 0.3  # 너무 어두우면(검은 화면) 채워진 것으로 보지 않는다
BRIGHT_VAL = 0.7  # base_fill "bright": 흰 칸은 밝기 0.92~1.00, 빈 칸 0.11~0.14 (MBC 실측)
BRIGHT_SAT = 0.3

Point = tuple[float, float]
Rgb = tuple[float, float, float]


@dataclass(frozen=True)
class Layout:
    name: str
    bases: dict[int, Point]  # 1·2·3루 다이아몬드 중심 (화면 비율)
    outs: tuple[Point, ...]  # 아웃 점 중심들 (화면 비율)
    crop: Optional[tuple[float, float, float, float]] = None  # 점수판 영역(x, y, w, h 비율)
    scale: float = 3.0  # VLM에 보낼 때 crop을 키우는 배율 — 넓은 띠는 1.5면 된다
    base_fill: str = "color"  # 주자 칸을 채우는 방식: color(색) | bright(흰색)
    half: Optional[tuple[float, float, float, float]] = None  # ▲/▼ 칸(x0, y0, x1, y1 비율)
    half_fill: str = "color"  # 화살표 색: color | bright
    half_by: str = "shape"  # shape(화살표 하나의 모양) | position(▲·▼ 중 켜진 쪽 위치)
    base_hue: Optional[tuple[float, float]] = None  # 주자 칸 색상 범위(도, lo>hi면 0도를 넘는다)
    mark: Optional[tuple[float, float, float, float]] = None  # 삼진 표시가 덮는 칸(비율)
    mark_hue: tuple[float, float] = (340.0, 20.0)  # 삼진 표시 글자 색상 범위(도) — 기본 빨강
    mark_fill: str = "color"  # 삼진 표시 글자: color(mark_hue 색) | dark(검은 글자)
    cover_val: Optional[float] = None  # 아웃 점이 모두 이보다 밝은 무채색이면 표시가 덮은 것

    @classmethod
    def load(cls, path: Path) -> "Layout":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("crop", "half", "mark"):
            if raw.get(key) is not None and len(raw[key]) != 4:
                raise ValueError(f"{path}: {key}는 숫자 4개(x0, y0, x1, y1 또는 x, y, w, h)다")
        if raw.get("half_by", "shape") not in ("shape", "position"):
            raise ValueError(f"{path}: half_by는 shape 또는 position이다")
        if raw.get("mark_fill", "color") not in ("color", "dark"):
            raise ValueError(f"{path}: mark_fill은 color 또는 dark다")
        crop = raw.get("crop")
        half = raw.get("half")
        return cls(
            name=str(raw["name"]),
            bases={int(k): (float(v[0]), float(v[1])) for k, v in raw["bases"].items()},
            outs=tuple((float(x), float(y)) for x, y in raw["outs"]),
            crop=tuple(float(v) for v in crop) if crop else None,  # type: ignore[arg-type]
            scale=float(raw.get("scale", 3.0)),
            base_fill=str(raw.get("base_fill", "color")),
            half=tuple(float(v) for v in half) if half else None,  # type: ignore[arg-type]
            half_fill=str(raw.get("half_fill", "color")),
            half_by=str(raw.get("half_by", "shape")),
            base_hue=_pair(raw.get("base_hue")),
            mark=tuple(float(v) for v in raw["mark"]) if raw.get("mark") else None,  # type: ignore[arg-type]
            mark_hue=_pair(raw.get("mark_hue")) or (340.0, 20.0),
            mark_fill=str(raw.get("mark_fill", "color")),
            cover_val=float(raw["cover_val"]) if raw.get("cover_val") is not None else None,
        )


def _pair(raw) -> Optional[tuple[float, float]]:
    return (float(raw[0]), float(raw[1])) if raw else None


def hue_in(rgb: Rgb, hues: Optional[tuple[float, float]]) -> bool:
    """색상(도)이 범위 안인가. 범위가 없으면 늘 참. lo > hi면 0도를 넘는 범위(빨강 340~20)."""
    if hues is None:
        return True
    h = colorsys.rgb_to_hsv(*rgb)[0] * 360
    lo, hi = hues[0] % 360, hues[1] % 360  # [-20, 20]도 [340, 20]으로
    return lo <= h <= hi if lo <= hi else (h >= lo or h <= hi)


def is_filled(rgb: Rgb, mode: str = "color") -> bool:
    _, s, v = colorsys.rgb_to_hsv(*rgb)
    if mode == "bright":
        return v > BRIGHT_VAL and s < BRIGHT_SAT
    return s > FILLED_SAT and v > FILLED_VAL


def decode(layout: Layout, sample: Callable[[Point], Rgb]) -> tuple[list[int], int]:
    """(주자 베이스 목록, 아웃 수). `sample(점)`이 그 위치 둘레 평균 색(0~1)을 준다."""
    bases = sorted(n for n, p in layout.bases.items()
                   if is_filled(sample(p), layout.base_fill)
                   and (layout.base_fill == "bright" or hue_in(sample(p), layout.base_hue)))
    outs = sum(1 for p in layout.outs if is_filled(sample(p)))
    return bases, outs


HALF_MIN_PIXELS = 8  # 화살표로 볼 최소 화소 수(1280×720 프레임)
HALF_TILT = 0.08  # 무게중심이 세로 범위 가운데에서 이만큼(범위 비율) 넘게 쏠려야 가른다


def decode_half(layout: Layout, pixels: bytes, width: int, height: int) -> Optional[str]:
    """▲(top) / ▼(bot) / 모름(None) — 화살표 칸 안 채워진 화소의 세로 무게중심으로."""
    if layout.half is None:
        return None
    x0, y0, x1, y1 = (round(v * n) for v, n in zip(layout.half, (width, height, width, height)))
    ys = []
    for y in range(max(0, y0), min(height, y1 + 1)):
        for x in range(max(0, x0), min(width, x1 + 1)):
            i = (y * width + x) * 3
            rgb = (pixels[i] / 255, pixels[i + 1] / 255, pixels[i + 2] / 255)
            if is_filled(rgb, layout.half_fill):
                ys.append(y)
    if len(ys) < HALF_MIN_PIXELS:
        return None
    center = sum(ys) / len(ys)
    if layout.half_by == "position":  # 켜진 화살표가 칸의 위쪽 반이면 ▲(초)
        mid = (y0 + y1) / 2
        if abs(center - mid) < HALF_TILT * (y1 - y0):  # 둘 다 켜졌거나 가운데 — 모른다
            return None
        return "top" if center < mid else "bot"
    top, bottom = min(ys), max(ys)
    mid = (top + bottom) / 2
    if center < mid - HALF_TILT * (bottom - top):
        return "bot"
    if center > mid + HALF_TILT * (bottom - top):
        return "top"
    return None


MARK_MIN_PIXELS = 40  # 삼진 표시로 볼 최소 화소 수(1280×720) — 'nK' 글자는 수백 화소
DARK_VAL = 0.3  # mark_fill "dark": 이보다 어두우면 검은 글자
COVER_SAT = 0.1  # cover_val: 아웃 점 자리가 이보다 색이 없어야 흰 바탕


def decode_mark(layout: Layout, pixels: bytes, width: int, height: int) -> Optional[str]:
    """점수판이 주자·아웃 칸을 삼진 표시(K)로 덮었으면 "strikeout", 아니면 None."""
    if layout.mark is None:
        return None
    if layout.cover_val is not None and layout.outs:
        sample = rgb_sampler(pixels, width, height)
        hsv = [colorsys.rgb_to_hsv(*sample(p)) for p in layout.outs]
        if all(s < COVER_SAT and v >= layout.cover_val for _, s, v in hsv):
            return "strikeout"
    x0, y0, x1, y1 = (round(v * n) for v, n in zip(layout.mark, (width, height, width, height)))
    found = 0
    for y in range(max(0, y0), min(height, y1 + 1)):
        for x in range(max(0, x0), min(width, x1 + 1)):
            i = (y * width + x) * 3
            rgb = (pixels[i] / 255, pixels[i + 1] / 255, pixels[i + 2] / 255)
            if (max(rgb) < DARK_VAL if layout.mark_fill == "dark"
                    else is_filled(rgb) and hue_in(rgb, layout.mark_hue)):
                found += 1
                if found >= MARK_MIN_PIXELS:
                    return "strikeout"
    return None


def rgb_sampler(pixels: bytes, width: int, height: int) -> Callable[[Point], Rgb]:
    """rgb24 원본 프레임 → 비율 좌표 둘레(1080p에서 5×5) 평균 색."""
    r = max(1, round(height / 540))

    def sample(p: Point) -> Rgb:
        cx, cy = round(p[0] * width), round(p[1] * height)
        acc, n = [0, 0, 0], 0
        for y in range(max(0, cy - r), min(height, cy + r + 1)):
            for x in range(max(0, cx - r), min(width, cx + r + 1)):
                i = (y * width + x) * 3
                acc[0] += pixels[i]
                acc[1] += pixels[i + 1]
                acc[2] += pixels[i + 2]
                n += 1
        return (acc[0] / n / 255, acc[1] / n / 255, acc[2] / n / 255) if n else (0.0, 0.0, 0.0)

    return sample


def check(layout: Layout, frames: Sequence[tuple[bytes, int, int]]) -> list[tuple[list[int], int]]:
    """여러 프레임을 한 번에 읽는다 (보정 확인용)."""
    return [decode(layout, rgb_sampler(px, w, h)) for px, w, h in frames]
