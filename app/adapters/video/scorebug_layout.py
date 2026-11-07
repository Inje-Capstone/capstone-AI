"""점수판의 주자(다이아몬드)·아웃(점)을 화소 색으로 읽는다 — 중계사마다 위치를 한 번 잰다.

리허설 실측(2026-10-28, 1080p KBO 중계 6장): VLM은 볼카운트·점수·이닝은 맞혔지만 아웃 점과
주자 다이아몬드는 지시문을 바꿔도 0~1/6만 맞혔다. 같은 6장(+0아웃 1장)을 정해 둔 위치의 색으로
읽자 7/7 — 채워진 칸은 채도 0.84~0.98, 빈 칸 0.00~0.01, 흐린 아웃 점 0.33~0.35.

위치는 화면 비율(0~1)로 저장한다(`data/scorebug_layouts/*.json`) — 해상도가 달라도 쓴다.
새 중계사는 `scripts/scorebug_layout.py`로 한 프레임에서 재고, 정답을 아는 몇 장으로 확인한다.

점수판이 화면 아래 띠인 중계(MBC 스포츠+, 포스트시즌 TVING)는 모서리 자르기(가로 45%)로는
주자·아웃 칸이 잘린다 — 위치 파일의 `crop`(x, y, 너비, 높이 비율)으로 띠를 통째 자른다.
MBC는 주자 칸을 흰색으로 채운다 — `base_fill: "bright"`면 밝고 색 없는 칸을 채운 것으로 본다.
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

    @classmethod
    def load(cls, path: Path) -> "Layout":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        crop = raw.get("crop")
        return cls(
            name=str(raw["name"]),
            bases={int(k): (float(v[0]), float(v[1])) for k, v in raw["bases"].items()},
            outs=tuple((float(x), float(y)) for x, y in raw["outs"]),
            crop=tuple(float(v) for v in crop) if crop else None,  # type: ignore[arg-type]
            scale=float(raw.get("scale", 3.0)),
            base_fill=str(raw.get("base_fill", "color")),
        )


def is_filled(rgb: Rgb, mode: str = "color") -> bool:
    _, s, v = colorsys.rgb_to_hsv(*rgb)
    if mode == "bright":
        return v > BRIGHT_VAL and s < BRIGHT_SAT
    return s > FILLED_SAT and v > FILLED_VAL


def decode(layout: Layout, sample: Callable[[Point], Rgb]) -> tuple[list[int], int]:
    """(주자 베이스 목록, 아웃 수). `sample(점)`이 그 위치 둘레 평균 색(0~1)을 준다."""
    bases = sorted(n for n, p in layout.bases.items()
                   if is_filled(sample(p), layout.base_fill))
    outs = sum(1 for p in layout.outs if is_filled(sample(p)))
    return bases, outs


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
