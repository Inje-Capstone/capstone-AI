#!/usr/bin/env python
"""중계사 점수판 위치(주자 다이아몬드·아웃 점) 확인 — 정답을 아는 프레임 몇 장에 대 본다. API 없음.

새 중계사(시연 녹화본)를 쓰기 전에 한 번:
1. 점수판이 보이는 프레임을 크게 잘라 본다:
   ffmpeg -ss 30 -i game.mp4 -frames:v 1 -vf "crop=iw*0.45:ih*0.32:0:0" bug.png
2. 다이아몬드 셋(1·2·3루)·아웃 점들의 중심을 화면 비율(0~1)로 data/scorebug_layouts/<이름>.json에
   적는다 (sbs_sports.json 참고)
3. 주자·아웃을 눈으로 확인한 시각으로 맞는지 본다 — 채워진 칸 채도는 0.8 안팎, 빈 칸은 0.35 이하

사용:
    python scripts/scorebug_layout.py game.mp4 data/scorebug_layouts/sbs_sports.json 8 36 84
    python scripts/scorebug_layout.py game.mp4 layout.json 8:12:2 84:13:2   # 시각:주자:아웃 — 채점
"""

import argparse
import colorsys
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.video.scorebug_layout import Layout, decode, rgb_sampler  # noqa: E402
from app.adapters.video.scorebug_reader import RGB_H, RGB_W, ScorebugReader  # noqa: E402


def parse_point(arg: str) -> tuple[float, Optional[list[int]], Optional[int]]:
    """'84' → (84, None, None), '84:13:2' → (84, [1, 3], 2), '84:-:0' → 주자 없음."""
    parts = arg.split(":")
    if len(parts) == 1:
        return float(parts[0]), None, None
    bases = [] if parts[1] in ("-", "") else sorted(int(c) for c in parts[1])
    return float(parts[0]), bases, int(parts[2])


def saturation(sample, p: tuple[float, float]) -> str:
    return f"{colorsys.rgb_to_hsv(*sample(p))[1]:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("layout", type=Path)
    parser.add_argument("at", nargs="+", help="시각(초) 또는 시각:주자:아웃")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    layout = Layout.load(args.layout)
    reader = ScorebugReader("", "http://unused", "", ffmpeg=args.ffmpeg)
    graded = right = 0
    for arg in args.at:
        t, want_bases, want_outs = parse_point(arg)
        sample = rgb_sampler(reader._rgb(args.video, t), RGB_W, RGB_H)
        bases, outs = decode(layout, sample)
        sat = " ".join(f"{n}루 {saturation(sample, p)}" for n, p in sorted(layout.bases.items()))
        sat += " · 아웃점 " + " ".join(saturation(sample, p) for p in layout.outs)
        mark = ""
        if want_bases is not None:
            graded += 1
            ok = (bases, outs) == (want_bases, want_outs)
            right += ok
            mark = " 맞음" if ok else f" 틀림(정답 주자 {want_bases} 아웃 {want_outs})"
        print(f"{t:7.1f}s 주자 {bases or '-'} 아웃 {outs}{mark} | 채도 {sat}")
    if graded:
        print(f"{layout.name}: {right}/{graded}")
    return 0 if right == graded else 1


if __name__ == "__main__":
    raise SystemExit(main())
