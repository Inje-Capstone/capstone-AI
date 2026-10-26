#!/usr/bin/env python
"""영상 → 투수 자세 → 세트마다 동작 분석과 보크 의심 (app/domain/pitch_motion.py).

선택 설치: pip install -e ".[pose]". CPU로 돈다(10fps·lightweight 기준 1분 영상에 수 분).
자세 좌표는 영상 옆 `<영상>.pose.json`에 캐시해 규칙만 바꿔 다시 볼 때는 빠르다.

    python scripts/balk_check.py clip.mp4
    python scripts/balk_check.py clips/*.mp4 --fps 10 --mode lightweight
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.video.pose import (  # noqa: E402
    PitcherPoseExtractor,
    frames_from_json,
    frames_to_json,
)
from app.domain.pitch_motion import analyze_motions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--mode", default="lightweight", help="lightweight | balanced | performance"
    )
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로")
    args = parser.parse_args()
    extractor = None
    report = {}
    for video in args.videos:
        cache = video.with_suffix(".pose.json")
        if cache.exists():
            frames = frames_from_json(json.loads(cache.read_text(encoding="utf-8")))
        else:
            extractor = extractor or PitcherPoseExtractor(fps=args.fps, mode=args.mode)
            frames = extractor.extract(video)
            cache.write_text(json.dumps(frames_to_json(frames)), encoding="utf-8")
        motions = analyze_motions(frames)
        report[video.name] = [m.__dict__ for m in motions]
        if not args.json:
            print(f"== {video.name}: 투수 프레임 {len(frames)} · 세트 {len(motions)}")
            for m in motions:
                flag = f"  ⚠ {m.suspect}" if m.suspect else ""
                print(f"  {m.t_set:6.1f}s {m.outcome:8s}{flag}  {m.evidence[0]}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
