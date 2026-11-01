#!/usr/bin/env python
"""영상 → 투구 화면 → 타자 스윙 (app/adapters/video/pitch_view.py · app/domain/swing.py).

선택 설치: pip install -e ".[pose]". CPU로 돈다. 결과는 영상 옆 `<영상>.swings.json`에 캐시한다.

    python scripts/swing_check.py clip.mp4
    python scripts/swing_check.py clips/*.mp4 --again
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.video.pitch_view import SwingExtractor  # noqa: E402


def cache_path(video: Path) -> Path:
    return video.with_suffix(".swings.json")


def load_cached(video: Path) -> list[dict[str, Any]]:
    path = cache_path(video)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--mode", default="balanced", help="lightweight | balanced | performance")
    parser.add_argument("--again", action="store_true", help="캐시가 있어도 다시 잰다")
    args = parser.parse_args()
    extractor = None
    for video in args.videos:
        path = cache_path(video)
        if path.exists() and not args.again:
            continue
        extractor = extractor or SwingExtractor(args.mode)
        start = time.perf_counter()
        rows = extractor.extract(video)
        path.write_text(json.dumps(rows), encoding="utf-8")
        print(f"{video.name}: 투구 화면 {len(rows)} · 스윙 {sum(r['swing'] for r in rows)} · "
              f"{time.perf_counter() - start:.0f}초")


if __name__ == "__main__":
    main()
