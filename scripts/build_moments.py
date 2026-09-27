#!/usr/bin/env python
"""경기 전체 '방금 장면' 한 줄을 미리 만들어 스냅샷으로 굳힌다.

데모 중에는 모델을 부르지 않게 한다(카드 스냅샷과 같은 이유). 실제 모델이 있을 때만
의미가 있다 — mock이면 조립 문장이라 저장하지 않는다.

사용:
    python scripts/build_moments.py 20260823LGOB --levels 0 1 2
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.deps import get_moment_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game_id")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    service = get_moment_service()
    if service.llm.source != "llm":
        print("실제 모델이 없다(ANTHROPIC_API_KEY) — 조립 문장은 스냅샷으로 굳히지 않는다")
        return 1
    for level in args.levels:
        moments = service.all(args.game_id, level)
        sources = Counter(m.source for m in moments)
        print(f"L{level}: {len(moments)}줄 {dict(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
