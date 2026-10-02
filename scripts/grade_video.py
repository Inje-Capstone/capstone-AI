#!/usr/bin/env python
"""영상 판정 채점 — 같은 경기 네이버 문자중계(정답지)와 비교해 규칙별 정확도를 낸다.

서비스 화면은 영상 판정만 쓴다. 이 스크립트는 "영상만으로 몇 개를 맞혔나"를 재는 검증용이다.
시각은 영상 분석 스냅샷의 오프셋 추정(analyze_video.py) 또는 `--offset`으로 맞춘다.

사용:
    python scripts/grade_video.py 20260920HHLG02026
    python scripts/grade_video.py 20260920HHLG02026 --offset 312 --tolerance 30 --report g.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.relay.fixture import FixtureRelaySource  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.domain.detectors import detect  # noqa: E402
from app.domain.grading import UNSUPPORTED_RULES, grade, summary  # noqa: E402
from app.domain.video_sync import Anchor, estimate_offset  # noqa: E402


def load_video_feed(game_id: str, fixture_dir: Path):
    path = fixture_dir / f"video_{game_id}.json"
    if not path.exists():
        raise SystemExit(f"영상 판정 데이터가 없다: {path} — build_video_feed.py부터")
    from app.adapters.relay.fixture import _parse_fixture

    return _parse_fixture(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game_id")
    parser.add_argument("--offset", type=float, default=None, help="영상 t − 중계 t (초)")
    parser.add_argument("--tolerance", type=float, default=30.0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    truth = FixtureRelaySource(settings.truth_dir).load(args.game_id)
    video = load_video_feed(args.game_id, settings.fixture_dir)

    offset = args.offset
    if offset is None:
        snap_path = settings.video_dir / f"{args.game_id}.json"
        snap = json.loads(snap_path.read_text(encoding="utf-8")) if snap_path.exists() else {}
        saved = snap.get("offset") or {}
        est = estimate_offset(truth.events, [
            Anchor(float(e["t_start"]), e["event_type"]) for e in snap.get("events", [])])
        if (est is None or not est.confident) and saved.get("confident"):
            offset = float(saved["offset"])
        elif est is None or not est.confident:
            raise SystemExit("오프셋을 추정하지 못했다 — --offset으로 준다")
        else:
            offset = est.offset
            print(f"오프셋 추정 {offset:+.1f}s (짝 {est.matched}/{est.anchors})")

    # 영상이 덮은 구간: 점수판 판독이 처음·마지막으로 보인 시각
    times = [e.t for e in video.events]
    window = (min(times) - args.tolerance, max(times) + args.tolerance) if times else None
    if window:
        print(f"채점 구간(영상 기준): {window[0]:.0f}s ~ {window[1]:.0f}s")
    scores = grade(detect(truth), detect(video), offset=offset, tolerance=args.tolerance,
                   window=window)
    print(f"{'규칙':<22}{'정답':>5}{'판정':>5}{'일치':>5}{'재현율':>8}{'정밀도':>8}")
    for s in scores:
        rec = f"{s.recall:.0%}" if s.expected else "-"
        pre = f"{s.precision:.0%}" if s.found else "-"
        print(f"{s.rule_id:<22}{s.expected:>5}{s.found:>5}{s.matched:>5}{rec:>8}{pre:>8}")
    total = summary(scores)
    print(f"전체: 정답 {total['expected']} · 판정 {total['found']} · 일치 {total['matched']} · "
          f"재현율 {total['recall']:.0%} · 정밀도 {total['precision']:.0%}")
    print(f"채점 제외(영상 판정 범위 밖): {', '.join(UNSUPPORTED_RULES)}")
    if args.report:
        args.report.write_text(json.dumps({
            "game_id": args.game_id, "offset": offset, "tolerance": args.tolerance,
            "total": total, "rules": [s.__dict__ for s in scores],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
