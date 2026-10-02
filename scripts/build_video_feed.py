#!/usr/bin/env python
"""영상 분석 스냅샷 + 경기 데이터 → 서비스용 경기 데이터. 서비스 화면은 이 파일로 돈다.

- **영상**(`data/video/{id}.json`): 플레이 판정 — video_judge (보크·도루·주루·아웃·득점)
- **데이터**(`data/relay_truth/naver_{id}.json`, 있으면): 선수 이름·라인업·구종·구속·기록·
  팀 맥락 — enrich. 데이터의 플레이 결과는 쓰지 않는다(그건 grade_video.py 채점용)

결과는 `data/fixtures/video_{id}.json`. 시각은 영상 기준(오프셋 0). 데이터를 붙이려면
영상↔데이터 오프셋이 필요하다(analyze_video.py가 추정해 스냅샷에 저장, 또는 `--offset`).

사용:
    python scripts/build_video_feed.py 20260920HHLG02026
    python scripts/build_video_feed.py G1 --away 한화 --home LG --date 2026-09-20 --stadium 잠실
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.relay.base import RelaySourceError  # noqa: E402
from app.adapters.relay.fixture import FixtureRelaySource  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.domain.enrich import enrich  # noqa: E402
from app.domain.models import GameFeed  # noqa: E402
from app.domain.scorebug import stabilize  # noqa: E402
from app.domain.video_judge import (  # noqa: E402
    MIN_CONFIDENCE,
    from_snapshot,
    judge,
    to_relay_events,
)


def load_data(game_id: str) -> Optional[GameFeed]:
    try:
        return FixtureRelaySource(get_settings().truth_dir).load(game_id)
    except RelaySourceError:
        return None


def data_meta(data: Optional[GameFeed]) -> dict[str, Any]:
    if data is None:
        return {}
    m = data.meta
    return {"away_team": m.away_team, "home_team": m.home_team, "date": m.date,
            "stadium": m.stadium, "context": m.context}


def build(
    game_id: str, snap: dict[str, Any], meta: dict[str, Any],
    min_confidence: float = MIN_CONFIDENCE,
    data: Optional[GameFeed] = None, offset: Optional[float] = None,
) -> dict[str, Any]:
    readings, cues = from_snapshot(snap)
    judgments = judge(readings, cues)
    events = to_relay_events(judgments, min_confidence)
    enriched = data is not None and offset is not None
    if enriched:
        events = enrich(events, data.events, offset, stabilize(readings))
    last = max((r.t for r in readings), default=0)
    return {
        "_note": ("플레이 판정은 영상(점수판 판독 × 장면·해설 단서)" +
                  (", 선수·구종·기록은 경기 데이터(네이버)" if enriched else "") +
                  ". scripts/build_video_feed.py로 재생성한다."),
        "source": "video",
        "analysis": {"provider": snap.get("provider"), "model": snap.get("model"),
                     "prompt_version": snap.get("prompt_version"),
                     "readings": len(readings), "cues": len(cues),
                     "judgments": dict(Counter(j.code for j in judgments)),
                     "min_confidence": min_confidence,
                     "data": {"source": "naver", "offset": offset} if enriched else None},
        "game": {
            "id": game_id, "date": meta.get("date", ""), "stadium": meta.get("stadium", ""),
            "away_team": meta["away_team"], "home_team": meta["home_team"],
            "has_video": True, "video_duration_sec": int(last), "relay_video_offset_sec": 0,
            "context": meta.get("context") or {},
        },
        "events": [e.model_dump() for e in events],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game_id")
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--away")
    parser.add_argument("--home")
    parser.add_argument("--date", default="")
    parser.add_argument("--stadium", default="")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--offset", type=float, default=None,
                        help="영상 t − 데이터 t (초). 생략하면 스냅샷의 추정값")
    parser.add_argument("--no-data", action="store_true", help="경기 데이터를 붙이지 않는다")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    snap_path = args.snapshot or settings.video_dir / f"{args.game_id}.json"
    if not snap_path.exists():
        raise SystemExit(f"영상 분석 스냅샷이 없다: {snap_path} — analyze_video.py부터")
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    if not snap.get("scoreboard"):
        raise SystemExit("점수판 판독이 없다 — prompt v2로 다시 분석해야 한다")

    data = None if args.no_data else load_data(args.game_id)
    meta = data_meta(data)
    for key in ("away", "home"):
        if getattr(args, key):
            meta[f"{key}_team"] = getattr(args, key)
    meta.setdefault("date", args.date)
    meta.setdefault("stadium", args.stadium)
    if not meta.get("away_team") or not meta.get("home_team"):
        raise SystemExit("팀 이름을 모른다 — --away/--home을 준다")

    offset = args.offset
    saved = snap.get("offset") or {}
    if offset is None and saved.get("confident"):
        offset = float(saved["offset"])
    if data is not None and offset is None:
        print("데이터는 있지만 영상↔데이터 오프셋을 모른다 — 선수·구종 없이 만든다 (--offset)")
    feed = build(args.game_id, snap, meta, args.min_confidence, data, offset)
    out = (args.out_dir or settings.fixture_dir) / f"video_{args.game_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(feed, ensure_ascii=False, indent=1), encoding="utf-8")
    a = feed["analysis"]
    print(f"저장: {out}")
    print(f"판독 {a['readings']} · 단서 {a['cues']} · 이벤트 {len(feed['events'])}")
    print("판정:", ", ".join(f"{k} {v}" for k, v in sorted(a["judgments"].items())))
    print("데이터:", f"네이버 (오프셋 {offset:+.1f}s)" if a["data"] else "없음 — 영상 판정만")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
