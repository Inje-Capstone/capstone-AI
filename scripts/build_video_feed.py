#!/usr/bin/env python
"""영상 분석 스냅샷 → 영상 판정 경기 데이터. 서비스 화면은 이 파일로 돈다.

`analyze_video.py`가 만든 `data/video/{id}.json`(점수판 판독·장면 단서·해설 키워드)을
video_judge로 판정해 `data/fixtures/video_{id}.json`으로 굳힌다. 시각은 영상 기준이라
타임라인 오프셋이 0이다. 문자중계는 쓰지 않는다 — 채점은 grade_video.py가 따로 한다.

경기 정보(팀·날짜·구장)는 옵션으로 준다. 같은 경기 정답지가 있으면 거기서 메타만 가져온다
(팀 이름 같은 메타일 뿐 플레이 정보는 쓰지 않는다).

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
from app.domain.video_judge import (  # noqa: E402
    MIN_CONFIDENCE,
    from_snapshot,
    judge,
    to_relay_events,
)


def truth_meta(game_id: str) -> Optional[dict[str, Any]]:
    try:
        meta = FixtureRelaySource(get_settings().truth_dir).load(game_id).meta
    except RelaySourceError:
        return None
    return {"away_team": meta.away_team, "home_team": meta.home_team, "date": meta.date,
            "stadium": meta.stadium}


def build(
    game_id: str, snap: dict[str, Any], meta: dict[str, Any],
    min_confidence: float = MIN_CONFIDENCE,
) -> dict[str, Any]:
    readings, cues = from_snapshot(snap)
    judgments = judge(readings, cues)
    events = to_relay_events(judgments, min_confidence)
    last = max((r.t for r in readings), default=0)
    return {
        "_note": ("영상 판정 결과(점수판 판독 × 장면·해설 단서). scripts/build_video_feed.py로 "
                  "재생성한다. 문자중계는 쓰지 않았다."),
        "source": "video",
        "analysis": {"provider": snap.get("provider"), "model": snap.get("model"),
                     "prompt_version": snap.get("prompt_version"),
                     "readings": len(readings), "cues": len(cues),
                     "judgments": dict(Counter(j.code for j in judgments)),
                     "min_confidence": min_confidence},
        "game": {
            "id": game_id, "date": meta.get("date", ""), "stadium": meta.get("stadium", ""),
            "away_team": meta["away_team"], "home_team": meta["home_team"],
            "has_video": True, "video_duration_sec": int(last), "relay_video_offset_sec": 0,
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
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    snap_path = args.snapshot or settings.video_dir / f"{args.game_id}.json"
    if not snap_path.exists():
        raise SystemExit(f"영상 분석 스냅샷이 없다: {snap_path} — analyze_video.py부터")
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    if not snap.get("scoreboard"):
        raise SystemExit("점수판 판독이 없다 — prompt v2로 다시 분석해야 한다")

    meta = truth_meta(args.game_id) or {}
    for key in ("away", "home"):
        if getattr(args, key):
            meta[f"{key}_team"] = getattr(args, key)
    meta.setdefault("date", args.date)
    meta.setdefault("stadium", args.stadium)
    if not meta.get("away_team") or not meta.get("home_team"):
        raise SystemExit("팀 이름을 모른다 — --away/--home을 준다")

    feed = build(args.game_id, snap, meta, args.min_confidence)
    out = (args.out_dir or settings.fixture_dir) / f"video_{args.game_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(feed, ensure_ascii=False, indent=1), encoding="utf-8")
    a = feed["analysis"]
    print(f"저장: {out}")
    print(f"판독 {a['readings']} · 단서 {a['cues']} · 이벤트 {len(feed['events'])}")
    print("판정:", ", ".join(f"{k} {v}" for k, v in sorted(a["judgments"].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
