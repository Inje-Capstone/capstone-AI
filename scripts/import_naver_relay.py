#!/usr/bin/env python
"""네이버 스포츠 문자중계 한 경기를 받아 fixture JSON으로 굳힌다.

서버는 네이버를 직접 부르지 않는다. 이 스크립트로 한 번 받아 두면 FixtureRelaySource가
파일만 읽으므로 데모 중 외부 호출이 0이다. 받은 직후 GameSim으로 재생해 최종 점수가
네이버 점수와 같은지 확인하고, 다르면 저장하지 않는다.

⚠️ 결과 파일(data/fixtures/naver_*.json)은 gitignore 대상이다. 재배포 허용 여부를
확인하기 전까지 팀 레포에 올리지 않는다.

사용:
    python scripts/import_naver_relay.py 20260920HHLG02026
    python scripts/import_naver_relay.py 20260920HHLG02026 --video-offset 95 --no-video
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.relay.fixture import _parse_fixture  # noqa: E402
from app.adapters.relay.naver import NaverRelayError, convert_game, fetch_game  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.domain.game_state import replay  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game_id", help="네이버 경기 ID (예: 20260920HHLG02026)")
    parser.add_argument(
        "--video-offset", type=int, default=0,
        help="영상에서 첫 투구가 나오는 시각(초). 카드 타임코드가 이만큼 밀린다.",
    )
    parser.add_argument("--no-video", action="store_true", help="영상 미확보로 표시 (S3 비활성)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        info, innings = fetch_game(args.game_id)
    except (NaverRelayError, OSError) as exc:
        print(f"조회 실패: {exc}", file=sys.stderr)
        return 1

    fixture = convert_game(
        info, innings,
        video_offset_sec=args.video_offset,
        has_video=False if args.no_video else None,
    )
    out_dir = args.out_dir or get_settings().fixture_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"naver_{args.game_id}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(fixture, ensure_ascii=False, indent=1), encoding="utf-8")

    feed = _parse_fixture(tmp)
    state = replay(feed.events, feed.meta.away_team, feed.meta.home_team)
    got = (state.away_score, state.home_score)
    want = (info.get("awayTeamScore"), info.get("homeTeamScore"))
    if got != want:
        tmp.unlink()
        print(f"재생 점수 {got} ≠ 네이버 점수 {want} — 변환 규칙을 확인하라", file=sys.stderr)
        return 2
    tmp.replace(out)
    print(f"저장: {out} (이벤트 {len(feed.events)}개, {want[0]}:{want[1]}, 재생 일치)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
