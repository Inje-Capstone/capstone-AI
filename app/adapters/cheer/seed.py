"""응원가 안내(data/cheer_songs.json) 읽기.

곡명과 구단 공식 채널 링크만 담는다 — 가사·음원은 저작권 대상이라 넣지 않는다.
자동 수집하지 않고 팀이 직접 확인해 적는다(KBO 공식 사이트는 승인 없는 자동 수집을 금지한다).
읽을 때 모양을 검사해 잘못 적힌 줄은 건너뛴다 — 한 줄 실수로 API가 죽으면 안 된다.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

CHEER_PATH = Path(__file__).resolve().parents[3] / "data" / "cheer_songs.json"

TEAMS = ("한화", "삼성", "LG", "KIA", "SSG", "NC", "KT", "두산", "롯데", "키움")


def valid_song(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    player = row.get("player")
    return (
        row.get("team") in TEAMS
        and (player is None or (isinstance(player, str) and player.strip() != ""))
        and isinstance(row.get("title"), str) and row["title"].strip() != ""
        and isinstance(row.get("url"), str) and row["url"].startswith("https://")
    )


def load_cheer_songs(path: Path = CHEER_PATH) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("songs", []) if isinstance(raw, dict) else []
    except (OSError, ValueError) as exc:
        log.warning("응원가 파일을 읽지 못해 빈 목록으로: %s", exc)
        return []
    if not isinstance(rows, list):
        rows = []
    songs = []
    for n, row in enumerate(rows):
        if valid_song(row):
            player = row.get("player")
            songs.append(
                {"team": row["team"], "player": player.strip() if player else None,
                 "title": row["title"].strip(), "url": row["url"].strip()}
            )
        else:
            log.warning("응원가 %d번째 줄 모양이 틀려 건너뜀: %s", n + 1, str(row)[:120])
    return songs


def song_for(
    songs: list[dict[str, Any]], team: Optional[str], batter: Optional[str]
) -> Optional[dict[str, Any]]:
    """그 타자의 응원가, 없으면 그 팀 응원가(첫 줄)."""
    if not team:
        return None
    if batter:
        own = next((s for s in songs if s["team"] == team and s["player"] == batter), None)
        if own:
            return own
    return next((s for s in songs if s["team"] == team and s["player"] is None), None)
