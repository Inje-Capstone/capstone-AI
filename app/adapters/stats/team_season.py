"""2026 구단 성적·주전 기록(data/stats/team_2026.json)과 연봉(data/stats/salaries_2026.json) 읽기.

둘 다 공개 보도·나무위키에서 확인된 값만 적은 파일이다(KBO 공식 사이트는 승인 없는 자동 수집 금지).
파일이 없거나 깨져도 API가 죽지 않게 빈 값으로 돌려준다 — 팀 분석 패널만 '기록 없음'이 된다.
"""

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATS_DIR = Path(__file__).resolve().parents[3] / "data" / "stats"
TEAM_PATH = STATS_DIR / "team_2026.json"
SALARY_PATH = STATS_DIR / "salaries_2026.json"

EMPTY_TEAM: dict[str, Any] = {"season": None, "teams": [], "head_to_head": [], "batters": [],
                              "pitchers": []}
EMPTY_SALARY: dict[str, Any] = {"season": None, "league": {}, "teams": [], "players": [],
                                "foreign": []}


def _load(path: Path, empty: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))  # BOM 저장도 읽는다
    except (OSError, ValueError) as exc:
        log.warning("기록 파일을 읽지 못해 빈 값으로 (%s): %s", path.name, exc)
        return dict(empty)
    if not isinstance(raw, dict):
        return dict(empty)
    out = dict(empty)
    for key, default in empty.items():
        value = raw.get(key, default)
        out[key] = value if isinstance(value, type(default)) or default is None else default
    return out


def load_team_season(path: Path = TEAM_PATH) -> dict[str, Any]:
    return _load(path, EMPTY_TEAM)


def load_salaries(path: Path = SALARY_PATH) -> dict[str, Any]:
    return _load(path, EMPTY_SALARY)
