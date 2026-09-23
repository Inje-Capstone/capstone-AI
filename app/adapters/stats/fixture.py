"""파일 기반 기록 소스. `data/stats/stats_seed.json`."""

import json
from pathlib import Path
from typing import Any, Optional

from app.adapters.stats.base import BatterFacts, StatsSourceError

DEFAULT_STATS_PATH = Path(__file__).resolve().parents[3] / "data" / "stats" / "stats_seed.json"


class FixtureStatsSource:
    def __init__(self, path: Path = DEFAULT_STATS_PATH) -> None:
        self.path = Path(path)
        self._data: Optional[dict[str, Any]] = None

    def batter_facts(
        self, batter: str, pitcher: Optional[str], team: str
    ) -> BatterFacts:
        data = self._load()
        entry = (data.get("batters") or {}).get(batter)
        if not entry:
            raise StatsSourceError(f"기록이 없는 타자: {batter}")

        vs = None
        if pitcher:
            vs_map = entry.get("vs") or {}
            matched = vs_map.get(pitcher)
            if matched:
                vs = f"vs {pitcher} {matched}"

        return BatterFacts(
            batter=batter,
            avg=entry.get("avg"),
            recent=entry.get("recent"),
            vs_pitcher=vs,
            team_form=(data.get("team_form") or {}).get(team),
        )

    def _load(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StatsSourceError(f"기록 파일을 읽을 수 없다: {self.path} ({exc})") from exc
        return self._data
