"""2026 선수 연봉 기록(data/stats/salaries_2026.json) — 출처·중복·단위·집계 수 검사.

공개 보도로 확인된 것만 적는 파일이다.
손으로 줄을 더할 때 출처를 빼먹거나 단위를 틀리면 여기서 걸린다.
"""

import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "data" / "stats" / "salaries_2026.json"
TEAMS = {"한화", "삼성", "LG", "KIA", "SSG", "NC", "KT", "두산", "롯데", "키움"}
KINDS = {None, "재계약", "FA", "다년", "육성"}
FOREIGN_KINDS = {"외국인", "아시아쿼터"}


@pytest.fixture(scope="module")
def data():
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_every_row_cites_known_sources(data):
    keys = set(data["sources"])
    rows = data["players"] + data["foreign"] + data["teams"] + [data["league"]]
    for row in rows:
        assert row["sources"], row
        assert set(row["sources"]) <= keys, row
    for src in data["sources"].values():
        assert src["url"].startswith("https://")
        assert src["reliability"] in {"official", "press", "blog", "wiki"}
    for row in data["players"]:
        if "NAMU" in row["sources"]:
            assert (row.get("page_url") or "").startswith("https://namu.wiki/"), row


def test_no_duplicate_players_and_teams_are_known(data):
    seen = set()
    for row in data["players"] + data["foreign"]:
        key = (row["team"], row["name"])
        assert key not in seen, key
        seen.add(key)
        assert row["team"] in TEAMS, row
    assert {t["team"] for t in data["teams"]} == TEAMS


def test_amounts_are_positive_integers_in_units(data):
    for row in data["players"]:
        assert row["kind"] in KINDS, row
        assert isinstance(row["salary_10k"], int) and row["salary_10k"] >= 3000, row
        prev = row["prev_salary_10k"]
        assert prev is None or (isinstance(prev, int) and prev > 0), row
    for row in data["foreign"]:
        assert row["kind"] in FOREIGN_KINDS, row
        assert isinstance(row["contract_total_usd"], int) and row["contract_total_usd"] > 0
        kbo = row["kbo_salary_usd"]
        assert kbo is None or (isinstance(kbo, int) and kbo <= row["contract_total_usd"])


def test_missing_players_are_not_also_recorded(data):
    recorded = {(p["team"], p["name"]) for p in data["players"]}
    for row in data.get("missing", []):
        assert row["team"] in TEAMS
        assert (row["team"], row["name"]) not in recorded, row
    for row in data.get("conflicts", []):
        assert (row["team"], row["name"]) in recorded, row


def test_coverage_counts_match_rows(data):
    cov = data["coverage"]
    assert cov["recorded_domestic"] == len(data["players"])
    assert cov["recorded_foreign"] == len(data["foreign"])
    # 621명은 1월 31일 등록 마감 기준 — 육성선수·시즌 중 영입·이적은 그 뒤라 기록이 더 많을 수 있다
    assert cov["missing_listed"] == len(data.get("missing", []))
    assert data["league"]["players"] == cov["salary_pool_players"]


def test_league_average_matches_total(data):
    league = data["league"]
    assert abs(league["total_salary_10k"] / league["players"] - league["avg_salary_10k"]) < 1
