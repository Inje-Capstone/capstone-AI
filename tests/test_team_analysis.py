"""팀 분석 — 기록 파일(data/stats/team_2026.json) 검사 + 연봉 대비 활약·맞대결 로직 + API."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.stats.team_season import load_team_season
from app.api.deps import reset_services
from app.api.schemas import TeamAnalysisOut
from app.config import get_settings
from app.domain.team_analysis import (
    FAIR,
    OVER,
    TEAMS,
    UNDER,
    head_to_head,
    salary_values,
    team_key,
    team_report,
)
from app.main import app

PATH = Path(__file__).resolve().parents[1] / "data" / "stats" / "team_2026.json"
RATE = re.compile(r"^\d\.\d{3}$")
TWO = re.compile(r"^\d{1,2}\.\d{2}$")
IP = re.compile(r"^\d{1,3}(\.[012])?$")


@pytest.fixture(scope="module")
def data():
    return json.loads(PATH.read_text(encoding="utf-8"))


# ── 기록 파일 ─────────────────────────────────────────────


def test_every_team_once_with_regular_season_rank(data):
    assert sorted(t["team"] for t in data["teams"]) == sorted(TEAMS)
    assert sorted(t["rank"] for t in data["teams"]) == list(range(1, 11))
    for t in data["teams"]:
        assert t["season_url"].startswith("https://namu.wiki/"), t
        assert t["w"] + t["l"] + t["d"] == 144, t
        assert t["win_pct"] == round(t["w"] / (t["w"] + t["l"]), 3)
    pct = sorted((t["win_pct"] for t in data["teams"]), reverse=True)
    assert [t["win_pct"] for t in sorted(data["teams"], key=lambda t: t["rank"])] == pct


def test_head_to_head_covers_all_pairs_and_sums_to_record(data):
    pairs = {frozenset((h["team"], h["opponent"])) for h in data["head_to_head"]}
    assert len(pairs) == len(data["head_to_head"]) == 45
    assert data["conflicts"] == []
    for t in data["teams"]:
        won = lost = drawn = 0
        for h in data["head_to_head"]:
            if h["team"] == t["team"]:
                won, lost, drawn = won + h["w"], lost + h["l"], drawn + h["d"]
            elif h["opponent"] == t["team"]:
                won, lost, drawn = won + h["l"], lost + h["w"], drawn + h["d"]
        assert (won, lost, drawn) == (t["w"], t["l"], t["d"]), t["team"]
    for h in data["head_to_head"]:
        assert h["w"] + h["l"] + h["d"] == 16, h  # 구단당 16차전


def test_player_rows_have_sources_and_well_formed_numbers(data):
    for b in data["batters"]:
        assert b["team"] in TEAMS and b["name"], b
        assert (b["page_url"] or "").startswith("https://namu.wiki/"), b
        for k in ("avg", "ops"):
            assert b[k] is None or RATE.match(b[k]), b
        assert b["war"] is None or -5 < b["war"] < 15, b
    for p in data["pitchers"]:
        assert p["team"] in TEAMS and p["name"], p
        assert (p["page_url"] or "").startswith("https://namu.wiki/"), p
        assert p["role"] in {"선발", "마무리", "셋업"}, p
        for k in ("era", "whip"):
            assert p[k] is None or TWO.match(p[k]), p
        assert p["ip"] is None or IP.match(p["ip"]), p
        assert isinstance(p["pitches"], list)
    for team in TEAMS:  # 구단마다 주전이 실제로 있다
        assert sum(1 for b in data["batters"] if b["team"] == team and b["avg"]) >= 5, team
        assert sum(1 for p in data["pitchers"] if p["team"] == team and p["era"]) >= 4, team


def test_loader_falls_back_to_empty_on_bad_file(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_team_season(bad)["teams"] == []
    assert load_team_season(tmp_path / "none.json")["batters"] == []
    bad.write_text('{"teams": "oops"}', encoding="utf-8")
    assert load_team_season(bad)["teams"] == []


# ── 로직 ─────────────────────────────────────────────────

TEAM_DATA = {
    "season": 2026,
    "teams": [{"team": "LG", "rank": 1, "w": 80, "l": 60, "d": 4, "win_pct": 0.571}],
    "head_to_head": [{"team": "LG", "opponent": "두산", "w": 10, "l": 5, "d": 1}],
    "batters": [
        {"team": "LG", "name": "가", "position": "포수", "avg": "0.300", "hr": 20, "rbi": 80,
         "ops": "0.900", "war": 6.0},
        {"team": "LG", "name": "나", "position": "1루수", "avg": "0.250", "war": 0.5},
        {"team": "LG", "name": "다", "position": "유격수", "avg": "0.280", "war": 3.0},
        {"team": "LG", "name": "라", "position": "중견수", "avg": "0.270", "war": 2.0},
        {"team": "LG", "name": "외", "position": "좌익수", "avg": "0.310", "war": 4.0},
        {"team": "두산", "name": "남", "position": "포수", "avg": "0.200", "war": 1.0},
    ],
    "pitchers": [
        {"team": "LG", "name": "마", "role": "선발", "w": 12, "l": 5, "era": "3.10", "ip": "150.1",
         "war": 4.5, "pitches": ["직구", "슬라이더"]},
    ],
}
SALARIES = {
    "league": {"avg_salary_10k": 17000},
    "teams": [{"team": "LG", "avg_salary_10k": 20000, "top29_avg_salary_10k": 30000,
               "total_salary_10k": 1000000}],
    "players": [
        {"team": "LG", "name": "가", "salary_10k": 5000},  # 싼데 WAR 1위 → 활약
        {"team": "LG", "name": "나", "salary_10k": 100000},  # 제일 비싼데 WAR 꼴찌 → 아쉬움
        {"team": "LG", "name": "다", "salary_10k": 30000},
        {"team": "LG", "name": "라", "salary_10k": 20000},
        {"team": "두산", "name": "마", "salary_10k": 40000},  # 이적 — 이름이 하나뿐이면 붙인다
    ],
    "foreign": [{"team": "LG", "name": "외", "contract_total_usd": 1000000}],
}


def test_salary_vs_war_verdicts():
    rows = {v.name: v for v in salary_values("LG", TEAM_DATA, SALARIES)}
    assert set(rows) == {"가", "나", "다", "라", "외", "마"}  # 다른 구단 선수는 안 섞인다
    assert rows["가"].verdict == OVER and rows["가"].war_per_eok == 12.0
    assert rows["나"].verdict == UNDER and rows["나"].salary_rank == 1
    assert rows["다"].verdict == FAIR
    assert rows["외"].verdict is None and rows["외"].contract_usd == 1000000  # 외국인은 비교 안 함
    assert rows["마"].salary_10k == 40000  # 이적 선수
    assert rows["가"].line == "타율 0.300 · 20홈런 · 80타점 · OPS 0.900"
    assert rows["마"].line == "12승 5패 · 평균자책점 3.10 · 150.1이닝"
    order = [v.name for v in salary_values("LG", TEAM_DATA, SALARIES)]
    assert order[0] == "나" and order[-1] == "외"  # 연봉 큰 순, 연봉 모르면 뒤


def test_too_few_players_skip_verdicts():
    few = {**TEAM_DATA, "batters": TEAM_DATA["batters"][:2], "pitchers": []}
    assert all(v.verdict is None for v in salary_values("LG", few, SALARIES))


def test_same_name_in_two_teams_is_not_joined():
    extra = [{"team": "KT", "name": "마", "salary_10k": 9000}]
    sal = {**SALARIES, "players": SALARIES["players"] + extra}
    rows = {v.name: v for v in salary_values("LG", TEAM_DATA, sal)}
    assert rows["마"].salary_10k is None


def test_head_to_head_reads_both_directions():
    lg = head_to_head(TEAM_DATA, "LG", "두산")
    assert (lg.opponent, lg.wins, lg.losses, lg.draws) == ("두산", 10, 5, 1)
    ob = head_to_head(TEAM_DATA, "두산", "LG")
    assert (ob.opponent, ob.wins, ob.losses, ob.draws) == ("LG", 5, 10, 1)
    assert head_to_head(TEAM_DATA, "LG", "KT") is None


def test_team_report_and_names():
    r = team_report("LG", TEAM_DATA, SALARIES)
    assert r.found and r.rank == 1 and r.wins == 80
    assert r.avg_salary_10k == 20000 and r.league_avg_salary_10k == 17000
    assert [h.opponent for h in r.head_to_head] == ["두산"]
    assert r.pitchers[0].pitches == ("직구", "슬라이더")
    assert not team_report("KT", TEAM_DATA, SALARIES).found
    assert team_key("kt") == "KT" and team_key("HH") == "한화"
    assert team_key("삼성 라이온즈") == "삼성"
    assert team_key("기아") == "KIA" and team_key("없는팀") is None and team_key(None) is None


def test_hand_edited_garbage_becomes_null_not_500():
    """손으로 고친 파일의 '4위'·78.5·NaN·문자열 구종이 API를 죽이거나 판정을 지어내지 않는다."""
    bad = {
        "season": "2026시즌",
        "teams": [{"team": "LG", "rank": "4위", "w": 78.5, "l": True, "d": None,
                   "win_pct": float("nan")}],
        "head_to_head": [{"team": "LG", "opponent": "두산", "w": "10승", "l": 5, "d": 1}],
        "batters": [{**b, "war": float("nan")} if b["name"] == "가" else b
                    for b in TEAM_DATA["batters"]] + [{"team": "LG", "name": 123}],
        "pitchers": [{**TEAM_DATA["pitchers"][0], "pitches": "직구, 슬라이더", "w": "12"}],
    }
    r = team_report("LG", bad, {**SALARIES, "league": {"avg_salary_10k": 17536.4}})
    assert (r.season, r.rank, r.wins, r.losses, r.win_pct) == (None, None, None, None, None)
    assert r.league_avg_salary_10k is None
    assert r.head_to_head[0].wins is None and r.head_to_head[0].losses == 5
    assert r.pitchers[0].pitches == () and r.pitchers[0].line == "평균자책점 3.10 · 150.1이닝"
    rows = {v.name: v for v in r.values}
    assert 123 not in rows
    assert rows["가"].war is None and rows["가"].verdict is None and rows["가"].war_rank is None
    TeamAnalysisOut(team="LG", available=True, season=r.season, rank=r.rank)  # 모양 통과


def test_foreign_contract_is_not_joined_by_name_across_teams():
    sal = {**SALARIES, "foreign": [{"team": "NC", "name": "외", "contract_total_usd": 1300000}]}
    rows = {v.name: v for v in salary_values("LG", TEAM_DATA, sal)}
    assert rows["외"].contract_usd is None  # 시즌 중 이적한 외국인에게 앞 구단 계약을 붙이지 않는다


def test_salary_row_alone_is_not_a_team_report():
    assert not team_report("LG", {"teams": []}, SALARIES).found


def test_loader_reads_bom_file(tmp_path):
    f = tmp_path / "bom.json"
    f.write_text('{"teams": [{"team": "LG"}]}', encoding="utf-8-sig")
    assert load_team_season(f)["teams"] == [{"team": "LG"}]


# ── API ──────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    reset_services()
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def test_api_team_analysis(client):
    body = client.get("/api/teams/LG/analysis").json()
    assert body["available"] and body["team"] == "LG" and 1 <= body["rank"] <= 10
    assert len(body["head_to_head"]) == 9
    assert body["values"] and any(v["verdict"] for v in body["values"])
    assert client.get("/api/teams/kt/analysis").json()["team"] == "KT"
    assert client.get("/api/teams/없는팀/analysis").status_code == 404


def test_api_game_team_analysis(client):
    games = client.get("/api/games").json()
    gid = games[0]["id"]
    body = client.get(f"/api/games/{gid}/team-analysis").json()
    assert body["away"]["available"] and body["home"]["available"]
    h = body["season_head_to_head"]
    assert h["opponent"] == body["home"]["team"]
    assert h["wins"] + h["losses"] + h["draws"] == 16
    assert client.get("/api/games/nope/team-analysis").status_code == 404
