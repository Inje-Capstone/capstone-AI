"""응원가 안내 — 모양 검사·선수 우선·팀 응원가 대체·가사 없음."""

import json

import pytest
from fastapi.testclient import TestClient

from app.adapters.cheer.seed import CHEER_PATH, load_cheer_songs, song_for, valid_song
from app.api import routes_cheer
from app.api.deps import reset_services
from app.config import get_settings
from app.main import app

SONGS = [
    {"team": "두산", "player": None, "title": "두산 팀 응원가", "url": "https://example.com/team"},
    {"team": "두산", "player": "임하람", "title": "임하람 응원가", "url": "https://example.com/p"},
    {"team": "LG", "player": None, "title": "LG 팀 응원가", "url": "https://example.com/lg"},
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    reset_services()
    monkeypatch.setattr(routes_cheer, "get_cheer_songs", lambda: SONGS)
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def _game_id(client) -> str:
    return client.get("/api/games").json()[0]["id"]


def test_committed_file_loads_and_has_no_lyrics_field():
    raw = json.loads(CHEER_PATH.read_text(encoding="utf-8"))
    for row in raw["songs"]:
        assert set(row) <= {"team", "player", "title", "url", "note"}
    assert isinstance(load_cheer_songs(), list)


def test_bad_rows_are_skipped(tmp_path):
    p = tmp_path / "songs.json"
    p.write_text(json.dumps({"songs": [
        SONGS[0],
        {"team": "없는팀", "player": None, "title": "x", "url": "https://a"},
        {"team": "LG", "player": None, "title": "x", "url": "http://insecure"},
        {"team": "LG", "player": " ", "title": "x", "url": "https://a"},
        "not a dict",
    ]}), encoding="utf-8")
    assert load_cheer_songs(p) == [SONGS[0]]
    assert not valid_song({"team": "LG", "title": "", "url": "https://a"})


def test_broken_file_gives_empty_list_and_player_is_trimmed(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_cheer_songs(broken) == []
    as_list = tmp_path / "list.json"
    as_list.write_text("[]", encoding="utf-8")
    assert load_cheer_songs(as_list) == []
    assert load_cheer_songs(tmp_path / "missing.json") == []
    spaced = tmp_path / "spaced.json"
    spaced.write_text(json.dumps({"songs": [
        {"team": "두산", "player": " 임하람 ", "title": " 곡 ", "url": "https://a"},
    ]}), encoding="utf-8")
    rows = load_cheer_songs(spaced)
    assert rows[0]["player"] == "임하람" and rows[0]["title"] == "곡"
    assert song_for(rows, "두산", "임하람") is rows[0]


def test_player_song_first_then_team_song():
    assert song_for(SONGS, "두산", "임하람")["title"] == "임하람 응원가"
    assert song_for(SONGS, "두산", "다른타자")["title"] == "두산 팀 응원가"
    assert song_for(SONGS, "한화", "누구") is None
    assert song_for(SONGS, None, "임하람") is None


def test_list_filters_by_team(client):
    assert len(client.get("/api/cheer-songs").json()) == 3
    assert [s["title"] for s in client.get("/api/cheer-songs", params={"team": "LG"}).json()] == [
        "LG 팀 응원가"
    ]
    assert client.get("/api/cheer-songs", params={"team": "XX"}).status_code == 404


def test_game_cheer_song_for_current_batter(client):
    gid = _game_id(client)
    body = client.get(f"/api/games/{gid}/cheer-song").json()
    assert body["available"] is True
    assert body["song"]["team"] == body["batting_team"]
    before = client.get(f"/api/games/{gid}/cheer-song", params={"t": 0}).json()
    assert before["available"] is False and before["song"] is None


def test_game_cheer_song_unknown_game_404(client):
    assert client.get("/api/games/nope/cheer-song").status_code == 404
