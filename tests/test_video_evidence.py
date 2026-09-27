"""영상 관찰을 카드 근거·챗봇 맥락에 붙이기 — 판정은 바꾸지 않고, 없으면 조용히 빠진다."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.video.notes import VideoNotes, clock, near, recent
from app.api.deps import reset_services
from app.config import get_settings
from app.main import app

GAME_ID = "20260823LGOB"
DEMO = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "game_20260823_LG_OB.json"
BALK_T = 7550


def _video_snapshot(path: Path, events):
    path.write_text(json.dumps({"game_id": GAME_ID, "events": events}, ensure_ascii=False),
                    encoding="utf-8")


def _ev(t, kind="other", desc=""):
    return {"t_start": t, "t_end": t + 2, "event_type": kind, "description": desc,
            "provider": "fake", "model": "fake"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    fixtures, video = tmp_path / "fixtures", tmp_path / "video"
    fixtures.mkdir()
    video.mkdir()
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    monkeypatch.setenv("ROOKIE_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setenv("ROOKIE_FIXTURE_DIR", str(fixtures))
    monkeypatch.setenv("ROOKIE_VIDEO_DIR", str(video))
    shutil.copy(DEMO, fixtures / DEMO.name)
    get_settings.cache_clear()
    reset_services()
    yield fixtures / DEMO.name, video
    get_settings.cache_clear()
    reset_services()


def _balk_card(client, **params):
    cards = client.get(f"/api/games/{GAME_ID}/cards",
                       params={"t": BALK_T, "level": "입문", **params}).json()["cards"]
    return next(c for c in cards if c["rule_id"] == "balk")


def test_cards_unchanged_without_video_snapshot(env):
    card = _balk_card(TestClient(app))
    assert not any("영상 관찰" in r for r in card["reasons"])


def test_nearby_observation_added_to_card_reasons(env):
    _, video = env
    _video_snapshot(video / f"{GAME_ID}.json", [
        _ev(BALK_T + 3, "other", "투수가 투구 동작 중 멈춤"),
        _ev(BALK_T + 200, "hit", "먼 장면"),
    ])
    card = _balk_card(TestClient(app))
    lines = [r for r in card["reasons"] if "영상 관찰" in r]
    assert len(lines) == 1 and "투구 동작 중 멈춤" in lines[0] and "판정 아님" in lines[0]


def test_card_t_is_video_timecode_when_offset_applied(env):
    fixture, video = env
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    raw["game"]["relay_video_offset_sec"] = 300
    fixture.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    get_settings.cache_clear()
    reset_services()
    _video_snapshot(video / f"{GAME_ID}.json", [_ev(BALK_T + 300 + 2, "other", "보크 장면")])

    card = _balk_card(TestClient(app), t=BALK_T + 300)
    assert card["t"] == BALK_T + 300  # 프론트가 이 값으로 영상을 시킹한다
    assert any("보크 장면" in r for r in card["reasons"])


def test_broken_snapshot_is_ignored(env):
    _, video = env
    (video / f"{GAME_ID}.json").write_text("{not json", encoding="utf-8")
    assert _balk_card(TestClient(app))["reasons"]


def test_notes_reload_when_file_changes(tmp_path):
    notes = VideoNotes(tmp_path)
    path = tmp_path / "G.json"
    assert notes.events("G") == []
    _video_snapshot(path, [_ev(10)])
    assert len(notes.events("G")) == 1
    import os
    import time

    _video_snapshot(path, [_ev(10), _ev(20)])
    later = time.time() + 5
    os.utime(path, (later, later))
    assert len(notes.events("G")) == 2


def test_windows_and_clock():
    from app.adapters.video.base import VideoEvent

    evs = [VideoEvent(t_start=t, t_end=t, event_type="hit") for t in (90, 100, 114, 200)]
    assert [e.t_start for e in near(evs, 100)] == [100, 114]
    assert [e.t_start for e in recent(evs, 100, window=10)] == [90, 100]
    assert clock(65) == "1:05" and clock(3725) == "1:02:05"


def test_chat_prompt_gets_recent_video_lines(env):
    _, video = env
    _video_snapshot(video / f"{GAME_ID}.json", [_ev(BALK_T - 20, "other", "투수가 멈칫함")])
    from app.api.deps import get_chat_service

    service = get_chat_service()
    seen = {}
    real = service.llm.complete

    def spy(**kw):
        seen["user"] = kw["user"]
        return real(**kw)

    service.llm.complete = spy
    TestClient(app).post(f"/api/games/{GAME_ID}/chat",
                         json={"question": "방금 뭐예요?", "t": BALK_T})
    assert "[영상 장면(자동 관찰, 틀릴 수 있음)]" in seen["user"]
    assert "투수가 멈칫함" in seen["user"]
