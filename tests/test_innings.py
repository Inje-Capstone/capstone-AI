"""이닝 시청 분석 — 반 이닝 구간·본 초·올림 규칙(5회초 우천 취소)·API."""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import reset_services
from app.config import get_settings
from app.domain.innings import LAST_HALF_SEC, half_spans, innings_watched, watch
from app.domain.models import RelayEvent
from app.domain.timeline import Timeline
from app.main import app


def ev(t, inning, half):
    return RelayEvent(id=f"e{t}", t=t, inning=inning, half=half, kind="pitch", text="")


# 4회말까지 하고 5회초 도중 우천 취소된 경기(반 이닝 100초씩, 영상 오프셋 30초)
RAIN = [ev(100 * i + k * 10, i // 2 + 1, "top" if i % 2 == 0 else "bot")
        for i in range(9) for k in range(3)]


def test_half_spans_follow_events_and_offset():
    spans = half_spans(RAIN, Timeline(30), video_end=1000)
    assert [(s.inning, s.half) for s in spans][:3] == [(1, "top"), (1, "bot"), (2, "top")]
    assert (spans[0].start, spans[0].end) == (30, 130)
    assert spans[-1].inning == 5 and spans[-1].half == "top"
    assert spans[-1].end == 1000  # 마지막 반 이닝은 영상 끝까지
    no_end = half_spans(RAIN, Timeline(0))
    assert no_end[-1].end == 820 + LAST_HALF_SEC


def test_rain_called_in_the_fifth_top_counts_five_innings():
    """5회초 우천 취소 — 5회초를 봤으면 5회로 인정(올림)."""
    spans = half_spans(RAIN, Timeline(0), video_end=900)
    halves = watch(spans, [(0, 900)])
    assert innings_watched(halves) == [1, 2, 3, 4, 5]


def test_half_inning_needs_half_of_its_span_and_overlaps_count_once():
    spans = half_spans(RAIN, Timeline(0), video_end=900)
    # 1회초(0~100) 49초만 — 안 셈. 겹친 구간은 한 번만
    halves = watch(spans, [(0, 30), (10, 49)])
    assert halves[0].watched_sec == 49 and not halves[0].counted
    assert innings_watched(halves) == []
    halves = watch(spans, [(0, 30), (20, 50)])
    assert halves[0].watched_sec == 50 and halves[0].counted
    # 1회말만 봐도 1회(올림)
    assert innings_watched(watch(spans, [(100, 200)])) == [1]
    assert innings_watched(watch(spans, [])) == []
    assert innings_watched(watch(spans, [(50, 40)])) == []  # 끝이 시작보다 앞이면 무시


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    reset_services()
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def test_api_innings_watched(client):
    gid = client.get("/api/games").json()[0]["id"]
    full = client.post(f"/api/games/{gid}/innings-watched",
                       json={"ranges": [{"start": 0, "end": 86_400}]}).json()
    assert full["innings"] == list(range(1, full["innings_count"] + 1))
    assert full["innings_count"] >= 9
    assert all(h["counted"] for h in full["halves"])
    none = client.post(f"/api/games/{gid}/innings-watched", json={"ranges": []}).json()
    assert none["innings_count"] == 0 and none["rule"]
    assert client.post("/api/games/nope/innings-watched", json={"ranges": []}).status_code == 404
    bad = client.post(f"/api/games/{gid}/innings-watched",
                      json={"ranges": [{"start": -1, "end": 5}]})
    assert bad.status_code == 422
