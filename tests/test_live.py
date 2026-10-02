"""실시간 모드 — 청크 결과가 들어오는 대로 판정·노출, 일반 경기 엔드포인트로도 보인다."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.relay.fixture import FixtureRelaySource
from app.api.deps import reset_services
from app.config import get_settings
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
TRUTH = "20260920HHLG02026"
TOKEN = "test-token"
H = {"X-Live-Token": TOKEN}


def _load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    monkeypatch.setenv("ROOKIE_LIVE_TOKEN", TOKEN)
    monkeypatch.setenv("ROOKIE_SNAPSHOT_DIR", str(tmp_path / "snap"))
    monkeypatch.setenv("ROOKIE_VIDEO_DIR", str(tmp_path / "video"))
    get_settings.cache_clear()
    reset_services()
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def _chunks(seconds=10.0, limit_t=2400.0):
    """정답 중계로 만든 '완벽한 눈' 판독·단서를 10초 청크로 나눈 ingest 본문들."""
    sim = _load("simulate_video_judge")
    feed = FixtureRelaySource(ROOT / "data" / "relay_truth").load(TRUTH)
    readings, cues = sim.perfect_eye(feed.events, feed.meta.away_team, feed.meta.home_team)
    buckets: dict[int, dict] = {}
    for r in readings:
        if r.t > limit_t:
            continue
        b = buckets.setdefault(int(r.t // seconds), {"scoreboard": [], "events": [], "speech": []})
        b["scoreboard"].append({"t": r.t, "inning": r.inning, "half": r.half, "balls": r.balls,
                                "strikes": r.strikes, "outs": r.outs,
                                "bases": [n for n, on in zip((1, 2, 3), r.bases) if on],
                                "away": r.away, "home": r.home})
    for c in cues:
        if c.t > limit_t:
            continue
        b = buckets.setdefault(int(c.t // seconds), {"scoreboard": [], "events": [], "speech": []})
        if c.kind == "speech":
            b["speech"].append({"t": c.t, "text": c.text})
        else:
            b["events"].append({"t_start": c.t, "t_end": c.t + 1, "event_type": c.kind,
                                "description": c.kind})
    return [buckets[k] for k in sorted(buckets)]


def test_token_required(client, monkeypatch):
    body = {"away_team": "한화", "home_team": "LG"}
    assert client.post("/api/live/L1", json=body).status_code == 401
    assert client.post("/api/live/L1", json=body, headers={"X-Live-Token": "no"}).status_code == 401
    monkeypatch.setenv("ROOKIE_LIVE_TOKEN", "")
    get_settings.cache_clear()
    assert client.post("/api/live/L1", json=body, headers=H).status_code == 403


def test_live_game_grows_and_is_served_like_any_game(client):
    assert client.post("/api/live/L1", json={"away_team": "한화", "home_team": "LG"},
                       headers=H).status_code == 200
    chunks = _chunks()
    new_rules = []
    first_ids = None
    for i, body in enumerate(chunks):
        res = client.post("/api/live/L1/ingest", json=body, headers=H).json()
        new_rules += res["new_situations"]
        if i == len(chunks) // 2:
            first_ids = {e["id"] for e in _events(client)}
    # 실시간(청크마다 다시 판정)과 한꺼번에 판정한 결과가 같아야 한다
    from app.domain.detectors import detect
    from app.domain.models import GameFeed, GameMeta
    from app.domain.video_judge import from_snapshot, judge, to_relay_events

    whole = {k: [x for b in chunks for x in b[k]] for k in ("scoreboard", "events", "speech")}
    batch = to_relay_events(judge(*from_snapshot(whole)))
    meta = GameMeta(id="B", date="", stadium="", away_team="한화", home_team="LG")
    expected = [s.rule_id for s in detect(GameFeed(meta=meta, events=batch))]
    assert sorted(new_rules) == sorted(expected) and "walk" in new_rules
    # 판정이 늘어도 앞서 나온 이벤트 id는 그대로다 (카드 캐시 키가 흔들리지 않게)
    assert first_ids and first_ids <= {e["id"] for e in _events(client)}

    games = {g["id"]: g for g in client.get("/api/games").json()}
    assert games["L1"]["has_video"] is True
    state = client.get("/api/games/L1/state").json()
    assert state["away_team"] == "한화"
    cards = client.get("/api/games/L1/cards", params={"level": "입문"}).json()["cards"]
    assert cards and all(any(r.startswith("감지:") for r in c["reasons"]) for c in cards)
    assert client.get("/api/games/L1/moment").status_code == 200

    assert client.post("/api/live/L1/end", headers=H).status_code == 200
    assert client.post("/api/live/L1/ingest", json=chunks[0], headers=H).status_code == 409

    with client.stream("GET", "/api/live/L1/stream") as res:
        lines = [ln for ln in res.iter_lines() if ln.startswith("data: ")]
    msgs = [json.loads(ln[6:]) for ln in lines]
    assert msgs[-1] == {"type": "end"}
    assert sum(len(m.get("situations", [])) for m in msgs) == len(new_rules)
    with client.stream("GET", "/api/live/L1/stream", params={"since": len(msgs) - 1}) as res:
        rest = [ln for ln in res.iter_lines() if ln.startswith("data: ")]
    assert [json.loads(ln[6:]) for ln in rest] == [{"type": "end"}]


def _events(client):
    from app.api.deps import get_live_source

    return [e.model_dump() for e in get_live_source().get("L1").feed.events]


def test_unknown_live_game_404(client):
    assert client.get("/api/live/nope/stream").status_code == 404
    assert client.post("/api/live/nope/ingest", json={}, headers=H).status_code == 404


# ── 분석기 스크립트 ─────────────────────────────────────────────────────
def test_segment_cmd_realtime_flag():
    live = _load("live_video")
    cmd = live.segment_cmd("ffmpeg", "g.mp4", Path("out"), 10, 360, realtime=True)
    assert cmd[cmd.index("-re") + 2] == "g.mp4" and "segment" in cmd
    assert "-re" not in live.segment_cmd("ffmpeg", "g.mp4", Path("out"), 10, 360, False)


def test_ready_segments_waits_for_next_while_producing(tmp_path):
    live = _load("live_video")
    for n in range(3):
        (tmp_path / f"seg_{n:05d}.mp4").write_bytes(b"x")
    assert [p.name for p in live.ready_segments(tmp_path, set(), True)] == [
        "seg_00000.mp4", "seg_00001.mp4"]
    assert len(live.ready_segments(tmp_path, {"seg_00000.mp4"}, False)) == 2


def test_run_sends_each_segment_once_with_offsets(tmp_path):
    live = _load("live_video")
    from app.adapters.video.base import ClipAnalysis

    for n in range(3):
        (tmp_path / f"seg_{n:05d}.mp4").write_bytes(b"x")

    class Analyzer:
        def analyze(self, clip):
            return ClipAnalysis(events=[], speech=[], scoreboard=[{"t": clip.start + 1}])

    class FakeServer:
        def __init__(self):
            self.bodies = []

        def post(self, path, body):
            self.bodies.append((path, body))
            return {"new_situations": []}

    server = FakeServer()
    sent = live.run("L1", Analyzer(), server, tmp_path, 10.0, None, workers=2, log=lambda m: None)
    assert sent == 3
    starts = sorted(b["scoreboard"][0]["t"] for _, b in server.bodies)
    assert starts == [1.0, 11.0, 21.0]
    assert {p for p, _ in server.bodies} == {"/api/live/L1/ingest"}


# ── 실시간 데이터 보강 (선수·구종·기록) ──────────────────────────────────
def _data_body():
    raw = json.loads((ROOT / "data" / "relay_truth" / f"naver_{TRUTH}.json")
                     .read_text(encoding="utf-8"))
    return {"events": raw["events"], "context": raw["game"].get("context") or {}}


def test_live_data_adds_names_after_offset_is_locked(client):
    client.post("/api/live/L2", json={"away_team": "한화", "home_team": "LG"}, headers=H)
    chunks = _chunks(limit_t=3000)
    # 데이터 먼저 와도 영상 단서가 모자라면 시간차를 고정하지 않는다
    res = client.post("/api/live/L2/data", json=_data_body(), headers=H).json()
    assert res["offset"] is None and res["data_events"] > 100
    for body in chunks:
        client.post("/api/live/L2/ingest", json=body, headers=H)
    res = client.post("/api/live/L2/data", json=_data_body(), headers=H).json()
    assert res["offset"] == 0.0  # 합성 영상은 데이터와 같은 시각

    state = client.get("/api/games/L2/state", params={"t": 1500}).json()
    assert state["batter"] and state["pitcher"]  # 데이터로 붙은 선수 이름
    matchup = client.get("/api/games/L2/matchup", params={"t": 1500}).json()
    assert matchup["available"] and "시즌 타율" in matchup["batter_line"]
    assert "위" in matchup["team_form"]  # 데이터의 팀 맥락
    rules = {c["rule_id"] for c in
             client.get("/api/games/L2/cards", params={"level": "익숙"}).json()["cards"]}
    assert "cheer_song" in rules  # 데이터의 타석 시작이 붙어야 뜨는 카드
    from app.api.deps import get_live_source

    feed = get_live_source().get("L2").feed
    assert not any(e.detail.get("source") == "data" and e.kind == "result" for e in feed.events)
    client.post("/api/live/L2/end", headers=H)  # 끝나야 스트림이 닫힌다
    with client.stream("GET", "/api/live/L2/stream") as r:
        msgs = [json.loads(ln[6:]) for ln in r.iter_lines() if ln.startswith("data: ")]
    assert msgs[0]["data"] is False  # 첫 업데이트 땐 아직 데이터 미적용
    assert any(m.get("data") for m in msgs) and msgs[-1] == {"type": "end"}


def test_live_data_rejects_bad_events(client):
    client.post("/api/live/L3", json={"away_team": "A", "home_team": "B"}, headers=H)
    bad = {"events": [{"id": "x"}], "context": {}}
    assert client.post("/api/live/L3/data", json=bad, headers=H).status_code == 422
    assert client.post("/api/live/L3/data", json=_data_body()).status_code == 401


def test_data_poller_survives_fetch_errors():
    live = _load("live_video")
    from app.adapters.relay.naver import NaverRelayError

    calls = {"n": 0}

    def payload(naver_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise NaverRelayError("네트워크")
        return {"events": [{"id": "a"}], "context": {}}

    class FakeServer:
        def __init__(self):
            self.paths = []

        def post(self, path, body):
            self.paths.append(path)
            return {"offset": 12.0}

    server, logs = FakeServer(), []
    poller = live.DataPoller("N1", server, "L1", every=0, log=logs.append, payload=payload)
    poller.poll_once()
    poller.poll_once()
    assert server.paths == ["/api/live/L1/data"] and poller.sent == 1
    assert "실패" in logs[0] and "+12s" in logs[1]


def test_data_payload_uses_converter():
    live = _load("live_video")
    raw = json.loads((ROOT / "data" / "relay_truth" / f"naver_{TRUTH}.json")
                     .read_text(encoding="utf-8"))
    assert raw["events"]  # 변환 결과 형식 = /data 본문 형식
    fake_info = {"gameId": "G", "awayTeamName": "A", "homeTeamName": "H"}

    def fetch(gid):
        option = {"seqno": 1, "type": 8, "text": "1번타자 가", "currentGameState": {}}
        relay = {"inn": 1, "homeOrAway": "0", "no": 1, "textOptions": [option]}
        return fake_info, [{"textRelays": [relay]}]

    body = live.data_payload("G", fetch=fetch, preview=lambda gid: None)
    assert [e["kind"] for e in body["events"]] == ["atbat"] and body["context"] == {}
