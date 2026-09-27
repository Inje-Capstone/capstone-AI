"""E2E — mock LLM으로 API 계약 전체를 밟는다. API 키가 없어도 전량 통과해야 한다."""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import reset_services
from app.config import get_settings
from app.main import app

GAME_ID = "20260823LGOB"
DEMO_FIXTURE = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "game_20260823_LG_OB.json"
)
BALK_T = 7550
FINAL_T = 9200


@pytest.fixture(autouse=True)
def mock_backend(monkeypatch, tmp_path):
    """기본은 mock 백엔드 + 빈 스냅샷 디렉터리.

    스냅샷을 비워두는 게 중요하다 — 안 그러면 로컬에 굴러다니는 스냅샷 때문에
    실시간 생성 경로가 한 번도 실행되지 않고 테스트가 통과해버린다.
    """
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    monkeypatch.setenv("ROOKIE_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    # 로컬에 import_naver_relay.py로 받아 둔 실경기가 있어도 계약 테스트는 데모 경기만 본다.
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    shutil.copy(DEMO_FIXTURE, fixtures / DEMO_FIXTURE.name)
    monkeypatch.setenv("ROOKIE_FIXTURE_DIR", str(fixtures))
    get_settings.cache_clear()
    reset_services()
    yield
    get_settings.cache_clear()
    reset_services()


@pytest.fixture
def client():
    return TestClient(app)


def use_backend(monkeypatch, name: str) -> None:
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", name)
    get_settings.cache_clear()
    reset_services()


# ── 기본 ────────────────────────────────────────────────────────────────
def test_health_reports_backend_and_never_500s(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_backend"] == "mock"
    assert body["relay_ok"] is True
    assert body["games"] >= 1


def test_list_games_marks_video_availability(client):
    games = client.get("/api/games").json()
    assert len(games) == 1
    game = games[0]
    assert game["id"] == GAME_ID
    assert game["has_video"] is True
    assert game["score_text"] == "두산 3 : 5 LG"


def test_unknown_game_is_404(client):
    assert client.get("/api/games/nope/state").status_code == 404
    assert client.post("/api/games/nope/chat", json={"question": "왜요?"}).status_code == 404


# ── 스코어보드 ──────────────────────────────────────────────────────────
def test_state_matches_wireframe_scoreboard(client):
    body = client.get(f"/api/games/{GAME_ID}/state", params={"t": 7810}).json()
    assert body["scoreboard"] == "두산 3 : 4 LG · 7회말 · B2 S1 O2"
    assert body["bases"] == [False, True, False]
    assert body["runners_text"] == "주자 2루"


def test_state_at_end_of_game(client):
    body = client.get(f"/api/games/{GAME_ID}/state", params={"t": FINAL_T}).json()
    assert (body["away_score"], body["home_score"]) == (3, 5)
    assert body["game_over"] is True


# ── 설명 카드 ───────────────────────────────────────────────────────────
def test_cards_at_balk_returns_the_balk_card_with_reasons(client):
    body = client.get(
        f"/api/games/{GAME_ID}/cards", params={"t": BALK_T, "level": "입문"}
    ).json()
    assert body["degraded"] is False
    assert body["level"] == 0
    cards = body["cards"]
    assert cards, "보크 시점인데 카드가 하나도 없다"

    balk = next(c for c in cards if c["rule_id"] == "balk")
    assert balk["t"] == BALK_T
    assert balk["term_id"] == "balk"  # 용어 사전 딥링크
    assert balk["source"] == "mock"  # 시연임을 화면에서 밝힐 수 있어야 한다
    assert balk["can_simplify"] is False  # 이미 입문 난이도
    assert any("노출 점수" in r for r in balk["reasons"])
    assert any("감지: 보크" in r for r in balk["reasons"])


def test_cards_are_newest_first(client):
    cards = client.get(
        f"/api/games/{GAME_ID}/cards", params={"t": FINAL_T}
    ).json()["cards"]
    times = [c["t"] for c in cards]
    assert times == sorted(times, reverse=True)


def test_cards_before_any_situation_are_empty_but_not_degraded(client):
    # 첫 중계 이벤트(t=60) 이전 — 1회초 첫 타석부터 응원 문화 카드가 감지된다.
    body = client.get(f"/api/games/{GAME_ID}/cards", params={"t": 30}).json()
    assert body["cards"] == []
    assert body["degraded"] is False  # 감지된 상황 자체가 없는 것이지 고장이 아니다


def test_level_changes_the_sentence_not_just_the_selection(client):
    """개인화 실증 — 같은 상황, 같은 시점, 난이도만 다르면 문장이 달라야 한다."""
    beginner = client.get(
        f"/api/games/{GAME_ID}/cards", params={"t": BALK_T, "level": "입문"}
    ).json()["cards"]
    expert = client.get(
        f"/api/games/{GAME_ID}/cards", params={"t": BALK_T, "level": "익숙"}
    ).json()["cards"]

    b = next(c for c in beginner if c["rule_id"] == "balk")
    e = next(c for c in expert if c["rule_id"] == "balk")
    assert b["body"] != e["body"]
    assert b["title"] != e["title"]
    assert e["can_simplify"] is True


def test_interest_categories_filter_which_cards_appear(client):
    params = {"t": FINAL_T, "level": "익숙", "category": ["전술 · 기록"]}
    rules = {
        c["rule_id"]
        for c in client.get(f"/api/games/{GAME_ID}/cards", params=params).json()["cards"]
    }
    assert "double_play" in rules
    assert "balk" not in rules  # 기본 룰을 고르지 않은 '익숙' 사용자


def test_simplify_lowers_the_level_and_rewrites(client):
    cards = client.get(
        f"/api/games/{GAME_ID}/cards", params={"t": BALK_T, "level": "익숙"}
    ).json()["cards"]
    balk = next(c for c in cards if c["rule_id"] == "balk")

    res = client.post(
        f"/api/games/{GAME_ID}/cards/{balk['id']}/simplify", params={"level": "익숙"}
    )
    assert res.status_code == 200
    easier = res.json()
    assert easier["level"] == balk["level"] - 1
    assert easier["body"] != balk["body"]


def test_simplify_at_the_easiest_level_is_409(client):
    cards = client.get(
        f"/api/games/{GAME_ID}/cards", params={"t": BALK_T, "level": "입문"}
    ).json()["cards"]
    balk = next(c for c in cards if c["rule_id"] == "balk")
    res = client.post(f"/api/games/{GAME_ID}/cards/{balk['id']}/simplify")
    assert res.status_code == 409


# ── 분석 패널 ───────────────────────────────────────────────────────────
def test_matchup_returns_records_plus_optional_note(client):
    body = client.get(f"/api/games/{GAME_ID}/matchup", params={"t": 7810}).json()
    assert body["available"] is True
    assert body["batter"] == "강태림"
    assert "0.279" in body["batter_line"]
    assert body["vs_pitcher"].startswith("vs 조민서")
    assert body["team_form"] == "최근 10경기 0.600"
    assert body["ai_note"]
    assert body["source"] == "fixture"  # 데모 경기 기록은 가상이다 — 화면에서 밝힌다


# ── 챗봇 ────────────────────────────────────────────────────────────────
def test_chat_answers_with_game_context(client):
    res = client.post(
        f"/api/games/{GAME_ID}/chat",
        json={"question": "방금 저게 왜 주자가 그냥 갔어요?", "t": BALK_T, "level": "입문"},
    )
    body = res.json()
    assert body["ok"] is True
    assert body["context"].startswith("두산 3 : 4 LG · 7회말")
    assert body["used_event_ids"]  # 어떤 중계를 근거로 썼는지 밝힌다


# ── 폴백: LLM이 죽어도 화면은 산다 ───────────────────────────────────────
def test_snapshot_serves_cards_even_when_the_model_is_dead(monkeypatch, tmp_path, client):
    """RELIABILITY 폴백 계층 ② — 배치로 미리 만들어두면 API가 죽어도 카드가 나온다."""
    import json

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / f"{GAME_ID}.json").write_text(
        json.dumps(
            {
                "game_id": GAME_ID,
                "backend": "claude",
                "cards": [
                    {
                        "id": f"{GAME_ID}:e117:balk@L0",
                        "t": BALK_T,
                        "situation_id": f"{GAME_ID}:e117:balk",
                        "rule_id": "balk",
                        "term_id": "balk",
                        "category": "basic_rules",
                        "level": 0,
                        "title": "미리 만들어 둔 카드",
                        "body": "배치로 생성해 굳혀둔 설명입니다.",
                        "reasons": [],
                        "source": "llm",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    use_backend(monkeypatch, "fail")

    body = client.get(f"/api/games/{GAME_ID}/cards", params={"t": BALK_T}).json()
    balk = next(c for c in body["cards"] if c["rule_id"] == "balk")
    assert balk["title"] == "미리 만들어 둔 카드"
    assert balk["source"] == "snapshot"
    assert body["degraded"] is False


def test_llm_failure_degrades_cards_only(monkeypatch, client):
    """스냅샷마저 없을 때. 카드만 비고 나머지 화면은 정상이어야 한다."""
    use_backend(monkeypatch, "fail")

    state = client.get(f"/api/games/{GAME_ID}/state", params={"t": BALK_T})
    assert state.status_code == 200
    assert state.json()["scoreboard"].startswith("두산 3 : 4 LG")

    cards = client.get(f"/api/games/{GAME_ID}/cards", params={"t": BALK_T}).json()
    assert cards["cards"] == []
    assert cards["degraded"] is True  # 조용히 사라지지 않고 '고장'임을 알린다

    matchup = client.get(f"/api/games/{GAME_ID}/matchup", params={"t": 7810}).json()
    assert matchup["available"] is True  # 기록은 살아 있고
    assert matchup["ai_note"] is None  # 한 줄 해석만 생략된다

    chat = client.post(
        f"/api/games/{GAME_ID}/chat", json={"question": "왜요?", "t": BALK_T}
    ).json()
    assert chat["ok"] is False
    assert "다시 시도" in chat["answer"]


def test_culture_only_profile_still_gets_cards(client):
    """응원 문화만 고른 입문자도 빈 화면이 아니어야 한다."""
    cards = client.get(
        f"/api/games/{GAME_ID}/cards",
        params={"t": FINAL_T, "level": "입문", "category": "응원 문화"},
    ).json()["cards"]
    culture = [c for c in cards if c["category"] == "culture"]
    assert {c["rule_id"] for c in culture} == {"cheer_song", "homerun_cheer"}
    assert all(c["reasons"] for c in culture)
