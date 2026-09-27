"""용어 사전(S5) API — 카드 딥링크가 끝까지 이어지는지."""

import pytest
from fastapi.testclient import TestClient

from app.adapters.glossary.seed import load_glossary
from app.api.deps import reset_services
from app.config import get_settings
from app.main import app

GAME_ID = "20260823LGOB"
FINAL_T = 9200


@pytest.fixture
def client(monkeypatch, tmp_path):
    # 로컬 .env에 키가 있어도 실제 모델을 부르지 않게 — test_api와 같은 격리.
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    monkeypatch.setenv("ROOKIE_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    get_settings.cache_clear()
    reset_services()
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def test_list_returns_every_seed_term_sorted_by_name(client):
    body = client.get("/api/glossary").json()
    assert {t["id"] for t in body} == set(load_glossary())
    names = [t["name"] for t in body]
    assert names == sorted(names)
    assert all(t["summary"] and t["category_label"] for t in body)


def test_search_matches_alias_ignoring_spaces(client):
    body = client.get("/api/glossary", params={"q": "낫 아웃"}).json()
    assert [t["id"] for t in body] == ["dropped_third_strike"]
    body = client.get("/api/glossary", params={"q": "데드볼"}).json()
    assert [t["id"] for t in body] == ["hit_by_pitch"]


def test_category_filter_accepts_onboarding_labels(client):
    body = client.get("/api/glossary", params={"category": "응원 문화"}).json()
    assert {t["id"] for t in body} == {"cheer_song", "homerun_celebration"}
    assert all(t["category_label"] == "응원 문화" for t in body)


def test_detail_body_follows_level(client):
    easy = client.get("/api/glossary/balk", params={"level": "입문"}).json()
    deep = client.get("/api/glossary/balk", params={"level": "익숙"}).json()
    assert easy["level"] == 0 and deep["level"] == 2
    assert easy["body"] != deep["body"]
    assert easy["body"] == easy["levels"]["입문"]
    assert deep["body"] == deep["levels"]["익숙"]
    assert set(easy["levels"]) == {"입문", "초보", "익숙"}


def test_detail_includes_resolvable_related_terms(client):
    body = client.get("/api/glossary/walk").json()
    assert "hit_by_pitch" in {r["id"] for r in body["related"]}


def test_unknown_term_is_404(client):
    assert client.get("/api/glossary/nope").status_code == 404


def test_every_related_key_exists_in_seed():
    """관련 용어 칩이 404로 끝나지 않게."""
    terms = load_glossary()
    dangling = {(k, r) for k, e in terms.items() for r in e.get("related", []) if r not in terms}
    assert dangling == set()


def test_every_card_term_id_deep_links(client):
    """카드의 term_id를 누르면 S5 상세가 열려야 한다 (전 난이도·전 카테고리)."""
    for level in ("입문", "초보", "익숙"):
        cards = client.get(
            f"/api/games/{GAME_ID}/cards", params={"t": FINAL_T, "level": level}
        ).json()["cards"]
        assert cards
        for card in cards:
            res = client.get(f"/api/glossary/{card['term_id']}")
            assert res.status_code == 200, card["term_id"]
