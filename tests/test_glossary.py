"""용어 사전(S5) API — 카드 딥링크가 끝까지 이어지는지."""

import pytest
from fastapi.testclient import TestClient

from app.adapters.glossary.seed import load_glossary, load_rules
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


def test_rules_list_every_catalog_rule_by_name_ascending(client):
    body = client.get("/api/rules").json()
    assert {r["id"] for r in body} == {r["id"] for r in load_rules()}
    assert len(body) == 302
    names = [r["name"] for r in body]
    assert names == sorted(names)


def test_rules_sort_descending(client):
    names = [r["name"] for r in client.get("/api/rules", params={"sort": "-name"}).json()]
    assert names == sorted(names, reverse=True)


def test_each_category_button_splits_the_rules_without_overlap(client):
    labels = ("기본 룰", "구종", "전술 · 기록", "심판 콜", "응원 문화")
    parts = [client.get("/api/rules", params={"category": c}).json() for c in labels]
    ids = [r["id"] for part in parts for r in part]
    assert len(ids) == len(set(ids)) == 302
    assert all(part for part in parts)


def test_balk_rules_sit_under_umpire_calls_as_in_the_wireframe(client):
    calls = client.get("/api/rules", params={"category": "심판 콜"}).json()
    balk = {r["id"] for r in load_rules() if r["chapter"].startswith("D-2.")}
    assert balk and balk <= {r["id"] for r in calls}
    assert all(r["category_label"] == "심판 콜" for r in calls)


def test_category_label_accepts_spacing_and_full_name(client):
    full = client.get("/api/rules", params={"category": "구종 · 투구"}).json()
    assert full == client.get("/api/rules", params={"category": "구종"}).json()


def test_unknown_category_or_sort(client):
    assert client.get("/api/rules", params={"category": "없는 버튼"}).json() == []
    assert client.get("/api/rules", params={"sort": "learned"}).status_code == 422


def test_rule_stays_apart_from_the_term_of_the_same_name(client):
    rule = next(r for r in client.get("/api/rules").json() if r["name"] == "볼넷")
    assert (rule["id"], rule["term_id"]) == ("walk_basic", "walk")
    assert rule["rule_ref"] == "5.05(b)(1)"
    terms = [t["id"] for t in client.get("/api/glossary").json() if t["name"] == "볼넷"]
    assert terms == ["walk"]


def test_rule_shows_situation_ruling_and_clause(client):
    rule = next(r for r in client.get("/api/rules").json() if r["id"] == "box_switch")
    assert rule == {
        "id": "box_switch", "name": "투수 준비 중 타석 바꿈", "chapter": "B. 타자·타자주자",
        "category_label": "기본 룰", "rule_ref": "6.03(a)(2)",
        "action": "투수 세트 후 반대 타석으로 이동", "ruling": "타자 아웃",
        "term_id": "batter_rules",
    }


def test_rule_without_a_clause_has_empty_reference(client):
    rule = next(r for r in client.get("/api/rules").json() if r["id"] == "squeeze_r")
    assert rule["rule_ref"] == ""


def test_every_rule_links_a_term_that_opens(client):
    terms = set(load_glossary())
    assert {r["term_id"] for r in client.get("/api/rules").json()} <= terms


def test_search_matches_alias_ignoring_spaces(client):
    body = client.get("/api/glossary", params={"q": "낫 아웃"}).json()
    assert [t["id"] for t in body] == ["dropped_third_strike"]
    body = client.get("/api/glossary", params={"q": "데드볼"}).json()
    assert [t["id"] for t in body] == ["hit_by_pitch"]


def test_category_filter_accepts_onboarding_labels(client):
    body = client.get("/api/glossary", params={"category": "응원 문화"}).json()
    culture = {k for k, v in load_glossary().items() if v["category"] == "culture"}
    assert {t["id"] for t in body} == culture
    assert {"cheer_song", "homerun_celebration"} <= culture
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
