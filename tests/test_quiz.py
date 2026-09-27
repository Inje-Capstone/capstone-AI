"""오늘 본 룰 + 퀴즈 — 화면에 뜬 카드가 곧 퀴즈 재료이고, 정답이 구조적으로 보장되는지."""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.glossary.seed import load_glossary
from app.adapters.llm.base import LLMError, LLMResult
from app.api.deps import reset_services
from app.config import get_settings
from app.main import app
from app.services.quiz_service import QuizService

GAME_ID = "20260823LGOB"
DEMO = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "game_20260823_LG_OB.json"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    monkeypatch.setenv("ROOKIE_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    shutil.copy(DEMO, fixtures / DEMO.name)
    monkeypatch.setenv("ROOKIE_FIXTURE_DIR", str(fixtures))
    get_settings.cache_clear()
    reset_services()
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


# ── 오늘 본 룰 ──────────────────────────────────────────────────────────
def test_today_rules_match_the_cards_shown(client):
    params = {"t": 9200, "level": "입문", "category": "기본 룰"}
    cards = client.get(f"/api/games/{GAME_ID}/cards", params=params).json()["cards"]
    body = client.get(f"/api/games/{GAME_ID}/today-rules", params=params).json()
    assert {r["rule_id"] for r in body["rules"]} == {c["rule_id"] for c in cards}
    assert sum(r["count"] for r in body["rules"]) == len(cards)
    firsts = [r["first_t"] for r in body["rules"]]
    assert firsts == sorted(firsts)
    assert body["term_ids"] == list(dict.fromkeys(r["term_id"] for r in body["rules"]))


def test_today_rules_respect_watch_position(client):
    early = client.get(f"/api/games/{GAME_ID}/today-rules", params={"t": 100}).json()
    late = client.get(f"/api/games/{GAME_ID}/today-rules").json()
    assert len(early["rules"]) < len(late["rules"])
    assert all(r["first_t"] <= 100 for r in early["rules"])


def test_today_rules_work_even_when_llm_is_dead(client, monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "fail")
    get_settings.cache_clear()
    reset_services()
    body = client.get(f"/api/games/{GAME_ID}/today-rules").json()
    assert body["rules"]  # LLM을 부르지 않는다


def test_today_rules_unknown_game_404(client):
    assert client.get("/api/games/nope/today-rules").status_code == 404


# ── 퀴즈 (용어 사전 문항) ────────────────────────────────────────────────
def _correct_text(item, terms, level):
    field = {0: "easy", 1: "standard", 2: "deep"}[level]
    entry = terms[item["term_id"]]
    return entry[field], entry["name"]


def test_quiz_five_items_with_structurally_correct_answers(client):
    terms = load_glossary()
    body = client.post("/api/quiz", json={
        "term_ids": ["balk", "walk", "steal"], "level": "초보", "seed": "u1-2026-09-27",
    }).json()
    items = body["items"]
    assert body["level"] == 1 and len(items) == 5
    assert {"balk", "walk", "steal"} <= {i["term_id"] for i in items}  # 본 것부터
    assert len({i["term_id"] for i in items}) == 5
    for item in items:
        assert len(item["choices"]) == 4 and len(set(item["choices"])) == 4
        definition, name = _correct_text(item, terms, 1)
        answer = item["choices"][item["answer_index"]]
        assert answer in (definition, name)
        assert item["source"] == "glossary"
        assert name not in item["question"] or item["question"].startswith(f"'{name}'")


def test_quiz_same_seed_same_quiz_different_seed_differs(client):
    req = {"term_ids": ["balk", "walk", "steal", "homerun"], "seed": "u1-day1"}
    a = client.post("/api/quiz", json=req).json()
    b = client.post("/api/quiz", json=req).json()
    c = client.post("/api/quiz", json={**req, "seed": "u1-day2"}).json()
    assert a == b
    assert a != c


def test_quiz_with_nothing_watched_still_returns_count(client):
    body = client.post("/api/quiz", json={"count": 3}).json()
    assert len(body["items"]) == 3


def test_quiz_ignores_unknown_terms_and_validates_count(client):
    body = client.post("/api/quiz", json={"term_ids": ["nope", "balk"], "count": 2}).json()
    assert "nope" not in {i["term_id"] for i in body["items"]}
    assert client.post("/api/quiz", json={"count": 0}).status_code == 422
    assert client.post("/api/quiz", json={"count": 11}).status_code == 422


# ── 퀴즈 (LLM 상황형 문항) ───────────────────────────────────────────────
class _FakeLLM:
    source = "llm"
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.users = []

    def complete(self, **kw):
        self.users.append(kw["user"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(text=reply, model="fake", source="llm")


GOOD = json.dumps({
    "question": "1루 주자가 있을 때 투수가 던지다 멈추면?",
    "choices": ["주자가 한 베이스 간다", "타자 아웃", "볼 하나 추가", "아무 일 없다"],
    "answer_index": 0, "explanation": "보크라서 주자가 진루합니다.",
}, ensure_ascii=False)


def test_llm_item_is_shuffled_and_answer_tracked():
    svc = QuizService(load_glossary(), _FakeLLM([GOOD]))
    (item,) = svc.build(["balk"], level=0, count=1, seed="s")
    assert item.source == "llm"
    assert item.choices[item.answer_index] == "주자가 한 베이스 간다"
    assert "[정답 설명]" in svc.llm.users[0]


@pytest.mark.parametrize("bad", [
    LLMError("down"),
    "not json",
    json.dumps({"question": "q", "choices": ["a", "b", "c"], "answer_index": 0,
                "explanation": ""}),
    json.dumps({"question": "q", "choices": ["a", "a", "b", "c"], "answer_index": 0,
                "explanation": ""}),
    json.dumps({"question": "q", "choices": ["a", "b", "c", "d"], "answer_index": 4,
                "explanation": ""}),
    json.dumps({"question": "q", "choices": ["a", "b", "c", "d"], "answer_index": True,
                "explanation": ""}),
])
def test_bad_llm_item_falls_back_to_glossary(bad):
    svc = QuizService(load_glossary(), _FakeLLM([bad]))
    (item,) = svc.build(["balk"], level=0, count=1, seed="s")
    assert item.source == "glossary"
    assert item.term_id == "balk"
