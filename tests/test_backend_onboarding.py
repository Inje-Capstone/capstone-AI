"""백엔드(Spring Boot) 연동 온보딩 — DTO 모양·1부터 번호·LearningLevel enum·가중치."""

import pytest
from fastapi.testclient import TestClient

from app.adapters.diagnostic.seed import load_diagnostic
from app.api.deps import reset_services
from app.config import get_settings
from app.domain.detectors import RULES
from app.domain.models import CATEGORY_BASIC, CATEGORY_TACTICS
from app.domain.profile import (
    LEVEL_CATEGORY_MULTIPLIER,
    UNSELECTED_CATEGORY_WEIGHT,
    profile_from_onboarding,
    weights_for,
)
from app.main import app

BASE = "/api/backend/onboarding"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    reset_services()
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def _answers(n_correct: int) -> list[dict]:
    """앞의 `n_correct`문항만 정답 — 백엔드 요청 모양(1부터)."""
    out = []
    for i, q in enumerate(load_diagnostic()):
        pick = q["answer_index"] if i < n_correct else (q["answer_index"] + 1) % len(q["choices"])
        out.append({"questionId": i + 1, "optionId": pick + 1})
    return out


def test_questions_match_backend_dto_and_hide_answers(client):
    body = client.get(f"{BASE}/questions").json()
    bank = load_diagnostic()
    assert [q["questionId"] for q in body] == list(range(1, len(bank) + 1))
    for q, src in zip(body, bank):
        assert set(q) == {"questionId", "termId", "question", "options"}
        assert [o["optionId"] for o in q["options"]] == list(range(1, len(src["choices"]) + 1))
        assert [o["text"] for o in q["options"]] == src["choices"]


@pytest.mark.parametrize(
    "n_correct, level",
    [(0, "INTRODUCTORY"), (1, "INTRODUCTORY"), (2, "BEGINNER"), (3, "FAMILIAR")],
)
def test_diagnosis_returns_backend_learning_level(client, n_correct, level):
    body = client.post(f"{BASE}/diagnosis", json={"answers": _answers(n_correct)}).json()
    assert body["correctCount"] == n_correct
    assert body["totalCount"] == 3
    assert body["learningLevel"] == level
    assert body["weights"]["learningLevel"] == level
    assert len(body["weights"]["rules"]) == len(RULES)


def test_diagnosis_tolerates_missing_and_out_of_range(client):
    answers = [{"questionId": 1, "optionId": 99}, {"questionId": 7, "optionId": 1}]
    body = client.post(f"{BASE}/diagnosis", json={"answers": answers}).json()
    assert body["correctCount"] == 0 and body["learningLevel"] == "INTRODUCTORY"
    results = body["results"]
    assert results[1]["chosenOptionId"] is None and results[1]["answerOptionId"] is None
    assert results[0]["answerOptionId"] is not None  # 답을 낸 문항에만 정답


def test_diagnosis_categories_shape_weights(client):
    body = client.post(
        f"{BASE}/diagnosis", json={"answers": _answers(3), "categories": ["전술 · 기록"]}
    ).json()
    cats = body["weights"]["categories"]
    assert cats[CATEGORY_TACTICS]["selected"] is True
    assert cats[CATEGORY_BASIC]["interestWeight"] == UNSELECTED_CATEGORY_WEIGHT


def test_weights_endpoint_accepts_enum_and_korean_label(client):
    a = client.get(f"{BASE}/weights", params={"learningLevel": "BEGINNER"}).json()
    b = client.get(f"{BASE}/weights", params={"learningLevel": "초보"}).json()
    assert a == b and a["levelLabel"] == "초보"


def test_weights_scores_match_selection_logic():
    """내보낸 shown 값이 실제 카드 선택(select)과 같은 공식인지."""
    w = weights_for("FAMILIAR", [CATEGORY_TACTICS])
    profile = profile_from_onboarding("FAMILIAR", [CATEGORY_TACTICS])
    mult = LEVEL_CATEGORY_MULTIPLIER[profile.level]
    for rule, out in zip(RULES, w["rules"]):
        interest = 1.0 if rule.category in profile.categories else UNSELECTED_CATEGORY_WEIGHT
        expected = round(rule.priority * interest * mult[rule.category], 3)
        assert out["score"] == expected
        assert out["shown"] == (expected >= profile.threshold)


def test_enum_names_feed_existing_profile():
    assert profile_from_onboarding("INTRODUCTORY").level == profile_from_onboarding("입문").level
    assert profile_from_onboarding("FAMILIAR").level == profile_from_onboarding("익숙").level
