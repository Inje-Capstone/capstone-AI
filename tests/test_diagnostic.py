"""온보딩 수준 진단 — 채점 경계, 정답 비노출, 모델 없이도 동작.

문항 본문에 의존하지 않는다(문제은행을 고쳐도 테스트가 살아 있도록) — 정답은
시드에서 읽어 만든다.
"""

import pytest
from fastapi.testclient import TestClient

from app.adapters.diagnostic.seed import load_diagnostic
from app.adapters.glossary.seed import load_glossary
from app.api.deps import reset_services
from app.config import get_settings
from app.domain.diagnostic import (
    RESULT_MESSAGES,
    SPEC_TOTAL,
    grade,
    level_for,
)
from app.domain.models import LEVEL_BEGINNER, LEVEL_FAMILIAR, LEVEL_NOVICE
from app.main import app

ENDPOINT = "/api/onboarding/diagnostic"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    reset_services()
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def _answers(n_correct: int) -> list[int]:
    """앞의 `n_correct`문항만 정답으로 채운 답안."""
    picks = []
    for i, q in enumerate(load_diagnostic()):
        answer = q["answer_index"]
        picks.append(answer if i < n_correct else (answer + 1) % len(q["choices"]))
    return picks


# ── 문제은행 자체 ────────────────────────────────────────────────────────
def test_bank_is_three_valid_questions_linked_to_glossary():
    questions = load_diagnostic()
    terms = load_glossary()
    assert len(questions) == SPEC_TOTAL
    assert len({q["id"] for q in questions}) == SPEC_TOTAL
    for q in questions:
        choices = q["choices"]
        assert len(choices) == 4 and len(set(choices)) == 4
        assert 0 <= q["answer_index"] < len(choices)
        assert q["term_id"] in terms  # 결과 화면에서 용어 사전으로 딥링크
        assert q["question"].strip() and q["explanation"].strip()


# ── 문항 조회 ────────────────────────────────────────────────────────────
def test_questions_endpoint_hides_the_answers(client):
    body = client.get(ENDPOINT).json()
    assert body["total"] == SPEC_TOTAL and len(body["questions"]) == SPEC_TOTAL
    for q in body["questions"]:
        assert set(q) == {"id", "term_id", "question", "choices"}
        assert len(q["choices"]) == 4


# ── 채점 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("n_correct", "level"),
    [(0, LEVEL_BEGINNER), (1, LEVEL_BEGINNER), (2, LEVEL_NOVICE), (3, LEVEL_FAMILIAR)],
)
def test_grading_boundaries_follow_the_spec(client, n_correct, level):
    body = client.post(ENDPOINT, json={"answers": _answers(n_correct)}).json()
    assert body["correct_count"] == n_correct
    assert body["total"] == SPEC_TOTAL
    assert body["level"] == level
    assert body["message"] == RESULT_MESSAGES[level]
    assert body["reasons"]  # 근거 없는 판정 금지 (core-belief 4)
    assert sum(1 for a in body["answers"] if a["correct"]) == n_correct


def test_result_level_label_feeds_other_endpoints(client):
    body = client.post(ENDPOINT, json={"answers": _answers(2)}).json()
    assert body["level_label"] == "초보"
    cards = client.get("/api/glossary", params={"level": body["level_label"]})
    assert cards.status_code == 200  # 그대로 다음 요청의 level로 쓸 수 있다


def test_out_of_range_answers_count_wrong_instead_of_erroring(client):
    body = client.post(ENDPOINT, json={"answers": [99, -5, 99]}).json()
    assert body["correct_count"] == 0
    assert body["level"] == LEVEL_BEGINNER


def test_missing_answers_count_wrong(client):
    """문항 수보다 짧은 답안이 와도 채점은 끝난다 — 온보딩이 막히면 가입이 끝난다."""
    body = client.post(ENDPOINT, json={"answers": [_answers(1)[0]]}).json()
    assert body["total"] == SPEC_TOTAL and body["correct_count"] == 1
    assert body["answers"][-1]["chosen_index"] is None


def test_unanswered_questions_do_not_reveal_the_answer_key(client):
    """빈 답안 한 번으로 정답표를 받아 가지 못한다."""
    body = client.post(ENDPOINT, json={"answers": [None, None, None]}).json()
    assert body["correct_count"] == 0
    for answer in body["answers"]:
        assert answer["answer_index"] is None
        assert not answer["explanation"]


def test_answered_questions_get_answer_and_explanation(client):
    """답을 냈으면 결과 화면이 정답과 해설을 보여준다."""
    body = client.post(ENDPOINT, json={"answers": _answers(3)}).json()
    for answer in body["answers"]:
        assert answer["answer_index"] is not None
        assert answer["explanation"]


def test_empty_answers_rejected(client):
    assert client.post(ENDPOINT, json={"answers": []}).status_code == 422


# ── 모델 비의존 ──────────────────────────────────────────────────────────
def test_works_when_llm_is_dead(client, monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "fail")
    get_settings.cache_clear()
    reset_services()
    assert client.get(ENDPOINT).status_code == 200
    assert client.post(ENDPOINT, json={"answers": _answers(3)}).status_code == 200


# ── 채점 함수 단위 ───────────────────────────────────────────────────────
def test_level_for_handles_other_bank_sizes():
    assert level_for(0, 0) == LEVEL_BEGINNER  # 문제은행이 비어도 죽지 않는다
    assert level_for(4, 4) == LEVEL_FAMILIAR
    assert level_for(3, 4) == LEVEL_NOVICE
    assert level_for(1, 4) == LEVEL_BEGINNER
    assert level_for(9, 3) == LEVEL_FAMILIAR  # 맞힌 수가 문항 수를 넘어도 경계 안으로


def test_grade_reports_each_answer_with_explanation():
    questions = load_diagnostic()
    result = grade(questions, [q["answer_index"] for q in questions])
    assert result.correct_count == len(questions)
    assert [a.term_id for a in result.answers] == [q["term_id"] for q in questions]
    assert all(a.explanation for a in result.answers)


BROKEN_QUESTION = {
    "id": "x",
    "term_id": "walk",
    "question": "정답이 빠진 문항",
    "choices": ["a", "b"],
    "explanation": "e",
}


@pytest.mark.parametrize("pick", [-1, 0, 1, None])
def test_question_without_answer_index_is_never_correct(pick):
    """시드에서 정답이 빠지면 조용히 오채점되지 않고 전부 틀림으로 센다."""
    result = grade([BROKEN_QUESTION], [pick])
    assert result.correct_count == 0
    assert result.answers[0].correct is False
    assert result.answers[0].answer_index is None


def test_boolean_answer_index_is_not_read_as_one():
    """True == 1이라 걸러내지 않으면 2번 보기가 정답이 된다."""
    result = grade([{**BROKEN_QUESTION, "answer_index": True}], [1])
    assert result.correct_count == 0
