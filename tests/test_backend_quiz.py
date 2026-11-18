"""백엔드 연동 학습 퀴즈 — 고정 번호·정답 분리·용어 사전 전부 포함."""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.glossary.seed import load_glossary
from app.adapters.quiz.bank import load_quiz_bank, sample_questions
from app.api.deps import reset_services
from app.config import get_settings
from app.main import app

BASE = "/api/backend/quiz"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    reset_services()
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def test_bank_covers_every_glossary_term_with_unique_ids():
    bank = load_quiz_bank()
    ids = [q["questionId"] for q in bank]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids) and ids[0] == 1001  # 백엔드 임시 문항(1번)과 안 겹치게
    assert set(load_glossary()) <= {q["termId"] for q in bank}


def test_bank_questions_have_four_distinct_options_and_valid_answer():
    for q in load_quiz_bank():
        assert len(q["options"]) == 4 and len(set(q["options"])) == 4, q["questionId"]
        assert 0 <= q["answerIndex"] < 4
        assert q["text"].strip() and q["explanation"].strip()


def test_check_script_passes_on_committed_bank():
    """용어를 사전에 더하고 문제은행을 다시 안 만들면 여기서 실패한다."""
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_quiz_bank.py"), "--check"],
        capture_output=True, cwd=ROOT,
    )
    assert run.returncode == 0, run.stdout


def test_same_seed_same_questions_and_different_seed_differs():
    bank = load_quiz_bank()
    a = [q["questionId"] for q in sample_questions(bank, 5, "7-2026-11-18")]
    assert a == [q["questionId"] for q in sample_questions(bank, 5, "7-2026-11-18")]
    assert a != [q["questionId"] for q in sample_questions(bank, 5, "7-2026-11-19")]
    assert len(set(a)) == 5


def test_growing_bank_keeps_most_of_todays_questions():
    """용어가 늘어 문제은행이 커져도 그날 문항이 통째로 바뀌지 않는다."""
    bank = load_quiz_bank()
    grown = bank + [dict(bank[0], questionId=9000 + n, termId=f"new{n}") for n in range(3)]
    for day in range(20):
        seed = f"7-2026-11-{day + 1:02d}"
        before = {q["questionId"] for q in sample_questions(bank, 5, seed)}
        after = {q["questionId"] for q in sample_questions(grown, 5, seed)}
        new_in = {i for i in after if i >= 9000}
        assert before - after == set() or len(before - after) == len(new_in)


def test_questions_have_backend_dto_shape_without_answers(client):
    res = client.get(f"{BASE}/questions", params={"count": 3, "seed": "1-2026-11-18"})
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 3
    for q in body:
        assert set(q) == {"questionId", "termId", "text", "options"}
        assert [o["optionId"] for o in q["options"]] == [1, 2, 3, 4]


def test_detail_has_answer_matching_bank(client):
    bank = {q["questionId"]: q for q in load_quiz_bank()}
    q = client.get(f"{BASE}/questions", params={"count": 1, "seed": "x"}).json()[0]
    detail = client.get(f"{BASE}/questions/{q['questionId']}").json()
    entry = bank[q["questionId"]]
    assert detail["correctOptionId"] == entry["answerIndex"] + 1
    assert detail["options"] == q["options"] and detail["text"] == q["text"]
    assert detail["explanation"] == entry["explanation"]


def test_unknown_question_is_404_and_bad_count_is_422(client):
    assert client.get(f"{BASE}/questions/99999").status_code == 404
    assert client.get(f"{BASE}/questions/0").status_code == 422
    assert client.get(f"{BASE}/questions", params={"count": 0}).status_code == 422
    assert client.get(f"{BASE}/questions", params={"count": 21}).status_code == 422
