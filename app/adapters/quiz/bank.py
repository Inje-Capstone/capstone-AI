"""학습 퀴즈 고정 문제은행(data/quiz_bank.json) — 백엔드가 풀이 기록을 남기는 퀴즈.

백엔드는 `questionId`(정수)로 풀이 기록을 저장하고, 나중에 그 번호로 문제·정답·해설을 다시
불러 보여 준다. 그래서 문항은 요청마다 만들지 않고 **한 번 만든 파일을 그대로** 쓴다 —
용어 사전이 늘어도 예전 번호의 문제·보기 순서가 바뀌지 않는다.
새 용어는 `scripts/build_quiz_bank.py`가 끝 번호 뒤에 덧붙인다(기존 번호는 그대로).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

QUIZ_BANK_PATH = Path(__file__).resolve().parents[3] / "data" / "quiz_bank.json"


def load_quiz_bank(path: Path = QUIZ_BANK_PATH) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("questions", [])


def find_question(bank: list[dict[str, Any]], question_id: int) -> Optional[dict[str, Any]]:
    return next((q for q in bank if q["questionId"] == question_id), None)


def _rank(seed: str, question_id: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def sample_questions(
    bank: list[dict[str, Any]], count: int, seed: Optional[str]
) -> list[dict[str, Any]]:
    """같은 `seed`(예: 사용자ID+날짜)면 같은 문항 — 그날 다시 열어도 문제가 안 바뀐다.

    문항마다 (seed, 번호) 해시로 순위를 매겨 앞에서 고른다. 문제은행에 문항이 늘어도
    이미 뽑힌 문항은 새 문항이 그보다 앞 순위일 때만 밀린다(통째로 다시 섞이지 않는다).
    """
    key = seed or "rookie"
    picked = sorted(bank, key=lambda q: _rank(key, q["questionId"]))[:count]
    return sorted(picked, key=lambda q: q["questionId"])
