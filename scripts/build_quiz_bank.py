"""용어 사전 → 학습 퀴즈 고정 문제은행(data/quiz_bank.json).

이미 있는 번호는 건드리지 않는다 — 백엔드 풀이 기록이 그 번호의 문제·보기 순서를 가리킨다.
용어 사전에 새로 생긴 용어만 끝 번호 뒤에 덧붙인다. 용어가 사전에서 빠져도 문항은 남긴다
(예전 풀이 기록을 보여 줘야 하므로).

    python scripts/build_quiz_bank.py          # 새 용어만 덧붙임
    python scripts/build_quiz_bank.py --check  # 사전에 문항 없는 용어가 있으면 실패(테스트용)
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.adapters.glossary.seed import load_glossary  # noqa: E402
from app.adapters.llm.mock import MockLLMClient  # noqa: E402
from app.adapters.quiz.bank import QUIZ_BANK_PATH  # noqa: E402
from app.services.quiz_service import QuizService, _seed_int  # noqa: E402

# 1000번 대부터 — 백엔드 임시 문항(1번)의 예전 풀이 기록과 번호가 겹치지 않게.
FIRST_ID = 1001

NOTE = (
    "학습 퀴즈 고정 문제은행 — scripts/build_quiz_bank.py가 만든다. "
    "questionId는 백엔드 풀이 기록의 키라 바꾸거나 지우지 않는다"
    "(1001부터 — 백엔드 임시 문항 번호와 안 겹치게). "
    "문항은 입문 설명으로, 해설은 한 단계 위 설명으로 만든다."
)


def build_entry(service: QuizService, term_id: str, question_id: int) -> dict:
    rng = random.Random(_seed_int(f"bank:{term_id}"))
    item = service._glossary_item(term_id, 0, question_id, rng)
    return {
        "questionId": question_id,
        "termId": term_id,
        "text": item.question,
        "options": item.choices,
        "answerIndex": item.answer_index,
        "explanation": item.explanation,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="쓰지 않고 빠진 용어만 확인")
    args = ap.parse_args()

    terms = load_glossary()
    raw = json.loads(QUIZ_BANK_PATH.read_text(encoding="utf-8")) if QUIZ_BANK_PATH.exists() else {}
    questions = raw.get("questions", [])
    have = {q["termId"] for q in questions}
    missing = [t for t in terms if t not in have]
    if args.check:
        if missing:
            print("문항 없는 용어:", ", ".join(missing))
            return 1
        print(f"ok: {len(questions)}문항")
        return 0

    service = QuizService(terms, MockLLMClient())
    next_id = max((q["questionId"] for q in questions), default=FIRST_ID - 1) + 1
    for term_id in missing:
        questions.append(build_entry(service, term_id, next_id))
        next_id += 1
    out = {"_note": NOTE, "questions": questions}
    text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    QUIZ_BANK_PATH.write_text(text, encoding="utf-8")
    print(f"{len(missing)}문항 추가 · 모두 {len(questions)}문항")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
