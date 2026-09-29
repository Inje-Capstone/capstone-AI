"""온보딩 수준 진단 채점. 순수 로직(I/O 없음).

고정 문제은행이라 LLM을 부르지 않는다(user-flow B①). 맞힌 수 하나로 설명 난이도가
정해지고, 그 난이도가 이후 카드·챗봇·퀴즈의 기준(`ExplainProfile.level`)이 된다.
"""

from collections.abc import Sequence
from typing import Any, Optional

from pydantic import BaseModel

from app.domain.models import (
    LEVEL_BEGINNER,
    LEVEL_FAMILIAR,
    LEVEL_LABELS,
    LEVEL_NOVICE,
)

# 스펙이 정한 문항 수. 이 수일 때는 아래 표를 그대로 쓴다 (user-flow.md B①-결과).
SPEC_TOTAL = 3

# 문제은행에 정답이 없을 때의 자리값. 어떤 선택과도 같아서는 안 된다 —
# 사용자가 보낸 음수 인덱스와 우연히 맞아 오채점되는 걸 막는다.
NO_ANSWER = -1

# 맞힌 수 → 수준. 경계를 코드 여러 곳에 흩지 않고 여기 한 곳에 둔다.
LEVEL_BY_CORRECT = {
    0: LEVEL_BEGINNER,
    1: LEVEL_BEGINNER,
    2: LEVEL_NOVICE,
    3: LEVEL_FAMILIAR,
}

# 문항 수가 스펙과 달라졌을 때 쓰는 정답률 경계 — 온보딩이 통째로 막히는 게 최악이다.
NOVICE_RATIO = 2 / 3
FAMILIAR_RATIO = 1.0

# 진단 결과 화면 문구 (user-flow.md §9).
RESULT_MESSAGES = {
    LEVEL_BEGINNER: "야구가 처음이시군요! 제일 쉬운 말로, 용어부터 천천히 설명할게요.",
    LEVEL_NOVICE: "기본은 아시네요. 헷갈리기 쉬운 판정 위주로 짚어 드릴게요.",
    LEVEL_FAMILIAR: "룰은 익숙하시네요. 구종과 작전까지 한 걸음 더 들어가 볼게요.",
}


class GradedAnswer(BaseModel):
    """문항 1개의 채점 결과. 결과 화면에서 해설과 용어 사전 링크에 쓴다.

    `answer_index`·`explanation`은 **답을 낸 문항에만** 채워진다 — 빈 답안 한 번으로
    정답표를 받아 가는 경로를 만들지 않기 위해서다.
    """

    question_id: str
    term_id: str
    chosen_index: Optional[int]
    answer_index: Optional[int] = None
    correct: bool
    explanation: Optional[str] = None


class Diagnosis(BaseModel):
    correct_count: int
    total: int
    level: int
    level_label: str
    message: str
    answers: list[GradedAnswer]
    reasons: list[str]


def level_for(correct: int, total: int) -> int:
    """맞힌 수 → 설명 난이도.

    문제은행이 스펙대로 3문항이면 표(0~1 입문 · 2 초보 · 3 익숙)를 그대로 쓴다.
    문항을 늘리거나 줄여도 같은 비율로 동작하게 해 둔다.
    """
    if total <= 0:
        return LEVEL_BEGINNER
    bounded = max(0, min(correct, total))
    if total == SPEC_TOTAL:
        return LEVEL_BY_CORRECT[bounded]
    ratio = bounded / total
    if ratio >= FAMILIAR_RATIO:
        return LEVEL_FAMILIAR
    if ratio >= NOVICE_RATIO:
        return LEVEL_NOVICE
    return LEVEL_BEGINNER


def _answer_index(question: dict[str, Any]) -> int:
    """문제은행의 정답 인덱스. 없거나 정수가 아니거나 음수면 `NO_ANSWER`.

    `True`는 파이썬에서 `1`과 같으므로 따로 걸러 낸다 — 시드가 망가졌을 때
    2번 보기가 정답이 되는 식으로 조용히 틀리는 걸 막는다.
    """
    raw = question.get("answer_index")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return NO_ANSWER
    return raw


def grade(
    questions: Sequence[dict[str, Any]], chosen: Sequence[Optional[int]]
) -> Diagnosis:
    """문항 순서대로 고른 보기 인덱스를 채점한다.

    안 보낸 문항·범위를 벗어난 인덱스는 **틀림으로 세고 에러를 내지 않는다** —
    온보딩은 건너뛸 수 없는 관문이라(user-flow A4) 여기서 막히면 가입이 끝난다.
    답을 내지 않은 문항에는 정답·해설을 붙이지 않는다.
    """
    graded: list[GradedAnswer] = []
    for n, question in enumerate(questions):
        pick = chosen[n] if n < len(chosen) else None
        answer = _answer_index(question)
        answered = pick is not None
        graded.append(
            GradedAnswer(
                question_id=str(question.get("id", f"q{n + 1}")),
                term_id=str(question.get("term_id", "")),
                chosen_index=pick,
                answer_index=answer if answered and answer != NO_ANSWER else None,
                correct=answered and pick >= 0 and answer != NO_ANSWER and pick == answer,
                explanation=str(question.get("explanation", "")) if answered else None,
            )
        )

    correct = sum(1 for g in graded if g.correct)
    total = len(graded)
    level = level_for(correct, total)
    label = LEVEL_LABELS.get(level, LEVEL_LABELS[LEVEL_BEGINNER])
    reasons = [f"{total}문항 중 {correct}개 정답 → {label}"]
    if total == SPEC_TOTAL:
        reasons.append("기준: 0~1개 입문 · 2개 초보 · 3개 익숙 (고정 문제은행, AI 호출 없음)")
    else:
        reasons.append(
            f"기준: 정답률 {NOVICE_RATIO:.0%} 이상 초보 · 전부 맞히면 익숙"
            f" (문항 수가 스펙 {SPEC_TOTAL}개와 달라 비율로 판정)"
        )

    return Diagnosis(
        correct_count=correct,
        total=total,
        level=level,
        level_label=label,
        message=RESULT_MESSAGES.get(level, RESULT_MESSAGES[LEVEL_BEGINNER]),
        answers=graded,
        reasons=reasons,
    )
