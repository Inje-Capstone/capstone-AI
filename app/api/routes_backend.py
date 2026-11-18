"""백엔드(Spring Boot) 연동용 온보딩 계약 — 진단 퀴즈·채점·개인화 가중치.

백엔드 DTO(`DiagnosisQuestionResponse`·`DiagnosisSubmitRequest`·`DiagnosisResultResponse`)와
**필드 이름·번호 체계를 그대로 맞춘다**: camelCase, 문항·보기 번호는 1부터, 수준은
`LearningLevel` enum 이름(INTRODUCTORY | BEGINNER | FAMILIAR). 백엔드는 매핑 코드 없이
받아서 `User.learningLevel`·`diagnosisScore`에 저장하면 된다.

저장·재응시 이력·포인트는 백엔드 소유다. 여기선 문항을 주고 채점·가중치만 계산한다.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from app.adapters.quiz.bank import find_question, sample_questions
from app.api.deps import get_diagnostic_questions, get_quiz_bank
from app.api.schemas import (
    BackendDiagnosisIn,
    BackendDiagnosisOut,
    BackendOptionOut,
    BackendQuestionOut,
    BackendQuizDetailOut,
    BackendQuizQuestionOut,
    BackendWeightsOut,
)
from app.domain.diagnostic import grade
from app.domain.profile import BACKEND_LEVEL_NAMES, weights_for

router = APIRouter(prefix="/api/backend/onboarding", tags=["backend"])
quiz_router = APIRouter(prefix="/api/backend/quiz", tags=["backend"])


@router.get(
    "/questions",
    response_model=list[BackendQuestionOut],
    summary="진단 퀴즈 문항 (백엔드 DTO 모양 · 정답 없음)",
)
def backend_questions():
    """`questionId`·`optionId`는 1부터. 정답·해설은 담지 않는다 — 채점은 POST가 서버에서 한다."""
    return [
        BackendQuestionOut(
            questionId=n + 1,
            termId=q.get("term_id", ""),
            question=q["question"],
            options=[
                BackendOptionOut(optionId=i + 1, text=text)
                for i, text in enumerate(q.get("choices", []))
            ],
        )
        for n, q in enumerate(get_diagnostic_questions())
    ]


@router.post(
    "/diagnosis",
    response_model=BackendDiagnosisOut,
    summary="진단 채점 → learningLevel + 개인화 가중치",
)
def backend_diagnosis(payload: BackendDiagnosisIn):
    """백엔드 `DiagnosisSubmitRequest`를 그대로 받는다.

    빠진 문항·모르는 번호는 틀림으로 센다(에러 없음 — 온보딩은 막히면 안 된다).
    관심 카테고리를 같이 보내면 가중치에 반영하고, 없으면 전부 관심으로 본다.
    """
    questions = get_diagnostic_questions()
    chosen: list[Optional[int]] = [None] * len(questions)
    for a in payload.answers:
        if 1 <= a.questionId <= len(questions):
            chosen[a.questionId - 1] = a.optionId - 1
    diagnosis = grade(questions, chosen)
    return BackendDiagnosisOut(
        correctCount=diagnosis.correct_count,
        totalCount=diagnosis.total,
        learningLevel=BACKEND_LEVEL_NAMES[diagnosis.level],
        levelLabel=diagnosis.level_label,
        message=diagnosis.message,
        results=[
            {
                "questionId": n + 1,
                "termId": g.term_id,
                "chosenOptionId": None if g.chosen_index is None else g.chosen_index + 1,
                "answerOptionId": None if g.answer_index is None else g.answer_index + 1,
                "correct": g.correct,
                "explanation": g.explanation,
            }
            for n, g in enumerate(diagnosis.answers)
        ],
        weights=weights_for(diagnosis.level, payload.categories),
    )


@router.get(
    "/weights",
    response_model=BackendWeightsOut,
    summary="learningLevel(+관심 카테고리)별 개인화 가중치",
)
def backend_weights(
    learningLevel: str = Query(
        "INTRODUCTORY", description="INTRODUCTORY | BEGINNER | FAMILIAR (입문 | 초보 | 익숙도 허용)"
    ),
    categories: Optional[list[str]] = Query(
        None, description="관심 카테고리 — 키(basic_rules…) 또는 한글 라벨. 없으면 전부"
    ),
):
    """저장된 `User.learningLevel`이 바뀌었을 때(재진단·설정 변경) 다시 받아 가는 용도."""
    return weights_for(learningLevel, categories)


def _options(q: dict) -> list[BackendOptionOut]:
    return [BackendOptionOut(optionId=i + 1, text=text) for i, text in enumerate(q["options"])]


@quiz_router.get(
    "/questions",
    response_model=list[BackendQuizQuestionOut],
    summary="학습 퀴즈 문항 (백엔드 QuizQuestionResponse 모양 · 정답 없음)",
)
def backend_quiz_questions(
    count: int = Query(5, ge=1, le=20, description="문항 수"),
    seed: Optional[str] = Query(
        None, max_length=100, description="같은 값이면 같은 문항 (예: 사용자ID+날짜)"
    ),
):
    """고정 문제은행에서 `count`개. 번호는 바뀌지 않아 풀이 기록 키로 쓸 수 있다."""
    return [
        BackendQuizQuestionOut(
            questionId=q["questionId"], termId=q["termId"], text=q["text"], options=_options(q)
        )
        for q in sample_questions(get_quiz_bank(), count, seed)
    ]


@quiz_router.get(
    "/questions/{question_id}",
    response_model=BackendQuizDetailOut,
    summary="학습 퀴즈 한 문항 + 정답·해설 (백엔드 채점·풀이 기록 상세용)",
)
def backend_quiz_question(question_id: int = Path(..., ge=1)):
    """정답이 들어 있다 — 백엔드가 채점에 쓰고, 프론트에 그대로 중계하지 않는다."""
    q = find_question(get_quiz_bank(), question_id)
    if q is None:
        raise HTTPException(status_code=404, detail=f"없는 문항입니다: {question_id}")
    return BackendQuizDetailOut(
        questionId=q["questionId"],
        termId=q["termId"],
        text=q["text"],
        options=_options(q),
        correctOptionId=q["answerIndex"] + 1,
        explanation=q["explanation"],
    )
