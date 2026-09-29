"""온보딩 수준 진단 (user-flow B①). 결과 저장·재응시 이력은 백엔드 소유다."""

from fastapi import APIRouter

from app.api.deps import get_diagnostic_questions
from app.api.schemas import (
    DiagnosticAnswersIn,
    DiagnosticOut,
    DiagnosticQuestionOut,
    DiagnosticResultOut,
)
from app.domain.diagnostic import grade

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])


@router.get(
    "/diagnostic",
    response_model=DiagnosticOut,
    summary="수준 진단 문항 3개 (정답 없음 · LLM 호출 없음)",
)
def diagnostic_questions():
    """고정 문제은행을 그대로 내보낸다.

    **정답과 해설은 담지 않는다** — 클라이언트 번들에 정답이 실리면 진단이 무의미해진다.
    채점은 같은 경로의 POST가 서버에서 한다.
    """
    questions = get_diagnostic_questions()
    return DiagnosticOut(
        total=len(questions),
        questions=[DiagnosticQuestionOut.of(q) for q in questions],
    )


@router.post(
    "/diagnostic",
    response_model=DiagnosticResultOut,
    summary="수준 진단 채점 → 입문 · 초보 · 익숙",
)
def diagnostic_result(payload: DiagnosticAnswersIn):
    """맞힌 수로 수준을 정하고, 결과 문구·문항별 해설·판정 근거를 함께 돌려준다.

    돌려준 `level_label`을 이후 요청의 `level`(입문 | 초보 | 익숙)에 그대로 넘기면 된다.
    """
    return DiagnosticResultOut.of(grade(get_diagnostic_questions(), payload.answers))
