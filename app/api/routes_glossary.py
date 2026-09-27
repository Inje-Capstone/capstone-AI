"""용어 사전(S5) 엔드포인트. 카드의 `term_id`가 여기로 딥링크된다."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from app.api.deps import get_glossary_service
from app.api.schemas import TermOut, TermSummaryOut
from app.domain.profile import profile_from_onboarding
from app.services.glossary_service import LEVEL_FIELDS, category_label, level_texts

router = APIRouter(prefix="/api/glossary", tags=["glossary"])


def _summary(entry: dict) -> TermSummaryOut:
    return TermSummaryOut.of(entry, category_label(entry.get("category", "")))


@router.get("", response_model=list[TermSummaryOut], summary="용어 목록·검색 (S5)")
def list_terms(
    q: Optional[str] = Query(default=None, max_length=50, description="이름·별칭 부분일치"),
    category: Optional[list[str]] = Query(
        default=None, description="기본 룰 | 구종 · 투구 | 전술 · 기록 | 응원 문화 (복수)"
    ),
):
    return [_summary(e) for e in get_glossary_service().search(q, category)]


@router.get("/{term_id}", response_model=TermOut, summary="용어 상세 (S5, 카드 딥링크)")
def get_term(
    term_id: str = Path(...),
    level: Optional[str] = Query(default=None, description="입문 | 초보 | 익숙"),
):
    service = get_glossary_service()
    entry = service.get(term_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"없는 용어입니다: {term_id}")
    level_index = profile_from_onboarding(level=level).level
    base = _summary(entry)
    return TermOut(
        **base.model_dump(),
        level=level_index,
        body=entry.get(LEVEL_FIELDS.get(level_index, "easy"), ""),
        levels=level_texts(entry),
        related=[_summary(r) for r in service.related(term_id)],
    )
