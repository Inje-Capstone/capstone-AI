"""규칙 전체 목록(S5) 엔드포인트. 용어 사전과 따로 둔다."""

from typing import Literal, Optional

from fastapi import APIRouter, Query

from app.api.deps import get_rule_service
from app.api.schemas import RuleOut
from app.services.rule_service import RULE_CATEGORY_LABELS, rule_category

router = APIRouter(prefix="/api/rules", tags=["rules"])


@router.get("", response_model=list[RuleOut], summary="규칙 목록 — 버튼별, 이름순 (S5)")
def list_rules(
    category: Optional[str] = Query(
        default=None, description="기본 룰 | 구종 | 전술 · 기록 | 심판 콜 | 응원 문화 (생략: 전체)"
    ),
    sort: Literal["name", "-name"] = Query(
        default="name", description="name: 가나다 오름차순(기본) | -name: 가나다 내림차순"
    ),
):
    rows = get_rule_service().search(category, descending=sort == "-name")
    return [RuleOut.of(r, RULE_CATEGORY_LABELS[rule_category(r)]) for r in rows]
