"""규칙 전체 목록(S5). 용어 사전과 합치지 않는다 — 사전 화면의 버튼(카테고리)별로 나눠 준다."""

from collections.abc import Sequence
from typing import Any, Optional

from app.domain.models import (
    CATEGORY_BASIC,
    CATEGORY_CULTURE,
    CATEGORY_LABELS,
    CATEGORY_PITCHING,
    CATEGORY_TACTICS,
)

CATEGORY_UMPIRE = "umpire_calls"
RULE_CATEGORY_LABELS = {**CATEGORY_LABELS, CATEGORY_UMPIRE: "심판 콜"}
# 규칙 목록의 장 → 버튼. 보크는 와이어프레임대로 심판 콜이다.
CHAPTER_CATEGORIES = {
    "A": CATEGORY_BASIC, "B": CATEGORY_BASIC, "C": CATEGORY_BASIC, "E": CATEGORY_BASIC,
    "F": CATEGORY_BASIC, "J": CATEGORY_BASIC, "L": CATEGORY_BASIC,
    "D": CATEGORY_PITCHING,
    "H": CATEGORY_TACTICS, "M": CATEGORY_TACTICS,
    "D-2": CATEGORY_UMPIRE, "G": CATEGORY_UMPIRE, "I": CATEGORY_UMPIRE, "K": CATEGORY_UMPIRE,
    "N": CATEGORY_CULTURE,
}
_BY_LABEL = {label.replace(" ", ""): key for key, label in RULE_CATEGORY_LABELS.items()}
_BY_LABEL["구종"] = CATEGORY_PITCHING  # 와이어프레임의 버튼 이름


def rule_category(rule: dict[str, Any]) -> str:
    return CHAPTER_CATEGORIES[rule["chapter"].split(".")[0]]


class RuleService:
    def __init__(self, rules: Sequence[dict[str, Any]]) -> None:
        self._rules = rules

    def search(
        self, category: Optional[str] = None, descending: bool = False
    ) -> list[dict[str, Any]]:
        """카테고리(한글 라벨 또는 내부 키)로 거르고 이름순. 없는 카테고리는 빈 목록."""
        wanted = None
        if category:
            wanted = _BY_LABEL.get(category.replace(" ", ""), category)
        found = [r for r in self._rules if wanted is None or rule_category(r) == wanted]
        return sorted(found, key=lambda r: r["name"], reverse=descending)
