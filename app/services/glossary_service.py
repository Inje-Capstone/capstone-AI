"""용어 사전(S5). 시드 조회·검색만 하고 생성은 하지 않는다 — 사전은 정답지여야 한다."""

from collections.abc import Iterable
from typing import Any, Optional

from app.domain.models import CATEGORY_LABELS, LEVEL_LABELS
from app.domain.profile import CATEGORY_BY_LABEL

# 난이도 인덱스 → 시드 필드. mock LLM과 같은 매핑이다.
LEVEL_FIELDS = {0: "easy", 1: "standard", 2: "deep"}


class GlossaryService:
    def __init__(self, terms: dict[str, dict[str, Any]]) -> None:
        self._terms = terms

    def get(self, term_id: str) -> Optional[dict[str, Any]]:
        entry = self._terms.get(term_id)
        return None if entry is None else {"id": term_id, **entry}

    def search(
        self, q: Optional[str] = None, categories: Optional[Iterable[str]] = None
    ) -> list[dict[str, Any]]:
        """이름·별칭·키 부분일치 + 카테고리(한글 라벨 또는 내부 키) 필터. 이름순."""
        wanted = {CATEGORY_BY_LABEL.get(c, c) for c in categories or ()}
        needle = (q or "").strip().lower().replace(" ", "")
        found = []
        for term_id, entry in self._terms.items():
            if wanted and entry.get("category") not in wanted:
                continue
            if needle:
                keys = [term_id, entry.get("name", ""), *entry.get("aliases", [])]
                if not any(needle in k.lower().replace(" ", "") for k in keys):
                    continue
            found.append({"id": term_id, **entry})
        return sorted(found, key=lambda e: e.get("name", ""))

    def related(self, term_id: str) -> list[dict[str, Any]]:
        """존재하지 않는 관련 용어 키는 조용히 뺀다 — 딥링크가 404로 끝나면 안 된다."""
        entry = self._terms.get(term_id) or {}
        return [self.get(r) for r in entry.get("related", []) if r in self._terms]


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)


def level_texts(entry: dict[str, Any]) -> dict[str, str]:
    """{"입문": ..., "초보": ..., "익숙": ...} — S5 난이도 토글용."""
    return {
        LEVEL_LABELS[level]: entry.get(field, "")
        for level, field in LEVEL_FIELDS.items()
    }
