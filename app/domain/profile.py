"""온보딩 답변 → 노출 판단. 순수 로직(I/O 없음).

**분기 대신 파라미터** (core-belief 3). 난이도 3 × 카테고리 4 = 12가지 조합에
if문을 두지 않는다. 규칙마다 점수를 계산하고 임계값 하나로 거른다. 조합이 늘어도
코드 경로는 그대로 하나다.

점수 = 규칙 고유 중요도 × 카테고리 관심도 × 난이도 배율
"""

from collections.abc import Iterable, Sequence
from typing import Optional

from app.domain.models import (
    ALL_CATEGORIES,
    CATEGORY_BASIC,
    CATEGORY_CULTURE,
    CATEGORY_LABELS,
    CATEGORY_PITCHING,
    CATEGORY_TACTICS,
    LEVEL_BEGINNER,
    LEVEL_FAMILIAR,
    LEVEL_LABELS,
    LEVEL_NOVICE,
    ExplainProfile,
    Situation,
)

# 관심 카테고리로 고르지 않았다고 완전히 죽이지는 않는다 — 정말 중요한 상황(보크 등)은
# 여전히 뚫고 올라올 수 있게 하는 바닥값.
UNSELECTED_CATEGORY_WEIGHT = 0.35

# 난이도별 카테고리 배율. 입문자에겐 기본 룰을 키우고, 룰을 아는 사람에겐 줄인다.
LEVEL_CATEGORY_MULTIPLIER = {
    LEVEL_BEGINNER: {
        CATEGORY_BASIC: 1.3,
        CATEGORY_PITCHING: 0.9,
        CATEGORY_TACTICS: 0.7,
        CATEGORY_CULTURE: 1.0,
    },
    LEVEL_NOVICE: {
        CATEGORY_BASIC: 1.0,
        CATEGORY_PITCHING: 1.1,
        CATEGORY_TACTICS: 0.9,
        CATEGORY_CULTURE: 1.0,
    },
    LEVEL_FAMILIAR: {
        CATEGORY_BASIC: 0.6,
        CATEGORY_PITCHING: 1.2,
        CATEGORY_TACTICS: 1.3,
        CATEGORY_CULTURE: 1.0,
    },
}

# 온보딩 화면(S2)의 한글 선택지 → 내부 키
LEVEL_BY_LABEL = {label: level for level, label in LEVEL_LABELS.items()}
CATEGORY_BY_LABEL = {label: key for key, label in CATEGORY_LABELS.items()}
# 표기 흔들림 흡수 (칩 라벨의 공백 유무 등)
CATEGORY_BY_LABEL.update(
    {label.replace(" ", ""): key for key, label in CATEGORY_LABELS.items()}
)


def profile_from_onboarding(
    level: Optional[str] = None,
    categories: Optional[Iterable[str]] = None,
    threshold: float = 0.5,
) -> ExplainProfile:
    """온보딩 답변(한글 라벨 또는 내부 키)을 가중치 객체로 바꾼다.

    모르는 값은 조용히 버리지 않고 기본값으로 되돌린다 — 개인화가 안 되는 건 괜찮지만
    카드가 통째로 사라지는 건 안 된다(바닥 품질, core-belief 1).
    """
    level_index = LEVEL_BY_LABEL.get(level or "", LEVEL_BEGINNER)
    if isinstance(level, int):  # 숫자로 들어오는 경로도 허용
        level_index = level

    keys = set()
    for raw in categories or ():
        key = CATEGORY_BY_LABEL.get(raw, raw if raw in ALL_CATEGORIES else None)
        if key:
            keys.add(key)
    if not keys:
        keys = set(ALL_CATEGORIES)  # 아무것도 못 알아들으면 전부 열어둔다

    return ExplainProfile(
        level=level_index, categories=frozenset(keys), threshold=threshold
    )


def score(situation: Situation, profile: ExplainProfile) -> tuple[float, list[str]]:
    """상황 하나의 노출 점수와 그 계산 근거를 함께 돌려준다.

    근거를 못 대면 설계가 틀린 것이다 (core-belief 4) — 그래서 점수만이 아니라
    어떤 항이 얼마였는지를 문자열로 남긴다. 디버깅과 발표 설명에 같이 쓴다.
    """
    selected = situation.category in profile.categories
    category_weight = 1.0 if selected else UNSELECTED_CATEGORY_WEIGHT
    level_mult = LEVEL_CATEGORY_MULTIPLIER.get(profile.level, {}).get(
        situation.category, 1.0
    )
    value = situation.priority * category_weight * level_mult

    category_label = CATEGORY_LABELS.get(situation.category, situation.category)
    reasons = [
        f"노출 점수 {value:.2f} = 중요도 {situation.priority:.2f}"
        f" × 관심도 {category_weight:.2f}({category_label}"
        f"{'선택함' if selected else '선택 안 함'})"
        f" × 난이도배율 {level_mult:.2f}({profile.level_label})",
        f"임계값 {profile.threshold:.2f} "
        f"{'이상 → 노출' if value >= profile.threshold else '미만 → 생략'}",
    ]
    return value, reasons


def select(
    situations: Sequence[Situation], profile: ExplainProfile
) -> list[Situation]:
    """프로필에 맞는 상황만 남기고, 판단 근거를 `reasons`에 덧붙여 돌려준다."""
    kept: list[Situation] = []
    for situation in situations:
        value, reasons = score(situation, profile)
        if value < profile.threshold:
            continue
        enriched = situation.model_copy(deep=True)
        enriched.reasons = list(enriched.reasons) + reasons
        kept.append(enriched)
    return kept
