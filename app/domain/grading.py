"""영상 판정 채점 — 같은 경기 문자중계를 정답지로. 순수 로직(I/O 없음).

서비스 화면은 영상 판정만 쓰고, 문자중계는 여기서 "몇 개를 맞혔나"를 재는 데만 쓴다.
단위는 카드가 뜨는 상황(Situation.rule_id)이다 — 사용자가 실제로 보는 것이 채점 대상.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.models import Situation

# 영상으로 판정하는 규칙. decisive_offspeed(구종)는 영상 판정 범위 밖이라 따로 표시한다.
GRADED_RULES = (
    "balk", "steal", "walk", "hit_by_pitch", "dropped_third_strike", "infield_fly",
    "sac_fly", "homerun", "double_play", "pitching_change", "bases_loaded",
)
UNSUPPORTED_RULES = ("decisive_offspeed",)


@dataclass
class RuleScore:
    rule_id: str
    expected: int  # 문자중계에서 나온 수
    found: int  # 영상 판정에서 나온 수
    matched: int

    @property
    def recall(self) -> float:
        return self.matched / self.expected if self.expected else 1.0

    @property
    def precision(self) -> float:
        return self.matched / self.found if self.found else 1.0


def _match(truth: list[float], guess: list[float], tol: float) -> int:
    """시각 기준 탐욕 매칭. 하나의 정답은 하나의 판정과만 짝짓는다."""
    used = [False] * len(guess)
    matched = 0
    for t in sorted(truth):
        best, best_d = None, tol + 1
        for i, g in enumerate(guess):
            d = abs(g - t)
            if not used[i] and d <= tol and d < best_d:
                best, best_d = i, d
        if best is not None:
            used[best] = True
            matched += 1
    return matched


def grade(
    truth: Sequence[Situation],
    video: Sequence[Situation],
    offset: float = 0.0,
    tolerance: float = 30.0,
    rules: Sequence[str] = GRADED_RULES,
) -> list[RuleScore]:
    """규칙별 점수. `offset`은 영상 t − 중계 t (영상 싱크 추정값)."""
    scores = []
    for rule in rules:
        t_truth = [s.t + offset for s in truth if s.rule_id == rule]
        t_video = [float(s.t) for s in video if s.rule_id == rule]
        scores.append(RuleScore(rule, len(t_truth), len(t_video),
                                _match(t_truth, t_video, tolerance)))
    return scores


def summary(scores: Sequence[RuleScore]) -> dict:
    expected = sum(s.expected for s in scores)
    found = sum(s.found for s in scores)
    matched = sum(s.matched for s in scores)
    return {
        "expected": expected, "found": found, "matched": matched,
        "recall": round(matched / expected, 3) if expected else 1.0,
        "precision": round(matched / found, 3) if found else 1.0,
    }
