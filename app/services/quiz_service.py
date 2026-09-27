"""오늘 본 룰 퀴즈 (하루 5문제, 시청 종료 팝업 → 퀴즈).

저장 버튼은 없다 — 카드로 본 룰이 곧 그날 퀴즈의 재료다. 포인트(+20P)·출제 기록은
백엔드 소유이고, 이 서비스는 "어떤 룰을 봤나"와 "문항"만 만든다.

문항은 두 경로다.
1. **용어 사전 문항**(기본·폴백): 정답은 그 용어의 난이도별 정의, 오답은 다른 용어의 정의.
   정답이 구조적으로 보장된다. LLM이 없어도, 실패해도 이 경로로 나간다.
2. **LLM 상황형 문항**(실제 모델일 때만): 정의를 근거로 준 상황형 문제. 모양 검증을
   통과 못 하면 그 문항만 1번으로 대체한다.
"""

import hashlib
import json
import logging
import random
from collections.abc import Sequence
from typing import Any, Optional

from pydantic import BaseModel

from app.adapters.llm.base import LLMClient, LLMError
from app.domain.models import LEVEL_LABELS
from app.prompt_templates import system_prompt
from app.services.glossary_service import LEVEL_FIELDS

log = logging.getLogger(__name__)

QUIZ_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "choices": {"type": "array", "items": {"type": "string"}},
        "answer_index": {"type": "integer"},
        "explanation": {"type": "string"},
    },
    "required": ["question", "choices", "answer_index", "explanation"],
    "additionalProperties": False,
}

N_CHOICES = 4


class QuizItem(BaseModel):
    id: str
    term_id: str
    question: str
    choices: list[str]
    answer_index: int
    explanation: str
    source: str  # glossary | llm


class QuizService:
    def __init__(self, terms: dict[str, dict[str, Any]], llm: LLMClient) -> None:
        self.terms = terms
        self.llm = llm

    def build(
        self,
        term_ids: Sequence[str],
        level: int = 0,
        count: int = 5,
        seed: Optional[str] = None,
    ) -> list[QuizItem]:
        """오늘 본 용어로 `count`문제. 모자라면 같은 카테고리 → 나머지 용어로 채운다.

        같은 `seed`(예: 사용자ID+날짜)면 같은 문제·같은 보기 순서가 나온다 — 앱을 다시 열어도
        그날의 퀴즈가 바뀌지 않게.
        """
        picked = self._pick_terms(term_ids, count, seed)
        rng = random.Random(_seed_int(seed or ",".join(picked)))
        items = []
        for n, term_id in enumerate(picked):
            item = None
            if self.llm.source == "llm":
                item = self._llm_item(term_id, level, n, rng)
            items.append(item or self._glossary_item(term_id, level, n, rng))
        return items

    # ── 출제 대상 ───────────────────────────────────────────────────────
    def _pick_terms(self, term_ids: Sequence[str], count: int, seed: Optional[str]) -> list[str]:
        seen: list[str] = []
        for t in term_ids:
            if t in self.terms and t not in seen:
                seen.append(t)
        rng = random.Random(_seed_int(seed or ",".join(seen) or "rookie"))
        watched = seen[:]
        rng.shuffle(watched)
        picked = watched[:count]
        if len(picked) < count:
            cats = {self.terms[t].get("category") for t in seen}
            rest = sorted(t for t in self.terms if t not in picked)
            rng.shuffle(rest)
            rest.sort(key=lambda t: self.terms[t].get("category") not in cats)  # 같은 카테고리 먼저
            picked += rest[: count - len(picked)]
        return picked

    # ── 용어 사전 문항 ──────────────────────────────────────────────────
    def _definition(self, term_id: str, level: int) -> str:
        entry = self.terms[term_id]
        return entry.get(LEVEL_FIELDS.get(level, "easy")) or entry.get("easy", "")

    def _glossary_item(self, term_id: str, level: int, n: int, rng: random.Random) -> QuizItem:
        entry = self.terms[term_id]
        others = sorted(t for t in self.terms if t != term_id)
        rng.shuffle(others)
        distractors = others[: N_CHOICES - 1]
        name = entry.get("name", term_id)

        if n % 2 == 0:  # 용어 → 설명
            question = f"'{name}'에 대한 설명으로 맞는 것은?"
            options = [(term_id, self._definition(term_id, level))] + [
                (t, self._definition(t, level)) for t in distractors
            ]
        else:  # 설명 → 용어
            # 설명 안에 정답 이름이 그대로 있으면 문제가 아니게 된다 — 가린다.
            hint = self._definition(term_id, 0)
            for word in [name, *entry.get("aliases", [])]:
                if word:
                    hint = hint.replace(word, "○○")
            question = f"다음 설명에 해당하는 것은? — {hint}"
            options = [(term_id, name)] + [(t, self.terms[t].get("name", t)) for t in distractors]
        rng.shuffle(options)
        answer = next(i for i, (t, _) in enumerate(options) if t == term_id)
        explanation = self._definition(term_id, min(level + 1, 2))
        return QuizItem(
            id=f"{term_id}:{n}:g",
            term_id=term_id,
            question=question,
            choices=[text for _, text in options],
            answer_index=answer,
            explanation=explanation,
            source="glossary",
        )

    # ── LLM 상황형 문항 ─────────────────────────────────────────────────
    def _llm_item(
        self, term_id: str, level: int, n: int, rng: random.Random
    ) -> Optional[QuizItem]:
        entry = self.terms[term_id]
        user = "\n".join([
            f"[용어] {entry.get('name', term_id)}",
            f"[난이도] {level} · {LEVEL_LABELS.get(level, '입문')}",
            f"[정답 설명] {entry.get('standard') or entry.get('easy', '')}",
            f"[입문 설명] {entry.get('easy', '')}",
            "",
            "이 룰을 확인하는 4지선다 문제를 하나 내라.",
        ])
        try:
            result = self.llm.complete(
                system=system_prompt("quiz_system"), user=user, max_tokens=600,
                schema=QUIZ_JSON_SCHEMA, effort="low", thinking=False,
            )
            data = json.loads(result.text)
        except (LLMError, json.JSONDecodeError) as exc:
            log.info("LLM 퀴즈 생략(%s): %s", term_id, exc)
            return None
        if not _valid(data):
            log.info("LLM 퀴즈 모양 불량(%s): %s", term_id, str(data)[:200])
            return None
        # 모델은 정답을 앞에 두는 버릇이 있다 — 보기 순서는 여기서 섞는다.
        order = list(range(N_CHOICES))
        rng.shuffle(order)
        choices = [data["choices"][i].strip() for i in order]
        return QuizItem(
            id=f"{term_id}:{n}:l",
            term_id=term_id,
            question=data["question"].strip(),
            choices=choices,
            answer_index=order.index(data["answer_index"]),
            explanation=data["explanation"].strip(),
            source="llm",
        )


def _valid(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    answer = data.get("answer_index")
    return (
        isinstance(data.get("question"), str) and data["question"].strip() != ""
        and isinstance(choices, list) and len(choices) == N_CHOICES
        and all(isinstance(c, str) and c.strip() for c in choices)
        and len({c.strip() for c in choices}) == N_CHOICES
        and isinstance(answer, int) and not isinstance(answer, bool)
        and 0 <= answer < N_CHOICES
        and isinstance(data.get("explanation"), str)
    )


def _seed_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
