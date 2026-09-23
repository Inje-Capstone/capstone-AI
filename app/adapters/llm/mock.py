"""키 없이 도는 결정적 LLM 대체물.

용도는 셋이다: ① `pytest`가 API 키 없이 전량 통과 ② 팀원·CI 환경에서 파이프라인 확인
③ 외부 API가 죽은 데모에서 화면이 살아있게.

**생성이 아니다.** 용어 사전 시드에서 난이도별 문장을 골라 조립할 뿐이며, 그렇게 만든
카드는 `source="mock"`으로 표시되어 UI·보고서에서 시연임을 밝힐 수 있다 (core-belief 2).
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from app.adapters.llm.base import LLMError, LLMResult

GLOSSARY_PATH = Path(__file__).resolve().parents[3] / "data" / "glossary_seed.json"

_LEVEL_RE = re.compile(r"\[난이도\]\s*(\d)")
_TERM_RE = re.compile(r"\[용어\]\s*(\S+)")
_LABEL_RE = re.compile(r"\[상황\]\s*(.+)")
_SCORE_RE = re.compile(r"\[경기\]\s*(.+)")
_QUESTION_RE = re.compile(r"\[질문\]\s*(.+)")
_FACT_RE = re.compile(r"^-\s*(.+)$", re.MULTILINE)

_LEVEL_FIELD = {0: "easy", 1: "standard", 2: "deep"}
_TITLE_TEMPLATE = {
    0: "{name}{이가} 뭐예요?",
    1: "{name}, 왜 나온 걸까요?",
    2: "{name} — 규정과 경기 흐름",
}

_HANGUL_START = 0xAC00
_HANGUL_END = 0xD7A3


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지. 조사('이/가')를 고르는 데 쓴다."""
    if not word:
        return False
    code = ord(word[-1])
    if not _HANGUL_START <= code <= _HANGUL_END:
        return False  # 한글이 아니면 받침 없는 것으로 취급
    return (code - _HANGUL_START) % 28 != 0


def load_glossary() -> dict[str, dict[str, str]]:
    raw = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    return raw.get("terms", {})


class MockLLMClient:
    source = "mock"

    def __init__(self, model: str = "mock-glossary") -> None:
        self.model = model
        self._glossary = load_glossary()

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        schema: Optional[dict[str, Any]] = None,
        effort: str = "low",
        thinking: bool = False,
    ) -> LLMResult:
        level = int(_first(_LEVEL_RE, user, "0"))
        term_id = _first(_TERM_RE, user, "")
        label = _first(_LABEL_RE, user, "이 상황").strip()
        scoreboard = _first(_SCORE_RE, user, "").strip()
        question = _first(_QUESTION_RE, user, "").strip()

        entry = self._glossary.get(term_id, {})
        if not entry and question:
            # 챗봇 요청에는 [용어]가 없다. 최근 중계 문장에서 아는 용어를 집어낸다.
            entry = self._term_in_text(user)
        name = entry.get("name", label)
        explanation = entry.get(_LEVEL_FIELD.get(level, "easy"), "")

        if question:
            text = self._chat_text(question, name, explanation, scoreboard)
        elif "[기록]" in user:
            text = self._matchup_text(user)
        else:
            text = self._card_text(level, name, explanation, scoreboard, schema)

        return LLMResult(
            text=text,
            model=self.model,
            source=self.source,
            input_tokens=len(user) // 4,
            output_tokens=len(text) // 4,
        )

    def _term_in_text(self, text: str) -> dict:
        """본문에 등장하는 용어 중 가장 뒤(=가장 최근)에 나온 것을 고른다.

        중계는 "포크볼"이라고 쓰지 "변화구"라고 쓰지 않으므로 별칭도 함께 본다.
        """
        best_pos, best_entry = -1, {}
        for entry in self._glossary.values():
            for word in [entry.get("name", "")] + list(entry.get("aliases", [])):
                if not word:
                    continue
                pos = text.rfind(word)
                if pos > best_pos:
                    best_pos, best_entry = pos, entry
        return best_entry if best_pos >= 0 else {}

    # ── 조립 ────────────────────────────────────────────────────────────
    def _card_text(
        self,
        level: int,
        name: str,
        explanation: str,
        scoreboard: str,
        schema: Optional[dict[str, Any]],
    ) -> str:
        title = _TITLE_TEMPLATE.get(level, _TITLE_TEMPLATE[0]).format(
            name=name, 이가="이" if has_batchim(name) else "가"
        )
        parts = [explanation] if explanation else []
        if scoreboard:
            parts.append(f"지금은 {scoreboard} 상황에서 나왔어요.")
        body = " ".join(parts) or f"{name} 상황입니다."

        if schema is None:
            return f"{title}\n\n{body}"
        return json.dumps({"title": title, "body": body}, ensure_ascii=False)

    def _matchup_text(self, user: str) -> str:
        """한 줄 해석. 기록을 해석하지 않고 첫 항목을 그대로 되짚어 준다(생성 아님)."""
        facts = _FACT_RE.findall(user)
        if not facts:
            return "기록이 없어 해석을 붙이지 않았어요."
        return f"기록 요약: {facts[0].strip()}"

    def _chat_text(
        self, question: str, name: str, explanation: str, scoreboard: str
    ) -> str:
        head = f"'{question}' 라고 물으셨죠."
        body = explanation or f"{name}에 대한 설명이 아직 준비되지 않았어요."
        tail = f" (현재 {scoreboard})" if scoreboard else ""
        return f"{head} {body}{tail}"


class FailingLLMClient:
    """항상 실패하는 클라이언트. 폴백 경로를 실제로 밟아보기 위한 것."""

    source = "fail"
    model = "always-fails"

    def complete(self, **_kwargs: Any) -> LLMResult:
        raise LLMError("의도적으로 실패시킨 LLM 백엔드 (ROOKIE_LLM_BACKEND=fail)")


def _first(pattern: "re.Pattern[str]", text: str, default: str) -> str:
    match = pattern.search(text)
    return match.group(1) if match else default
