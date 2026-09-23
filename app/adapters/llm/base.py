"""LLM 어댑터 경계.

공급자(Anthropic)를 이 뒤에 가둔다. 서비스 레이어는 `LLMClient`만 알고,
키가 없거나 API가 죽으면 `LLMError`를 받아 폴백한다 (core-belief 5).
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class LLMError(RuntimeError):
    """생성 실패. 상위는 이걸 잡아 '카드 미노출' 같은 조용한 폴백으로 처리한다."""


@dataclass
class LLMResult:
    text: str
    model: str
    source: str  # llm | mock
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    source: str
    model: str

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
        ...
