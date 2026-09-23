"""Anthropic Claude 클라이언트.

모델 기본값은 Sonnet 5 (`claude-sonnet-5`). 카드 생성은 짧고 정형화된 작업이라
thinking을 끄고 effort를 낮춰 지연·비용을 줄이고, 챗봇은 경기 맥락 추론이 필요하므로
adaptive thinking을 켠다.
"""

from typing import Any, Optional

from app.adapters.llm.base import LLMError, LLMResult
from app.config import get_settings


class ClaudeClient:
    source = "llm"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        key = api_key or settings.anthropic_api_key
        if not key:
            raise LLMError("ANTHROPIC_API_KEY가 없다")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - 의존성 누락은 설치 문제
            raise LLMError(f"anthropic SDK를 불러올 수 없다: {exc}") from exc
        self._client = anthropic.Anthropic(api_key=key)

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
        # 시스템 프롬프트는 매 호출 동일한 접두사다 → 캐시 표시를 붙인다.
        # 주의: 실제 캐시 적중은 접두사가 모델별 최소 토큰(수천 토큰)을 넘어야 일어난다.
        # 지금 프롬프트는 그보다 짧아 적중하지 않을 수 있고, 룰북·용어사전 컨텍스트를
        # 붙이는 순간부터 효과가 난다. (표시해두는 비용은 0)
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

        output_config: dict[str, Any] = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
            "thinking": {"type": "adaptive"} if thinking else {"type": "disabled"},
        }

        try:
            response = self._client.messages.create(**params)
        except Exception as exc:  # SDK 예외 계층 전체를 폴백 신호로 변환
            raise LLMError(f"Claude 호출 실패: {exc}") from exc

        # 안전 정책상 거절되면 content가 비어 있을 수 있다 — 인덱싱 전에 확인한다.
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("모델이 응답을 거절했다 (stop_reason=refusal)")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise LLMError("빈 응답")

        usage = response.usage
        return LLMResult(
            text=text,
            model=response.model,
            source=self.source,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
