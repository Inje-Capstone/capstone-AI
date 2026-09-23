"""LLM 백엔드 선택. 키가 없으면 조용히 mock으로 내려가서 화면은 계속 산다."""

import logging

from app.adapters.llm.base import LLMClient, LLMError, LLMResult
from app.adapters.llm.mock import FailingLLMClient, MockLLMClient
from app.config import get_settings

log = logging.getLogger(__name__)

__all__ = ["LLMClient", "LLMError", "LLMResult", "build_llm_client"]


def build_llm_client() -> LLMClient:
    settings = get_settings()
    backend = settings.resolved_backend

    if backend == "fail":
        log.warning("LLM 백엔드=fail — 폴백 경로 검증 모드")
        return FailingLLMClient()

    if backend == "claude":
        from app.adapters.llm.claude import ClaudeClient

        try:
            client = ClaudeClient()
            log.info("LLM 백엔드=claude (%s)", client.model)
            return client
        except LLMError as exc:
            # 키가 잘못됐다고 서버가 뜨지 못하면 안 된다 — 시연은 살려두고 크게 로그를 남긴다.
            log.error("Claude 클라이언트 초기화 실패, mock으로 대체한다: %s", exc)

    log.info("LLM 백엔드=mock (생성이 아니라 용어 사전 시드 조립)")
    return MockLLMClient()
