"""환경 설정. 시크릿은 오직 여기(=서버 환경변수)에서만 읽는다.

API 키가 클라이언트/번들로 새는 경로를 만들지 않는다. API 응답에 키를 싣지 않고,
어떤 백엔드를 썼는지(`source`)만 노출한다.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "claude-sonnet-5"


class Settings:
    def __init__(self) -> None:
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.llm_model = os.getenv("ROOKIE_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        # auto | claude | mock | fail
        self.llm_backend = os.getenv("ROOKIE_LLM_BACKEND", "auto").strip().lower()
        self.fixture_dir = Path(os.getenv("ROOKIE_FIXTURE_DIR", str(ROOT / "data" / "fixtures")))
        self.snapshot_dir = Path(
            os.getenv("ROOKIE_SNAPSHOT_DIR", str(ROOT / "data" / "snapshots"))
        )

    @property
    def resolved_backend(self) -> str:
        if self.llm_backend in ("claude", "mock", "fail"):
            return self.llm_backend
        return "claude" if self.anthropic_api_key else "mock"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
