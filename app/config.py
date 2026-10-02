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
# 방금 장면 한 줄 요약 — 짧고 많다(경기당 ~100줄). 저지연·저비용 모델 (조사서 2.3.4)
DEFAULT_SUMMARY_MODEL = "claude-haiku-4-5"
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:3000"


class Settings:
    def __init__(self) -> None:
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        # 워크스페이스에 묶이지 않은 키는 이 헤더가 있어야 호출된다(없으면 400).
        self.anthropic_workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
        self.llm_model = os.getenv("ROOKIE_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.summary_model = (
            os.getenv("ROOKIE_SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL).strip()
            or DEFAULT_SUMMARY_MODEL
        )
        # auto | claude | mock | fail
        self.llm_backend = os.getenv("ROOKIE_LLM_BACKEND", "auto").strip().lower()
        self.fixture_dir = Path(os.getenv("ROOKIE_FIXTURE_DIR", str(ROOT / "data" / "fixtures")))
        # 검증용 정답지(네이버 문자중계). 서비스 화면에는 쓰지 않는다 — 영상 판정 채점 전용.
        # 실시간 분석기(scripts/live_video.py)가 결과를 밀어 넣을 때 쓰는 토큰. 비우면 수집 꺼짐.
        self.live_token = os.getenv("ROOKIE_LIVE_TOKEN", "").strip()
        self.truth_dir = Path(os.getenv("ROOKIE_TRUTH_DIR", str(ROOT / "data" / "relay_truth")))
        self.snapshot_dir = Path(
            os.getenv("ROOKIE_SNAPSHOT_DIR", str(ROOT / "data" / "snapshots"))
        )
        self.video_dir = Path(os.getenv("ROOKIE_VIDEO_DIR", str(ROOT / "data" / "video")))
        # 쉼표 구분. 배포 시 프론트 도메인만 남긴다.
        self.cors_origins = [
            o.strip()
            for o in os.getenv("ROOKIE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
            if o.strip()
        ]

    @property
    def resolved_backend(self) -> str:
        if self.llm_backend in ("claude", "mock", "fail"):
            return self.llm_backend
        return "claude" if self.anthropic_api_key else "mock"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
