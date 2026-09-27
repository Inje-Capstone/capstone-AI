"""루키 AI 엔진 — FastAPI 진입점.

프론트(React)와 백엔드 팀이 붙는 지점이다. 이 서버는 상태를 갖지 않고
경기 데이터·설명 생성만 책임진다. 인증·유저 DB·북마크 영속화는 백엔드 팀 영역.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_chat, routes_game
from app.api.deps import get_relay_source
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="ROOKIE AI Engine",
    version="0.1.0",
    description=(
        "야구 입문자를 위한 상황 인지 · 룰 설명 엔진. "
        "현재 중계·기록 데이터는 fixture이며 실소스 연결은 어댑터 교체로 처리한다."
    ),
)

# 기본은 프론트 개발 서버. 배포 시에는 ROOKIE_CORS_ORIGINS로 도메인을 좁힌다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_game.router)
app.include_router(routes_chat.router)


@app.get("/health", tags=["ops"], summary="헬스체크 (데모 전 점검용)")
def health():
    """데모 전 점검 항목: 200 응답 + 어떤 LLM 백엔드로 도는지."""
    settings = get_settings()
    try:
        games = len(get_relay_source().list_games())
        relay_ok = True
    except Exception:  # noqa: BLE001 - 헬스체크는 어떤 이유로도 500이 되면 안 된다
        games, relay_ok = 0, False
    return {
        "status": "ok",
        "llm_backend": settings.resolved_backend,
        "llm_model": settings.llm_model if settings.resolved_backend == "claude" else None,
        "relay_ok": relay_ok,
        "games": games,
    }
