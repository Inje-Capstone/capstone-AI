"""실시간 모드 엔드포인트.

- 분석기(scripts/live_video.py)용: 경기 시작·청크 결과 수집·종료 — `X-Live-Token` 필수
- 화면용: `GET /api/live/{id}/stream` (SSE) — 새 상황이 생기면 바로 알린다.
  카드·스코어보드·한 줄 요약·챗봇은 기존 /api/games/{id}/... 를 그대로 쓴다.
"""

import asyncio
import json
import secrets
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_live_source
from app.config import get_settings
from app.domain.models import GameMeta

router = APIRouter(prefix="/api/live", tags=["live"])

POLL_SEC = 0.5


class LiveStartIn(BaseModel):
    away_team: str = Field(min_length=1, max_length=20)
    home_team: str = Field(min_length=1, max_length=20)
    date: str = ""
    stadium: str = ""


class LiveIngestIn(BaseModel):
    """analyze_video의 ClipAnalysis와 같은 모양. 시각은 영상(방송) 기준 초."""

    scoreboard: list[dict] = Field(default_factory=list, max_length=500)
    events: list[dict] = Field(default_factory=list, max_length=500)
    speech: list[dict] = Field(default_factory=list, max_length=500)


class LiveDataIn(BaseModel):
    """경기 데이터(네이버 중계를 변환한 이벤트 전체 + 팀 맥락). 보낼 때마다 전체를 바꿔 끼운다.

    선수 이름·구종·기록만 영상 판정에 붙는다. 데이터의 플레이 결과는 화면에 쓰지 않는다.
    """

    events: list[dict] = Field(default_factory=list, max_length=5000)
    context: dict = Field(default_factory=dict)


def _check_token(token: Optional[str]) -> None:
    expected = get_settings().live_token
    if not expected:
        raise HTTPException(status_code=403,
                            detail="실시간 수집이 꺼져 있다 (ROOKIE_LIVE_TOKEN 미설정)")
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="X-Live-Token이 맞지 않는다")


def _game(game_id: str):
    game = get_live_source().get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"진행 중인 실시간 경기가 아니다: {game_id}")
    return game


@router.post("/{game_id}", summary="실시간 경기 시작 (분석기용)")
def start(payload: LiveStartIn, game_id: str = Path(..., max_length=40),
          x_live_token: Optional[str] = Header(default=None)):
    _check_token(x_live_token)
    meta = GameMeta(id=game_id, date=payload.date, stadium=payload.stadium,
                    away_team=payload.away_team, home_team=payload.home_team, has_video=True)
    get_live_source().start(meta)
    return {"game_id": game_id, "status": "live"}


@router.post("/{game_id}/ingest", summary="청크 분석 결과 수집 (분석기용)")
def ingest(payload: LiveIngestIn, game_id: str = Path(...),
           x_live_token: Optional[str] = Header(default=None)):
    _check_token(x_live_token)
    game = _game(game_id)
    if game.ended:
        raise HTTPException(status_code=409, detail="이미 끝난 경기다")
    fresh = game.ingest(payload.model_dump())
    return {"new_situations": [s.rule_id for s in fresh], "updates": len(game.updates)}


@router.post("/{game_id}/data", summary="경기 데이터 갱신 — 선수·구종·기록 (분석기용)")
def data(payload: LiveDataIn, game_id: str = Path(...),
         x_live_token: Optional[str] = Header(default=None)):
    _check_token(x_live_token)
    game = _game(game_id)
    if game.ended:
        raise HTTPException(status_code=409, detail="이미 끝난 경기다")
    try:
        fresh = game.set_data(payload.events, payload.context or None)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"데이터 이벤트 형식 오류: {exc}") from exc
    return {"new_situations": [s.rule_id for s in fresh], "offset": game.offset,
            "data_events": len(game.data_events)}


@router.post("/{game_id}/end", summary="실시간 경기 종료 (분석기용)")
def end(game_id: str = Path(...), x_live_token: Optional[str] = Header(default=None)):
    _check_token(x_live_token)
    _game(game_id).end()
    return {"game_id": game_id, "status": "ended"}


@router.get("/{game_id}/stream", summary="실시간 업데이트 (SSE)")
async def stream(
    game_id: str = Path(...),
    since: int = Query(default=0, ge=0, description="이어 받기: 마지막으로 받은 순번"),
):
    """`data: {...}` 줄 단위. type=update(새 상황·스코어보드) | end.

    각 메시지에 순번 id가 붙는다 — 끊기면 since=마지막 id로 이어 받는다.
    """
    game = _game(game_id)

    async def gen():
        i = since
        while True:
            while i < len(game.updates):
                yield f"id: {i + 1}\ndata: {json.dumps(game.updates[i], ensure_ascii=False)}\n\n"
                i += 1
            if game.ended:
                return
            await asyncio.sleep(POLL_SEC)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
