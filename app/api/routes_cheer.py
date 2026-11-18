"""응원가 안내 — 곡명·구단 공식 채널 링크. 가사·음원은 담지 않는다(저작권)."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from app.adapters.cheer.seed import TEAMS, song_for
from app.adapters.relay.base import RelaySourceError
from app.api.deps import get_cheer_songs, get_relay_source
from app.api.schemas import CheerSongOut, GameCheerSongOut
from app.domain.game_state import replay
from app.domain.timeline import Timeline

router = APIRouter(tags=["cheer"])

NO_SONG = "이 팀의 응원가 안내가 아직 없어요"


@router.get("/api/cheer-songs", response_model=list[CheerSongOut], summary="응원가 목록")
def list_cheer_songs(
    team: Optional[str] = Query(default=None, description="팀 이름 (한화·삼성·LG …)"),
):
    if team is not None and team not in TEAMS:
        raise HTTPException(status_code=404, detail=f"모르는 팀입니다: {team}")
    return [CheerSongOut(**s) for s in get_cheer_songs() if team is None or s["team"] == team]


@router.get(
    "/api/games/{game_id}/cheer-song",
    response_model=GameCheerSongOut,
    summary="지금 타석 타자의 응원가 (없으면 팀 응원가)",
)
def game_cheer_song(
    game_id: str = Path(...),
    t: Optional[int] = Query(default=None, ge=0, description="영상 타임코드(초). 생략=경기 끝"),
):
    try:
        feed = get_relay_source().load(game_id)
    except RelaySourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    timeline = Timeline(feed.meta.relay_video_offset_sec)
    relay_t = timeline.to_relay(int(t)) if t is not None else None
    state = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=relay_t)
    if not state.batter:
        return GameCheerSongOut(available=False, unavailable_reason="아직 타석이 시작되지 않았어요")
    song = song_for(get_cheer_songs(), state.batting_team, state.batter)
    if song is None:
        return GameCheerSongOut(
            available=False, unavailable_reason=NO_SONG,
            batter=state.batter, batting_team=state.batting_team,
        )
    return GameCheerSongOut(
        available=True, batter=state.batter, batting_team=state.batting_team,
        song=CheerSongOut(**song),
    )
