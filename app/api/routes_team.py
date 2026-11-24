"""팀 분석 — 연봉 대비 활약·올 시즌 맞대결·주전 투수 구종(공개 보도·나무위키 기록)."""

from fastapi import APIRouter, HTTPException, Path

from app.adapters.relay.base import RelaySourceError
from app.api.deps import get_relay_source, get_salaries, get_team_season
from app.api.schemas import (
    GameTeamAnalysisOut,
    HeadToHeadOut,
    PitchMixOut,
    PlayerValueOut,
    TeamAnalysisOut,
)
from app.domain.team_analysis import TEAMS, head_to_head, team_key, team_report

router = APIRouter(tags=["team"])

NO_DATA = "이 팀의 시즌 기록이 아직 없어요"


def _analysis(team: str) -> TeamAnalysisOut:
    key = team_key(team)
    if key is None:
        return TeamAnalysisOut(
            team=team, available=False, unavailable_reason=f"모르는 팀입니다: {team}"
        )
    r = team_report(key, get_team_season(), get_salaries())
    if not r.found:
        return TeamAnalysisOut(team=key, available=False, unavailable_reason=NO_DATA)
    return TeamAnalysisOut(
        team=r.team, available=True, season=r.season, rank=r.rank,
        wins=r.wins, losses=r.losses, draws=r.draws,
        win_pct=r.win_pct, avg_salary_10k=r.avg_salary_10k,
        top29_avg_salary_10k=r.top29_avg_salary_10k, total_salary_10k=r.total_salary_10k,
        league_avg_salary_10k=r.league_avg_salary_10k,
        values=[PlayerValueOut(**v.__dict__) for v in r.values],
        head_to_head=[HeadToHeadOut(**h.__dict__) for h in r.head_to_head],
        pitchers=[PitchMixOut(name=p.name, role=p.role, pitches=list(p.pitches), line=p.line)
                  for p in r.pitchers],
    )


@router.get("/api/teams/{team}/analysis", response_model=TeamAnalysisOut, summary="팀 분석")
def team_analysis(team: str = Path(..., description="한화·삼성·LG·KIA·SSG·NC·KT·두산·롯데·키움")):
    if team_key(team) not in TEAMS:
        raise HTTPException(status_code=404, detail=f"모르는 팀입니다: {team}")
    return _analysis(team)


@router.get(
    "/api/games/{game_id}/team-analysis",
    response_model=GameTeamAnalysisOut,
    summary="경기 두 팀 분석 + 올 시즌 맞대결",
)
def game_team_analysis(game_id: str = Path(...)):
    try:
        meta = get_relay_source().load(game_id).meta
    except RelaySourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    away, home = team_key(meta.away_team), team_key(meta.home_team)
    h2h = head_to_head(get_team_season(), away, home) if away and home else None
    return GameTeamAnalysisOut(
        game_id=meta.id,
        away=_analysis(meta.away_team),
        home=_analysis(meta.home_team),
        season_head_to_head=HeadToHeadOut(**h2h.__dict__) if h2h else None,
    )
