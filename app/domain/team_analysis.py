"""팀 분석 — 순위표에 없는 것: 연봉 대비 활약, 올 시즌 맞대결, 주전 투수 구종. 순수 로직(I/O 없음).

재료는 data/stats의 두 파일(구단 성적·주전 기록, 연봉). 값이 없으면 지어내지 않고 None으로 둔다.

연봉 대비 활약: 한 구단 주전 중 연봉과 WAR가 둘 다 있는 국내 선수끼리 '연봉 순위'와
'WAR 순위'를 비교한다. WAR 순위가 연봉 순위보다 충분히(주전 수의 1/4, 최소 2계단) 높으면
'연봉보다 활약', 낮으면 '연봉보다 아쉬움'. 외국인 선수는 계약 단위(달러·총액)가 달라 비교에서 뺀다.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

TEAMS = ("한화", "삼성", "LG", "KIA", "SSG", "NC", "KT", "두산", "롯데", "키움")
_ALIASES = {
    "한화 이글스": "한화", "삼성 라이온즈": "삼성", "LG 트윈스": "LG", "KIA 타이거즈": "KIA",
    "기아": "KIA", "기아 타이거즈": "KIA", "SSG 랜더스": "SSG", "NC 다이노스": "NC",
    "KT 위즈": "KT",
    "두산 베어스": "두산", "롯데 자이언츠": "롯데", "키움 히어로즈": "키움",
    # 네이버 중계 팀 코드
    "HH": "한화", "SS": "삼성", "HT": "KIA", "SK": "SSG", "OB": "두산", "LT": "롯데", "WO": "키움",
}
OVER = "연봉보다 활약"
UNDER = "연봉보다 아쉬움"
FAIR = "연봉만큼"
MIN_RANKED = 4  # 이보다 적으면 순위 비교를 하지 않는다


def team_key(name: Optional[str]) -> Optional[str]:
    """'삼성 라이온즈'·'kt'·'HH' → '삼성'·'KT'·'한화'. 모르면 None."""
    if not name:
        return None
    s = name.strip()
    if s in TEAMS:
        return s
    if s in _ALIASES:
        return _ALIASES[s]
    up = s.upper()
    if up in TEAMS:
        return up
    return next((v for k, v in _ALIASES.items() if k.upper() == up), None)


@dataclass(frozen=True)
class PlayerValue:
    name: str
    group: str  # 타자 | 투수
    spot: Optional[str]  # 포지션 또는 보직
    line: str  # 기록 한 줄
    war: Optional[float]
    salary_10k: Optional[int]  # 국내 선수 연봉(만원)
    contract_usd: Optional[int]  # 외국인 계약 총액(달러)
    war_per_eok: Optional[float]  # 연봉 1억당 WAR
    salary_rank: Optional[int] = None
    war_rank: Optional[int] = None
    verdict: Optional[str] = None
    page_url: Optional[str] = None


@dataclass(frozen=True)
class PitchMix:
    name: str
    role: Optional[str]
    pitches: tuple[str, ...]
    line: str


@dataclass(frozen=True)
class HeadToHead:
    opponent: str
    wins: Optional[int]
    losses: Optional[int]
    draws: Optional[int]


@dataclass(frozen=True)
class TeamReport:
    team: str
    found: bool
    season: Optional[int] = None
    rank: Optional[int] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    draws: Optional[int] = None
    win_pct: Optional[float] = None
    avg_salary_10k: Optional[int] = None
    top29_avg_salary_10k: Optional[int] = None
    total_salary_10k: Optional[int] = None
    league_avg_salary_10k: Optional[int] = None
    values: list[PlayerValue] = field(default_factory=list)
    head_to_head: list[HeadToHead] = field(default_factory=list)
    pitchers: list[PitchMix] = field(default_factory=list)


def _int(v: Any) -> Optional[int]:
    """정수만(불리언·실수·문자열은 None). 손으로 고친 파일의 '4위'·78.5 같은 값을 막는다."""
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _num(v: Any) -> Optional[float]:
    """유한한 수만(NaN·무한대·불리언은 None)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _str(v: Any) -> Optional[str]:
    return v.strip() if isinstance(v, str) and v.strip() else None


def _pitches(v: Any) -> tuple[str, ...]:
    if not isinstance(v, list):
        return ()
    return tuple(x.strip() for x in v if isinstance(x, str) and x.strip())


def _rows(data: Mapping[str, Any], key: str, team: str) -> list[dict[str, Any]]:
    """data[key]에서 그 구단 줄만(모양이 틀린 줄은 건너뜀)."""
    rows = data.get(key)
    if not isinstance(rows, list):
        return []
    return [r for r in rows
            if isinstance(r, dict) and r.get("team") == team and _str(r.get("name"))]


def batter_line(b: Mapping[str, Any]) -> str:
    parts = []
    avg, ops = _str(b.get("avg")), _str(b.get("ops"))
    hr, rbi = _int(b.get("hr")), _int(b.get("rbi"))
    if avg:
        parts.append(f"타율 {avg}")
    if hr is not None:
        parts.append(f"{hr}홈런")
    if rbi is not None:
        parts.append(f"{rbi}타점")
    if ops:
        parts.append(f"OPS {ops}")
    return " · ".join(parts) or "기록 없음"


def pitcher_line(p: Mapping[str, Any]) -> str:
    parts = []
    w, lost = _int(p.get("w")), _int(p.get("l"))
    sv, hld = _int(p.get("sv")), _int(p.get("hld"))
    era, ip = _str(p.get("era")), _str(p.get("ip"))
    if w is not None and lost is not None:
        parts.append(f"{w}승 {lost}패")
    if sv:
        parts.append(f"{sv}세이브")
    if hld:
        parts.append(f"{hld}홀드")
    if era:
        parts.append(f"평균자책점 {era}")
    if ip:
        parts.append(f"{ip}이닝")
    return " · ".join(parts) or "기록 없음"


def _salary_index(salaries: Mapping[str, Any]):
    """(구단, 이름) → 연봉 줄, 이름 → 국내 선수 줄 목록(이적 선수용).

    이름만으로 잇는 건 국내 선수뿐이다. 국내 연봉은 사람을 따라가지만 외국인 계약은 구단과
    맺은 것이라, 시즌 중 옮긴 외국인에게 앞 구단 계약을 붙이면 틀린 값이 된다.
    """
    exact: dict[tuple[str, str], dict] = {}
    by_name: dict[str, list[dict]] = {}
    for kind in ("players", "foreign"):
        rows = salaries.get(kind)
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not _str(row.get("name")):
                continue
            exact[(row.get("team"), row["name"])] = row
            if kind == "players":
                by_name.setdefault(row["name"], []).append(row)
    return exact, by_name


def _find_salary(index, team: str, name: str) -> Optional[dict]:
    exact, by_name = index
    if (team, name) in exact:
        return exact[(team, name)]
    rows = by_name.get(name) or []
    return rows[0] if len(rows) == 1 else None  # 동명이인이 여럿이면 붙이지 않는다


def _ranks(values: Sequence[float]) -> list[int]:
    """큰 값이 1위. 같은 값은 같은 순위."""
    return [1 + sum(1 for v in values if v > x) for x in values]


def salary_values(team: str, team_data: Mapping[str, Any],
                  salaries: Mapping[str, Any]) -> list[PlayerValue]:
    index = _salary_index(salaries)
    rows: list[dict[str, Any]] = []
    for group, key, spot_key, line in (("타자", "batters", "position", batter_line),
                                       ("투수", "pitchers", "role", pitcher_line)):
        for p in _rows(team_data, key, team):
            s = _find_salary(index, team, p["name"]) or {}
            salary = _int(s.get("salary_10k"))
            salary = salary if salary and salary > 0 else None
            usd = _int(s.get("contract_total_usd"))
            war = _num(p.get("war"))
            rows.append({
                "name": p["name"], "group": group, "spot": _str(p.get(spot_key)),
                "line": line(p), "war": war, "salary_10k": salary, "contract_usd": usd,
                "war_per_eok": (round(war / (salary / 10000), 2)
                                if war is not None and salary else None),
                "page_url": _str(p.get("page_url")),
            })

    ranked = [r for r in rows if r["salary_10k"] is not None and r["war"] is not None]
    if len(ranked) >= MIN_RANKED:
        gap = max(2, round(len(ranked) / 4))
        s_rank = _ranks([r["salary_10k"] for r in ranked])
        w_rank = _ranks([r["war"] for r in ranked])
        for r, sr, wr in zip(ranked, s_rank, w_rank):
            r["salary_rank"], r["war_rank"] = sr, wr
            r["verdict"] = OVER if sr - wr >= gap else UNDER if wr - sr >= gap else FAIR

    rows.sort(key=lambda r: (r["salary_10k"] is None, -(r["salary_10k"] or 0), r["name"]))
    return [PlayerValue(**r) for r in rows]


def head_to_head(team_data: Mapping[str, Any], team: str, opponent: str) -> Optional[HeadToHead]:
    """team 쪽에서 본 opponent와의 시즌 맞대결. 없으면 None."""
    rows = team_data.get("head_to_head")
    for h in rows if isinstance(rows, list) else []:
        if not isinstance(h, dict):
            continue
        w, lost, d = _int(h.get("w")), _int(h.get("l")), _int(h.get("d"))
        if h.get("team") == team and h.get("opponent") == opponent:
            return HeadToHead(opponent, w, lost, d)
        if h.get("team") == opponent and h.get("opponent") == team:
            return HeadToHead(opponent, lost, w, d)
    return None


def _team_row(rows: Any, team: str) -> Optional[dict[str, Any]]:
    if not isinstance(rows, list):
        return None
    return next((r for r in rows if isinstance(r, dict) and r.get("team") == team), None)


def team_report(team: str, team_data: Mapping[str, Any],
                salaries: Mapping[str, Any]) -> TeamReport:
    """그 구단 시즌 기록 줄이 없으면 found=False(연봉만 있어도 '기록 없음'으로 내린다)."""
    row = _team_row(team_data.get("teams"), team)
    if row is None:
        return TeamReport(team=team, found=False)
    pay = _team_row(salaries.get("teams"), team) or {}
    league = salaries.get("league") if isinstance(salaries.get("league"), dict) else {}
    h2h = [h for h in (head_to_head(team_data, team, o) for o in TEAMS if o != team) if h]
    pitchers = [PitchMix(p["name"], _str(p.get("role")), _pitches(p.get("pitches")),
                         pitcher_line(p))
                for p in _rows(team_data, "pitchers", team)]
    return TeamReport(
        team=team, found=True, season=_int(team_data.get("season")),
        rank=_int(row.get("rank")), wins=_int(row.get("w")), losses=_int(row.get("l")),
        draws=_int(row.get("d")), win_pct=_num(row.get("win_pct")),
        avg_salary_10k=_int(pay.get("avg_salary_10k")),
        top29_avg_salary_10k=_int(pay.get("top29_avg_salary_10k")),
        total_salary_10k=_int(pay.get("total_salary_10k")),
        league_avg_salary_10k=_int(league.get("avg_salary_10k")),
        values=salary_values(team, team_data, salaries), head_to_head=h2h, pitchers=pitchers,
    )
