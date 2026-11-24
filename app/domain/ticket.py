"""티켓 인증 — 사진에서 읽은 값(경기 일자·구장·팀·좌석)이 말이 되나 검사한다. 순수 로직(I/O 없음).

읽기(비전 모델)는 adapters/vision/ticket.py. 중복·포인트(주 2회 이상 40점)는 백엔드가
`seatKey`로 판단한다. 사진만으로는 위조·남의 티켓을 막지 못한다 — 여기서 거르는 건 '티켓이 아님·
시험용 표시·미래·오래된 날짜·구장과 홈 팀이 안 맞음·지정석 좌석 없음'까지다. 경기 일정 대조는
일정 데이터가 없어 하지 않는다(scheduleChecked=false).

seatKey = 일자|구장|블록|열|번호. 블록·열·번호가 모두 있어야 통과한다 — 하나를 가린 사진으로
다른 키를 만들어 같은 좌석을 여러 계정이 쓰는 걸 막으려고. 구역 이름(1루 내야지정석A 등)은 표기가
제각각이라 키에 넣지 않는다. 열·번호가 없는 자유석·잔디석은 같은 좌석인지 가릴 수 없어 받지 않는다.
알려진 한계: 더블헤더 1·2차전의 같은 좌석은 같은 키가 된다(일자만 있고 경기 번호가 없음).
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from app.domain.team_analysis import _ALIASES, TEAMS, team_key

MAX_AGE_DAYS = 30  # 이보다 오래된 경기 티켓은 받지 않는다(인터넷의 옛 티켓 사진 막기)

# 구장 이름에 들어가는 말 → 그 구장을 홈으로 쓰는 팀(제2 구장 포함)
STADIUMS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("잠실", ("잠실",), ("LG", "두산")),
    ("고척", ("고척",), ("키움",)),
    ("인천", ("인천", "문학", "랜더스필드"), ("SSG",)),
    ("수원", ("수원", "위즈파크"), ("KT",)),
    ("대전", ("대전", "한화생명", "이글스파크"), ("한화",)),
    ("대구", ("대구", "라이온즈파크"), ("삼성",)),
    ("광주", ("광주", "챔피언스필드"), ("KIA",)),
    ("창원", ("창원", "NC파크", "엔씨파크"), ("NC",)),
    ("사직", ("사직",), ("롯데",)),
    ("울산", ("울산", "문수"), ("롯데",)),
    ("포항", ("포항",), ("삼성",)),
    ("청주", ("청주",), ("한화",)),
)
_DATE = re.compile(r"^\s*(\d{4})\D{1,3}(\d{1,2})\D{1,3}(\d{1,2})")
_SEAT_UNIT = re.compile(r"(블록|블럭|구역|열|번|호|석|BLOCK|ROW|NO)$")
_SEAT_JUNK = re.compile(r"[\s\-_.,·/#:()]")
# 팀 이름 안에 든 말(띄어쓰기 없이·영문 표기). 이름이 다른 팀 이름의 일부가 되는 경우는 없다.
_TEAM_WORDS: tuple[tuple[str, str], ...] = tuple(
    sorted({(w.upper().replace(" ", ""), t) for t in TEAMS for w in (t,)}
           | {(k.upper().replace(" ", ""), v) for k, v in _ALIASES.items() if len(k) > 2}
           | {("이글스", "한화"), ("라이온즈", "삼성"), ("트윈스", "LG"), ("타이거즈", "KIA"),
              ("랜더스", "SSG"), ("다이노스", "NC"), ("위즈", "KT"), ("베어스", "두산"),
              ("자이언츠", "롯데"), ("히어로즈", "키움"), ("TWINS", "LG"), ("TIGERS", "KIA"),
              ("LIONS", "삼성"), ("EAGLES", "한화"), ("LANDERS", "SSG"), ("DINOS", "NC"),
              ("WIZ", "KT"), ("BEARS", "두산"), ("GIANTS", "롯데"), ("HEROES", "키움")},
           key=lambda x: -len(x[0]))
)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class TicketCheck:
    ok: bool
    game_date: Optional[str]
    stadium: Optional[str]  # 구장 키(잠실·고척 …) — 읽은 이름이 아니라 정규화한 값
    stadium_text: Optional[str]
    home_team: Optional[str]
    away_team: Optional[str]
    section: Optional[str]
    block: Optional[str]
    row: Optional[str]
    number: Optional[str]
    kind: str
    seat_key: Optional[str]
    checks: list[Check] = field(default_factory=list)


def _text(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = re.sub(r"\s+", " ", v).strip()
    return s or None


def parse_date(v: Any) -> Optional[date]:
    """'2026-08-23'·'2026.08.23(토)'·'2026년 8월 23일' → date. 못 읽으면 None."""
    s = _text(v)
    m = _DATE.match(s) if s else None
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def stadium_of(name: Any) -> tuple[Optional[str], tuple[str, ...]]:
    """구장 이름 → (구장 키, 홈 팀들). 모르면 (None, ())."""
    s = (_text(name) or "").replace(" ", "")
    for key, words, homes in STADIUMS:
        if any(w.replace(" ", "") in s for w in words):
            return key, homes
    return None, ()


def team_of(text: Optional[str]) -> Optional[str]:
    """'LG트윈스'·'LG Twins'·'기아' → 'LG'·'LG'·'KIA'. 팀이 하나로 정해지지 않으면 None."""
    if not text:
        return None
    exact = team_key(text)
    if exact:
        return exact
    s = text.upper().replace(" ", "")
    found = {t for w, t in _TEAM_WORDS if w in s}
    return found.pop() if len(found) == 1 else None


def seat_part(v: Any) -> Optional[str]:
    """'314블록'·'314블럭'·'11열'·'006번석'·'B-12' → '314'·'314'·'11'·'6'·'B12'."""
    s = _text(v)
    if not s:
        return None
    s = _SEAT_JUNK.sub("", s).upper()
    prev = None
    while prev != s:  # '6번석'처럼 단위가 겹친 것도 벗긴다
        prev, s = s, _SEAT_UNIT.sub("", s)
    if s.isdigit():
        s = s.lstrip("0") or "0"
    return s or None


def check_ticket(raw: dict[str, Any], today: date) -> TicketCheck:
    """비전 모델이 읽은 값(raw) 검사. today = 서버 기준 오늘(KST)."""
    checks: list[Check] = []
    is_ticket = raw.get("is_ticket") is True
    checks.append(Check("is_ticket", is_ticket,
                        "야구 입장권으로 보임" if is_ticket else "입장권으로 보이지 않음"))
    sample = raw.get("marked_sample") is True
    checks.append(Check("not_sample", not sample,
                        "시험용·SAMPLE 표시가 있음" if sample else "시험용 표시 없음"))

    d = parse_date(raw.get("game_date"))
    if d is None:
        checks.append(Check("date", False, "경기 일자를 읽지 못함"))
    elif d > today:
        checks.append(Check("date", False, f"아직 열리지 않은 경기({d.isoformat()})"))
    elif (today - d).days > MAX_AGE_DAYS:
        checks.append(Check("date", False,
                            f"{MAX_AGE_DAYS}일이 지난 경기({d.isoformat()})는 인증할 수 없음"))
    else:
        checks.append(Check("date", True, d.isoformat()))

    stadium_text = _text(raw.get("stadium"))
    stadium, homes = stadium_of(stadium_text)
    home_text, away_text = _text(raw.get("home_team")), _text(raw.get("away_team"))
    home, away = team_of(home_text), team_of(away_text)
    if (home_text and home is None) or (away_text and away is None):
        checks.append(Check("teams", False,
                            f"모르는 팀: {home_text if home is None and home_text else away_text}"))
    elif home is not None and home == away:
        checks.append(Check("teams", False, "홈·원정 팀이 같음"))
    if stadium is None:
        checks.append(Check("stadium", False, f"모르는 구장: {stadium_text or '읽지 못함'}"))
    elif home is not None and home not in homes:
        checks.append(Check("stadium", False, f"{stadium} 구장은 {home}의 홈이 아님"))
    else:
        checks.append(Check("stadium", True, stadium))
        if home is None and not home_text and len(homes) == 1:
            home = homes[0]

    section = _text(raw.get("seat_section"))
    block, row, number = (seat_part(raw.get(k)) for k in ("seat_block", "seat_row",
                                                           "seat_number"))
    has_seat = block is not None and row is not None and number is not None
    checks.append(Check("seat", has_seat, "좌석 확인" if has_seat else
                        "지정석 좌석(블록·열·번호)을 모두 읽지 못함"))

    ok = all(c.ok for c in checks)
    seat_key = (f"{d.isoformat()}|{stadium}|{block}|{row}|{number}"
                if ok and d is not None else None)
    kind = raw.get("kind") if raw.get("kind") in ("online", "paper") else "unknown"
    return TicketCheck(
        ok=ok, game_date=d.isoformat() if d else None, stadium=stadium,
        stadium_text=stadium_text, home_team=home, away_team=away, section=section,
        block=block, row=row, number=number, kind=kind, seat_key=seat_key, checks=checks,
    )
