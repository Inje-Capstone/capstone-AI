"""네이버 스포츠 문자중계 → 내부 fixture 형식 변환.

서버가 네이버를 직접 부르지 않는다. `scripts/import_naver_relay.py`가 경기를 한 번 받아
`data/fixtures/`에 fixture JSON으로 굳히고, 서버는 FixtureRelaySource로 그 파일만 읽는다.
외부가 죽어도 화면이 살고, 데모 중 네트워크 호출이 0이 된다.

스키마는 2026-09-27 실측(3경기) 기준이다. 필드 의미는 docs/exec-plans의 조사 노트 참고.
네이버 필드명은 이 파일 밖으로 나가지 않는다 — 출력은 RelayEvent와 같은 모양의 dict다.

변환 원칙: 주루를 추론하지 않는다. 네이버가 텍스트마다 실어 보내는 `currentGameState`
(아웃·주자·점수)의 **차이**를 이벤트의 `outs_made`/`runs`/`bases_after`로 옮긴다.
그래서 GameSim으로 재생하면 네이버 상태와 항상 일치한다.
"""

import json
import re
import urllib.request
from datetime import datetime
from typing import Any, Optional

API_BASE = "https://api-gw.sports.naver.com/schedule/games"
MAX_INNINGS = 15  # 연장 포함 상한. 빈 이닝은 건너뛴다.

# textOptions.type
T_INNING_START = 0
T_PITCH = 1
T_SUB = 2
T_ETC = 7
T_BATTER = 8
T_RESULT = 13
T_RESULT_SCORING = 23
T_RUNNER = 14
T_RUNNER_SCORING = 24
T_GAME_END = 99

_RESULT_TYPES = (T_RESULT, T_RESULT_SCORING)
_RUNNER_TYPES = (T_RUNNER, T_RUNNER_SCORING)

PITCH_RESULTS = {
    "B": "ball",
    "T": "called_strike",
    "S": "swing_strike",
    "F": "foul",
    "W": "foul",  # 번트 파울
    "H": "in_play",
}

# 타자 결과 텍스트 → 결과 코드. 위에서부터 먼저 맞는 것.
_RESULT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("낫 아웃", "dropped_third_strike"),
    ("낫아웃", "dropped_third_strike"),
    ("삼진", "strikeout"),
    ("고의4구", "intentional_walk"),
    ("고의 4구", "intentional_walk"),
    ("볼넷", "walk"),
    ("몸에 맞는", "hbp"),
    ("홈런", "homerun"),
    ("희생플라이", "sac_fly"),
    ("희생번트", "sac_bunt"),
    ("삼중살", "triple_play"),
    ("병살", "double_play"),
    ("인필드플라이", "infield_fly"),
    ("3루타", "triple"),
    ("2루타", "double"),
    ("1루타", "single"),
    ("내야안타", "single"),
    ("번트안타", "single"),
    ("실책", "error"),
    ("야수선택", "fielders_choice"),
    ("땅볼로 출루", "fielders_choice"),
    ("땅볼", "groundout"),
    ("라인드라이브", "lineout"),
    ("플라이", "flyout"),
)

_PITCHER_CHANGE_RE = re.compile(r"^투수 \S+ : 투수 (\S+) \(으\)로 교체")


class NaverRelayError(RuntimeError):
    pass


# ── 조회 ────────────────────────────────────────────────────────────────
def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://m.sports.naver.com/"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:  # noqa: S310 - 고정 https 도메인
        body = json.loads(res.read().decode("utf-8"))
    if not body.get("success"):
        raise NaverRelayError(f"네이버 응답 실패: {url} ({body.get('code')})")
    return body["result"]


def fetch_preview(game_id: str) -> Optional[dict[str, Any]]:
    """경기 전 프리뷰(순위·최근 5경기·상대 전적·선발 상대 성적). 없어도 임포트는 된다."""
    try:
        return _get_json(f"{API_BASE}/{game_id}/preview").get("previewData")
    except (NaverRelayError, OSError, ValueError):
        return None


def fetch_game(game_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(경기 메타, 이닝별 textRelayData 목록). 네트워크를 쓰는 유일한 함수."""
    info = _get_json(f"{API_BASE}/{game_id}")["game"]
    innings = []
    for inning in range(1, MAX_INNINGS + 1):
        data = _get_json(f"{API_BASE}/{game_id}/relay?inning={inning}")["textRelayData"]
        if not any(r.get("inn") == inning for r in data.get("textRelays", [])):
            break  # 경기가 끝난 뒤의 이닝은 마지막 이닝 데이터를 그대로 돌려준다
        innings.append(data)
    if not innings:
        raise NaverRelayError(f"문자중계가 없는 경기: {game_id}")
    return info, innings


# ── 변환 ────────────────────────────────────────────────────────────────
def convert_game(
    info: dict[str, Any],
    innings: list[dict[str, Any]],
    video_offset_sec: int = 0,
    has_video: Optional[bool] = None,
    preview: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """네이버 원본 → fixture JSON(dict). FixtureRelaySource가 그대로 읽는다."""
    options = _flatten(innings, info["gameId"])
    names = _pitcher_names(innings)
    events = _Converter(names, video_offset_sec).run(options)
    return {
        "_note": (
            "네이버 스포츠 문자중계에서 변환한 실제 경기 데이터. "
            "scripts/import_naver_relay.py로 재생성한다. 팀 레포 커밋 허용(2026-09-28)."
        ),
        "source": "naver",
        "game": {
            "id": info["gameId"],
            "date": info.get("gameDate", ""),
            "stadium": info.get("stadium") or "",
            "away_team": info["awayTeamName"],
            "home_team": info["homeTeamName"],
            # 우리 서비스에 영상이 붙었는지(네이버 영상 유무 아님). analyze_video --apply가 켠다.
            "has_video": bool(has_video),
            "unavailable_reason": None if has_video else "영상 미확보",
            "video_duration_sec": events[-1]["t"] if events else 0,
            "relay_video_offset_sec": 0,  # 오프셋은 이미 t에 반영했다
            "final_score": {"away": info.get("awayTeamScore"), "home": info.get("homeTeamScore")},
            "context": preview_context(preview) if preview else {},
        },
        "events": events,
    }


def preview_context(preview: dict[str, Any]) -> dict[str, Any]:
    """프리뷰 → 분석 패널용 문장. 전부 **경기 전** 값이라 결과 스포일러가 없다.

    - team_form[팀명]: 순위·시즌 전적·최근 5경기·상대 전적
    - pitcher_vs_team[선발명]: 올 시즌 상대 팀 상대 성적 (타자 대 투수 전적은 네이버가 주지 않는다)
    """
    info = preview.get("gameInfo") or {}
    names = {"home": info.get("hName"), "away": info.get("aName")}
    vs = preview.get("seasonVsResult") or {}
    team_form: dict[str, str] = {}
    for side, key in (("home", "h"), ("away", "a")):
        name = names[side]
        if not name:
            continue
        parts = []
        st = preview.get(f"{side}Standings") or {}
        if st.get("rank"):
            draws = f" {st['d']}무" if st.get("d") else ""
            record = f"{st.get('w', 0)}승 {st.get('l', 0)}패{draws}"
            parts.append(f"{st['rank']}위 ({record})")
        prev = preview.get(f"{side}TeamPreviousGames") or []
        if prev:
            w = sum(1 for g in prev if g.get("result") == "승")
            lose = sum(1 for g in prev if g.get("result") == "패")
            draw = sum(1 for g in prev if g.get("result") == "무")
            recent = f"최근 {len(prev)}경기 {w}승 {lose}패" + (f" {draw}무" if draw else "")
            parts.append(recent)
        other = names["away" if side == "home" else "home"]
        if vs and other:
            parts.append(f"{other} 상대 {vs.get(key + 'w', 0)}승 {vs.get(key + 'l', 0)}패")
        if parts:
            team_form[name] = " · ".join(parts)

    pitcher_vs_team: dict[str, str] = {}
    for side, opp in (("home", names["away"]), ("away", names["home"])):
        starter = preview.get(f"{side}Starter") or {}
        name = (starter.get("playerInfo") or {}).get("name")
        stats = starter.get("currentSeasonStatsOnOpponents") or {}
        if name and opp and stats.get("era") is not None:
            pitcher_vs_team[name] = (
                f"{name} 올 시즌 {opp} 상대 {stats.get('gameCount', 0)}경기 "
                f"{stats.get('inn', '0')}이닝 ERA {stats['era']}"
            )
    return {"team_form": team_form, "pitcher_vs_team": pitcher_vs_team}


def _flatten(innings: list[dict[str, Any]], game_id: str) -> list[dict[str, Any]]:
    """타석 묶음(최신순)을 풀어 seqno 순 텍스트 목록으로. 이닝·초말·타석 번호를 붙인다."""
    seen: set[int] = set()
    flat = []
    for data in innings:
        for relay in data.get("textRelays", []):
            half = "top" if str(relay.get("homeOrAway")) == "0" else "bot"
            for opt in relay.get("textOptions", []):
                seq = opt.get("seqno")
                if seq is None or seq in seen:
                    continue
                seen.add(seq)
                flat.append({**opt, "_inn": relay["inn"], "_half": half, "_pa": relay["no"]})
    if not flat:
        raise NaverRelayError(f"변환할 중계 텍스트가 없다: {game_id}")
    return sorted(flat, key=lambda o: o["seqno"])


def _pitcher_names(innings: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for data in innings:
        for side in ("homeEntry", "awayEntry", "homeLineup", "awayLineup"):
            for group in ("pitcher", "batter"):
                for p in (data.get(side) or {}).get(group) or []:
                    if p.get("pcode") and p.get("name"):
                        names[str(p["pcode"])] = p["name"]
    return names


def _state_of(opt: dict[str, Any]) -> dict[str, Any]:
    gs = opt.get("currentGameState") or {}
    return {
        "outs": int(gs.get("out") or 0),
        "bases": [n for n, key in ((1, "base1"), (2, "base2"), (3, "base3"))
                  if str(gs.get(key) or "0") != "0"],
        "away": int(gs.get("awayScore") or 0),
        "home": int(gs.get("homeScore") or 0),
        "pitcher": str(gs.get("pitcher") or ""),
    }


def _after_colon(text: str) -> str:
    return text.split(" : ", 1)[1] if " : " in text else text


def classify_result(text: str) -> str:
    body = _after_colon(text)
    for needle, code in _RESULT_PATTERNS:
        if needle in body:
            return code
    return "out" if "아웃" in body else "other"


def _runner_call(text: str) -> tuple[str, dict[str, Any]]:
    """타석 도중 주자 이벤트 → (kind, detail). 도루는 steal, 나머지는 call."""
    body = _after_colon(text)
    if "도루실패" in body or "도루 실패" in body:
        return "steal", {"caught": True}
    if "도루" in body:
        m = re.search(r"(\d)루까지", body)
        return "steal", {"base": int(m.group(1)) if m else None}
    for needle, call in (("보크", "balk"), ("폭투", "wild_pitch"), ("포일", "passed_ball")):
        if needle in body:
            return "call", {"call": call}
    return "call", {"call": "runner"}


def _today_line(record: dict[str, Any]) -> str:
    """직전 타석까지의 오늘 기록. 예: `오늘 3타수 1안타 1타점 1볼넷`"""
    parts = [f"{record.get('ab', 0)}타수 {record.get('hit', 0)}안타"]
    for key, label in (("hr", "홈런"), ("rbi", "타점"), ("bb", "볼넷"), ("so", "삼진")):
        if record.get(key):
            parts.append(f"{record[key]}{label}")
    return "오늘 " + " ".join(parts)


class _Converter:
    def __init__(self, pitcher_names: dict[str, str], video_offset_sec: int) -> None:
        self.names = pitcher_names
        self.offset = video_offset_sec
        self.events: list[dict[str, Any]] = []
        self.first_pitch: Optional[datetime] = None
        self.t = 0
        self.prev = {"outs": 0, "bases": [], "away": 0, "home": 0}
        self.half_key: Optional[tuple[int, str]] = None
        self.batter: Optional[str] = None
        self.last_record: dict[str, dict[str, Any]] = {}  # 타자별 직전 타석 기록

    # 주 루프: 타자 결과 + 뒤따르는 주자 텍스트를 한 이벤트로 합친다.
    def run(self, options: list[dict[str, Any]]) -> list[dict[str, Any]]:
        i = 0
        while i < len(options):
            opt = options[i]
            self._enter_half(opt)
            kind = opt.get("type")
            if kind == T_PITCH:
                self._pitch(opt, options[i + 1:])
            elif kind == T_BATTER:
                self._atbat(opt)
            elif kind == T_SUB:
                self._sub(opt)
            elif kind in _RESULT_TYPES:
                j = i + 1
                while j < len(options) and options[j].get("type") in _RUNNER_TYPES \
                        and options[j]["_pa"] == opt["_pa"]:
                    j += 1
                self._result(opt, options[i:j])
                i = j
                continue
            elif kind in _RUNNER_TYPES:
                self._runner(opt)
            elif kind == T_ETC:
                self._emit(opt, "note", {})
            # 0(이닝 시작)·99(경기 종료)는 상태가 대신 말해준다.
            i += 1
        return self.events

    def _enter_half(self, opt: dict[str, Any]) -> None:
        key = (opt["_inn"], opt["_half"])
        if key != self.half_key:
            self.half_key = key
            # 새 공격: 아웃·주자는 초기화되고 점수만 이어진다(GameSim._sync_half와 같다).
            self.prev = {**self.prev, "outs": 0, "bases": []}

    def _clock(self, opt: dict[str, Any]) -> int:
        """투구 시각(ptsPitchId)으로 경과 초를 갱신한다. 시각 없는 텍스트는 직전 값."""
        pid = opt.get("ptsPitchId")
        if pid:
            try:
                at = datetime.strptime(pid, "%y%m%d_%H%M%S")
            except ValueError:
                at = None
            if at is not None:
                if self.first_pitch is None:
                    self.first_pitch = at
                self.t = max(self.t, int((at - self.first_pitch).total_seconds()))
        return self.t + self.offset

    def _pitcher(self, opt: dict[str, Any]) -> Optional[str]:
        code = _state_of(opt)["pitcher"]
        return self.names.get(code)

    def _emit(
        self, opt: dict[str, Any], kind: str, detail: dict[str, Any], suffix: str = ""
    ) -> None:
        self.events.append({
            "id": f"n{opt['seqno']}{suffix}",
            "t": self._clock(opt),
            "inning": opt["_inn"],
            "half": opt["_half"],
            "kind": kind,
            "text": opt.get("text", ""),
            "batter": self.batter,
            "pitcher": self._pitcher(opt),
            "detail": detail,
        })

    def _delta(self, last: dict[str, Any]) -> dict[str, Any]:
        now = _state_of(last)
        runs = (now["away"] - self.prev["away"]) + (now["home"] - self.prev["home"])
        outs_made = max(0, now["outs"] - self.prev["outs"])
        self.prev = now
        return {"outs_made": outs_made, "runs": max(0, runs), "bases_after": now["bases"]}

    def _atbat(self, opt: dict[str, Any]) -> None:
        record = opt.get("batterRecord") or {}
        words = opt.get("text", "").split()
        self.batter = record.get("name") or (words[-1] if words else self.batter)
        detail: dict[str, Any] = {}
        if record.get("name"):
            detail["stats"] = self._stats_before(record)
            self.last_record[record["name"]] = record
        self._emit(opt, "atbat", detail)

    def _stats_before(self, record: dict[str, Any]) -> dict[str, Any]:
        """타석 **시작 시점**에 보여줄 기록.

        네이버 batterRecord는 그 타석 결과까지 반영된 값이라 그대로 쓰면 결과를 미리
        알려주는 셈이 된다. 같은 타자의 직전 타석 기록을 쓰고, 첫 타석이면 오늘 기록은
        비운다. 시즌 타율만은 첫 타석에 직전 값이 없어 이 타석이 포함된 값을 쓴다
        (소수 셋째 자리 수준의 차이).
        """
        prev = self.last_record.get(record["name"])
        season = (prev or record).get("seasonHra")
        stats: dict[str, Any] = {
            "season_avg": f"{float(season):.3f}" if season is not None else None,
            "today": _today_line(prev) if prev else "오늘 첫 타석",
        }
        return stats

    def _pitch(self, opt: dict[str, Any], rest: list[dict[str, Any]]) -> None:
        detail: dict[str, Any] = {
            "result": PITCH_RESULTS.get(opt.get("pitchResult") or "", "unknown"),
            "pitch_type": opt.get("stuff") or None,
            "speed": opt.get("speed"),
        }
        # 삼진으로 끝난 타석의 마지막 공 = 결정구
        nxt = next((o for o in rest if o.get("type") in (T_PITCH, *_RESULT_TYPES)), None)
        if nxt is not None and nxt.get("type") in _RESULT_TYPES and nxt["_pa"] == opt["_pa"]:
            detail["decisive"] = classify_result(nxt.get("text", "")) == "strikeout"
        self._emit(opt, "pitch", detail)

    def _sub(self, opt: dict[str, Any]) -> None:
        m = _PITCHER_CHANGE_RE.match(opt.get("text", ""))
        if m:
            self._emit(opt, "sub", {"sub_type": "pitcher", "pitcher": m.group(1)})
        else:
            self._emit(opt, "sub", {"sub_type": "player"})

    def _result(self, opt: dict[str, Any], group: list[dict[str, Any]]) -> None:
        code = classify_result(opt.get("text", ""))
        if code == "dropped_third_strike":
            # 카드용 심판 콜을 먼저 흘리고, 결과는 출루 여부로 나눈다.
            self._emit(opt, "call", {"call": "dropped_third_strike"}, suffix="c")
            code = "dropped_third_strike_safe" if "출루" in opt.get("text", "") else "strikeout"
        elif code == "infield_fly":
            self._emit(opt, "call", {"call": "infield_fly"}, suffix="c")
        detail = {"result": code, **self._delta(group[-1])}
        self._emit(opt, "result", detail)

    def _runner(self, opt: dict[str, Any]) -> None:
        kind, detail = _runner_call(opt.get("text", ""))
        self._emit(opt, kind, {**detail, **self._delta(opt)})
