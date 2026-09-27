"""네이버 문자중계 변환. 실데이터는 커밋하지 않으므로 스키마를 본뜬 가상 경기로 검증한다.

샘플의 필드 모양은 2026-09-27 실측 응답과 같다(textRelays는 최신순, textOptions는 seqno순,
주자 값은 타순 번호, 득점은 뒤따르는 24번 '홈인' 텍스트에서 올라감).
"""

import json

import pytest

from app.adapters.relay.fixture import _parse_fixture
from app.adapters.relay.naver import classify_result, convert_game
from app.domain.detectors import detect
from app.domain.game_state import replay

GAME_ID = "20990401AABB02099"
INFO = {
    "gameId": GAME_ID,
    "gameDate": "2099-04-01",
    "stadium": "가상구장",
    "awayTeamName": "원정",
    "homeTeamName": "홈",
    "awayTeamScore": 1,
    "homeTeamScore": 1,
    "hasVideo": True,
}


class _Game:
    """가상 경기 빌더 — currentGameState를 누적 관리하며 textOption을 찍어낸다."""

    def __init__(self):
        self.seq = 0
        self.relays = []  # (inn, homeOrAway, no, options)
        self.gs = {"out": "0", "base1": "0", "base2": "0", "base3": "0",
                   "awayScore": "0", "homeScore": "0", "pitcher": "P1", "batter": ""}

    def pa(self, inn, hoa, title):
        self.relays.append({"inn": inn, "homeOrAway": hoa, "no": len(self.relays),
                            "title": title, "textOptions": []})

    def opt(self, type_, text, **extra):
        state = extra.pop("state", {})
        self.gs = {**self.gs, **{k: str(v) for k, v in state.items()}}
        self.seq += 1
        self.relays[-1]["textOptions"].append(
            {"seqno": self.seq, "type": type_, "text": text,
             "currentGameState": dict(self.gs), **extra}
        )

    def pitch(self, n, code, stuff, speed, clock):
        self.opt(1, f"{n}구", pitchNum=n, pitchResult=code, stuff=stuff, speed=speed,
                 ptsPitchId=f"990401_{clock}")

    def innings(self):
        lineup = {"pitcher": [{"pcode": "P1", "name": "홈선발"}, {"pcode": "P2", "name": "홈불펜"},
                              {"pcode": "P9", "name": "원정선발"}], "batter": []}
        # 실제 응답처럼 최신순으로 뒤집어서 이닝 하나에 담는다.
        return [{"homeLineup": lineup, "textRelays": list(reversed(self.relays))}]


@pytest.fixture(scope="module")
def game():
    g = _Game()
    g.pa(1, "0", "1회초 원정 공격")
    g.opt(0, "1회초 원정 공격")
    # 1번: 슬라이더 헛스윙 삼진
    g.pa(1, "0", "1번타자 가나다")
    g.opt(8, "1번타자 가나다", batterRecord={"name": "가나다"})
    g.pitch(1, "T", "직구", 145, "140000")
    g.pitch(2, "S", "슬라이더", 132, "140020")
    g.pitch(3, "S", "슬라이더", 131, "140040")
    g.opt(13, "가나다 : 삼진 아웃", state={"out": 1})
    # 2번: 볼넷 → 도루 → 투수 교체
    g.pa(1, "0", "2번타자 라마바")
    g.opt(8, "2번타자 라마바", batterRecord={"name": "라마바"})
    for i, c in enumerate(("B", "B", "B", "B"), 1):
        g.pitch(i, c, "직구", 144, f"1401{i}0")
    g.opt(13, "라마바 : 볼넷", state={"base1": 2})
    g.pa(1, "0", "3번타자 사아자")
    g.opt(8, "3번타자 사아자", batterRecord={"name": "사아자"})
    g.pitch(1, "B", "커브", 118, "140300")
    g.opt(14, "1루주자 라마바 : 도루로 2루까지 진루", state={"base1": 0, "base2": 2})
    g.opt(2, "투수 홈선발 : 투수 홈불펜 (으)로 교체", state={"pitcher": "P2"})
    # 적시타: 결과(23) 뒤 홈인(24)에서 점수가 오른다
    g.pitch(2, "H", "직구", 147, "140500")
    g.opt(23, "사아자 : 중견수 앞 1루타", state={"base1": 3, "base2": 0})
    g.opt(24, "2루주자 라마바 : 홈인", state={"awayScore": 1})
    # 4번: 병살로 이닝 종료 (주자 포스아웃까지 한 묶음)
    g.pa(1, "0", "4번타자 차카타")
    g.opt(8, "4번타자 차카타", batterRecord={"name": "차카타"})
    g.pitch(1, "H", "체인지업", 130, "140600")
    g.opt(13, "차카타 : 유격수 병살타 아웃", state={"out": 3, "base1": 0})
    g.opt(14, "1루주자 사아자 : 포스아웃", state={})
    # 1회말: 홈런
    g.pa(1, "1", "1회말 홈 공격")
    g.opt(0, "1회말 홈 공격", state={"out": 0, "pitcher": "P9"})
    g.pa(1, "1", "1번타자 파하")
    g.opt(8, "1번타자 파하", batterRecord={"name": "파하"})
    g.pitch(1, "H", "직구", 140, "141000")
    g.opt(23, "파하 : 좌익수 뒤 홈런 (홈런거리:120M)", state={})
    g.opt(24, "파하 : 홈인", state={"homeScore": 1})
    return convert_game(INFO, g.innings(), video_offset_sec=30)


@pytest.fixture(scope="module")
def feed(game, tmp_path_factory):
    path = tmp_path_factory.mktemp("fx") / "naver.json"
    path.write_text(json.dumps(game, ensure_ascii=False), encoding="utf-8")
    return _parse_fixture(path)


def _events(game, kind):
    return [e for e in game["events"] if e["kind"] == kind]


def test_meta_maps_to_fixture_game(game):
    meta = game["game"]
    assert meta["id"] == GAME_ID
    assert (meta["away_team"], meta["home_team"], meta["stadium"]) == ("원정", "홈", "가상구장")
    assert meta["has_video"] is True
    assert game["source"] == "naver"


def test_time_is_seconds_since_first_pitch_plus_offset(game):
    pitches = _events(game, "pitch")
    assert pitches[0]["t"] == 30  # 첫 투구 = 0초 + 오프셋 30
    assert pitches[1]["t"] == 50
    times = [e["t"] for e in game["events"]]
    assert times == sorted(times)  # 시각 없는 텍스트도 역행하지 않는다


def test_pitch_detail_and_decisive_strikeout_pitch(game):
    pitches = _events(game, "pitch")
    assert pitches[1]["detail"] == {
        "result": "swing_strike", "pitch_type": "슬라이더", "speed": 132,
    }  # 타석 중간 공엔 decisive 판단 자체를 붙이지 않는다
    assert pitches[2]["detail"]["decisive"] is True  # 삼진을 끝낸 공
    assert not any(p["detail"].get("decisive") for p in pitches[3:])


def test_scoring_result_merges_following_home_in(game):
    hit = next(e for e in _events(game, "result") if e["detail"]["result"] == "single")
    assert hit["detail"]["runs"] == 1
    assert hit["detail"]["bases_after"] == [1]


def test_double_play_group_makes_two_outs_in_one_event(game):
    dp = next(e for e in _events(game, "result") if e["detail"]["result"] == "double_play")
    assert dp["detail"]["outs_made"] == 2
    assert dp["detail"]["bases_after"] == []


def test_mid_atbat_steal_and_pitcher_change(game):
    (steal,) = _events(game, "steal")
    assert steal["detail"]["base"] == 2 and steal["detail"]["bases_after"] == [2]
    (sub,) = _events(game, "sub")
    assert sub["detail"] == {"sub_type": "pitcher", "pitcher": "홈불펜"}


def test_pitcher_names_resolve_from_pcode(game):
    first = _events(game, "pitch")[0]
    assert first["pitcher"] == "홈선발" and first["batter"] == "가나다"


def test_replay_matches_naver_score(feed):
    state = replay(feed.events, feed.meta.away_team, feed.meta.home_team)
    assert (state.away_score, state.home_score) == (1, 1)
    assert state.inning == 1 and state.half == "bot"


def test_detectors_fire_on_converted_feed(feed):
    fired = {s.rule_id for s in detect(feed)}
    assert {"walk", "steal", "pitching_change", "homerun", "double_play",
            "decisive_offspeed", "cheer_song"} <= fired


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("가 : 포수 스트라이크 낫 아웃", "dropped_third_strike"),
        ("가 : 자동 고의4구", "intentional_walk"),
        ("가 : 몸에 맞는 볼", "hbp"),
        ("가 : 3루수 희생번트 아웃 (3루수->1루수)", "sac_bunt"),
        ("가 : 유격수 앞 땅볼로 출루", "fielders_choice"),
        ("가 : 2루수 땅볼 실책으로 출루", "error"),
        ("가 : 유격수 왼쪽 내야안타", "single"),
        ("가 : 3루수 파울플라이 아웃", "flyout"),
        ("가 : 중견수 라인드라이브 아웃", "lineout"),
    ],
)
def test_classify_result(text, code):
    assert classify_result(text) == code


def test_offspeed_card_once_per_pitch_type(tmp_path):
    """같은 구종 결정구가 여러 번 나와도 설명 카드는 구종마다 한 장."""
    g = _Game()
    g.pa(1, "0", "1회초")
    for n, (name, clock) in enumerate((("가", "140000"), ("나", "140100"), ("다", "140200"))):
        g.pa(1, "0", f"{n + 1}번타자 {name}")
        g.opt(8, f"{n + 1}번타자 {name}", batterRecord={"name": name})
        g.pitch(1, "S", "슬라이더" if n < 2 else "포크볼", 130, clock)
        g.opt(13, f"{name} : 삼진 아웃", state={"out": n + 1})
    path = tmp_path / "g.json"
    path.write_text(json.dumps(convert_game(INFO, g.innings()), ensure_ascii=False), "utf-8")
    offspeed = [s for s in detect(_parse_fixture(path)) if s.rule_id == "decisive_offspeed"]
    assert ["슬라이더" in s.reasons[1] for s in offspeed] == [True, False]
    assert len(offspeed) == 2


def _two_pa_game():
    """같은 타자가 두 번 나오는 경기. batterRecord는 그 타석 결과까지 반영된 값이다."""
    g = _Game()
    g.pa(1, "0", "1회초")
    g.pa(1, "0", "1번타자 가나다")
    g.opt(8, "1번타자 가나다", batterRecord={
        "name": "가나다", "seasonHra": 0.301, "ab": 1, "hit": 1, "hr": 1, "rbi": 1})
    g.pitch(1, "H", "직구", 145, "140000")
    g.opt(23, "가나다 : 좌익수 뒤 홈런", state={})
    g.opt(24, "가나다 : 홈인", state={"awayScore": 1})
    g.pa(1, "0", "1번타자 가나다")
    g.opt(8, "1번타자 가나다", batterRecord={
        "name": "가나다", "seasonHra": 0.299, "ab": 2, "hit": 1, "hr": 1, "rbi": 1, "so": 1})
    g.pitch(1, "S", "직구", 145, "140500")
    g.opt(13, "가나다 : 삼진 아웃", state={"out": 1})
    return convert_game(INFO, g.innings())


def test_atbat_stats_do_not_spoil_the_current_atbat():
    first, second = _events(_two_pa_game(), "atbat")
    # 첫 타석: 오늘 기록 없음 (홈런을 미리 알려주지 않는다)
    assert first["detail"]["stats"] == {"season_avg": "0.301", "today": "오늘 첫 타석"}
    # 두 번째 타석: 직전 타석까지의 기록 (삼진은 아직 모른다)
    assert second["detail"]["stats"] == {
        "season_avg": "0.301", "today": "오늘 1타수 1안타 1홈런 1타점",
    }


def test_matchup_prefers_relay_stats(tmp_path):
    from app.domain.game_state import states_by_event
    from app.services.matchup_service import relay_facts

    path = tmp_path / "g.json"
    path.write_text(json.dumps(_two_pa_game(), ensure_ascii=False), encoding="utf-8")
    feed = _parse_fixture(path)
    second_atbat = [e for e in feed.events if e.kind == "atbat"][1]
    idx = feed.events.index(second_atbat)
    state = states_by_event(feed.events, "원정", "홈")[idx]
    facts = relay_facts(feed.events, second_atbat.t, state)
    assert facts is not None
    assert (facts.avg, facts.recent) == ("0.301", "오늘 1타수 1안타 1홈런 1타점")


def test_repeated_pitching_changes_and_bases_loaded_are_throttled(tmp_path):
    """한 팀의 투수 교체는 첫 번째만, 만루는 반 이닝에 한 번만 카드가 된다."""
    g = _Game()
    g.pa(1, "0", "1회초")
    for n, name in enumerate(("가", "나")):
        g.pa(1, "0", f"{n + 1}번타자 {name}")
        g.opt(8, f"{n + 1}번타자 {name}", batterRecord={"name": name})
        g.opt(2, f"투수 홈선발 : 투수 불펜{n} (으)로 교체")
        g.pitch(1, "B", "직구", 140, f"14{n}000")
        g.opt(13, f"{name} : 볼넷", state={"base1": 9, "base2": 9, "base3": 9})
    path = tmp_path / "g.json"
    path.write_text(json.dumps(convert_game(INFO, g.innings()), ensure_ascii=False), "utf-8")
    rules = [s.rule_id for s in detect(_parse_fixture(path))]
    assert rules.count("pitching_change") == 1
    assert rules.count("bases_loaded") == 1


PREVIEW = {
    "gameInfo": {"hName": "홈", "aName": "원정"},
    "homeStandings": {"rank": 3, "w": 74, "l": 55, "d": 1},
    "awayStandings": {"rank": 9, "w": 54, "l": 71, "d": 0},
    "homeTeamPreviousGames": [{"result": "승"}] * 3 + [{"result": "패"}, {"result": "무"}],
    "awayTeamPreviousGames": [{"result": "패"}] * 2,
    "seasonVsResult": {"hw": 7, "hl": 8, "aw": 8, "al": 7},
    "homeStarter": {"playerInfo": {"name": "홈선발"},
                    "currentSeasonStatsOnOpponents": {"era": "4.66", "gameCount": 2, "inn": "9.2"}},
    "awayStarter": {"playerInfo": {"name": "원정선발"}, "currentSeasonStatsOnOpponents": {}},
}


def test_preview_context_sentences():
    from app.adapters.relay.naver import preview_context

    ctx = preview_context(PREVIEW)
    assert ctx["team_form"]["홈"] == (
        "3위 (74승 55패 1무) · 최근 5경기 3승 1패 1무 · 원정 상대 7승 8패"
    )
    assert ctx["team_form"]["원정"] == "9위 (54승 71패) · 최근 2경기 0승 2패 · 홈 상대 8승 7패"
    assert ctx["pitcher_vs_team"] == {"홈선발": "홈선발 올 시즌 원정 상대 2경기 9.2이닝 ERA 4.66"}


def test_matchup_uses_preview_context(tmp_path):
    from app.domain.game_state import states_by_event
    from app.services.matchup_service import relay_facts

    fixture = _two_pa_game()
    fixture["game"]["context"] = {
        "team_form": {"원정": "9위"}, "pitcher_vs_team": {"홈선발": "홈선발 상대 ERA 4.66"}}
    path = tmp_path / "g.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    feed = _parse_fixture(path)
    assert feed.meta.context["team_form"] == {"원정": "9위"}
    atbat = [e for e in feed.events if e.kind == "atbat"][1]
    state = states_by_event(feed.events, "원정", "홈")[feed.events.index(atbat)]
    facts = relay_facts(feed.events, atbat.t, state, feed.meta.context)
    assert facts.team_form == "9위"
    assert facts.vs_pitcher == "홈선발 상대 ERA 4.66"
