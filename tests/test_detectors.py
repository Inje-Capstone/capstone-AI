import pytest

from app.adapters.relay.fixture import FixtureRelaySource
from app.domain.detectors import RULES, RULES_BY_ID, detect
from app.domain.models import (
    CATEGORY_BASIC,
    CATEGORY_CULTURE,
    CATEGORY_PITCHING,
    CATEGORY_TACTICS,
    LEVEL_BEGINNER,
)
from app.domain.profile import profile_from_onboarding, score, select
from app.domain.timeline import Timeline

GAME_ID = "20260823LGOB"
BALK_T = 7550


@pytest.fixture(scope="module")
def feed():
    return FixtureRelaySource().load(GAME_ID)


@pytest.fixture(scope="module")
def situations(feed):
    return detect(feed)


def _by_rule(situations, rule_id):
    return [s for s in situations if s.rule_id == rule_id]


# ── 감지 ────────────────────────────────────────────────────────────────
def test_every_rule_fires_at_least_once_in_the_fixture(situations):
    """fixture는 모든 규칙을 한 번 이상 밟아야 한다 — 안 밟는 규칙은 미검증 코드다."""
    fired = {s.rule_id for s in situations}
    never = sorted({r.id for r in RULES} - fired)
    assert never == ["bases_loaded"], (
        f"발화하지 않은 규칙: {never} — fixture에 해당 장면을 추가하거나 규칙을 지워라"
    )


def test_balk_is_detected_with_state_and_reasons(situations):
    balks = _by_rule(situations, "balk")
    assert len(balks) == 1
    balk = balks[0]
    assert balk.t == BALK_T
    assert balk.label == "보크"
    assert balk.term_id == "balk"  # 용어 사전 딥링크
    assert balk.state.inning == 7 and balk.state.half == "bot"
    assert balk.state.bases == (False, True, False)  # 보크로 주자가 2루에 가 있다
    assert any("보크" in r for r in balk.reasons)
    assert balk.event_ids == ["e117"]


def test_dropped_third_strike_detected_once_not_twice(situations):
    """낫아웃은 call + result 두 이벤트로 오지만 카드는 하나여야 한다."""
    assert len(_by_rule(situations, "dropped_third_strike")) == 1


def test_decisive_offspeed_only_matches_the_pitch_that_ended_the_atbat(situations):
    pitches = _by_rule(situations, "decisive_offspeed")
    assert len(pitches) == 1
    assert "포크볼" in pitches[0].trigger_text
    # 5회초 포크볼 헛스윙은 낫아웃으로 이어져 decisive가 아니므로 잡히지 않는다.
    assert pitches[0].t == 9100


def test_detect_respects_until_t(feed, situations):
    early = detect(feed, until_t=BALK_T)
    assert all(s.t <= BALK_T for s in early)
    assert len(early) < len(situations)
    assert any(s.rule_id == "balk" for s in early)


def test_situations_are_ordered_by_time(situations):
    times = [s.t for s in situations]
    assert times == sorted(times)


# ── 프로필: 분기가 아니라 가중치 ──────────────────────────────────────────
def test_onboarding_labels_map_to_internal_keys():
    p = profile_from_onboarding(level="입문", categories=["기본 룰", "구종 · 투구"])
    assert p.level == LEVEL_BEGINNER
    assert p.categories == frozenset({CATEGORY_BASIC, CATEGORY_PITCHING})


def test_unknown_onboarding_values_fall_back_to_everything():
    """모르는 값이 와도 카드가 통째로 사라지면 안 된다 (core-belief 1)."""
    p = profile_from_onboarding(level="???", categories=["없는카테고리"])
    assert p.level == LEVEL_BEGINNER
    assert len(p.categories) == 4


def test_score_explains_itself(situations):
    balk = _by_rule(situations, "balk")[0]
    beginner = profile_from_onboarding("입문", ["기본 룰"])
    value, reasons = score(balk, beginner)
    assert value == pytest.approx(1.0 * 1.0 * 1.3)
    assert any("노출 점수" in r for r in reasons)
    assert any("임계값" in r for r in reasons)


def test_beginner_sees_basic_rules_expert_sees_tactics(situations):
    beginner = profile_from_onboarding("입문", ["기본 룰"])
    expert = profile_from_onboarding("익숙", ["전술 · 기록", "구종 · 투구"])

    beginner_rules = {s.rule_id for s in select(situations, beginner)}
    expert_rules = {s.rule_id for s in select(situations, expert)}

    # 입문자에겐 보크·낫아웃·인필드플라이가 뜨고, 병살타·구종은 안 뜬다.
    assert {"balk", "dropped_third_strike", "infield_fly"} <= beginner_rules
    assert "double_play" not in beginner_rules
    assert "decisive_offspeed" not in beginner_rules

    # 룰을 아는 사용자에겐 반대로 전술·구종이 뜨고 보크 설명은 생략된다.
    assert {"double_play", "decisive_offspeed", "steal"} <= expert_rules
    assert "balk" not in expert_rules

    assert beginner_rules != expert_rules


def test_all_categories_selected_shows_balk_to_both_levels(situations):
    """카테고리를 모두 고르면 같은 상황이 두 난이도 모두에 뜬다 — 문장만 달라진다."""
    beginner = profile_from_onboarding("입문", None)
    expert = profile_from_onboarding("익숙", None)
    assert any(s.rule_id == "balk" for s in select(situations, beginner))
    assert any(s.rule_id == "balk" for s in select(situations, expert))


def test_select_attaches_selection_reasons(situations):
    profile = profile_from_onboarding("입문", ["기본 룰"])
    picked = select(situations, profile)
    assert picked
    for s in picked:
        assert any("노출 점수" in r for r in s.reasons)


def test_rules_table_has_no_duplicate_ids():
    assert len(RULES_BY_ID) == len(RULES)


def test_every_rule_has_a_glossary_term():
    for rule in RULES:
        assert rule.term_id, f"{rule.id}에 용어 사전 키가 없다"
        assert rule.category in (
            CATEGORY_BASIC, CATEGORY_PITCHING, CATEGORY_TACTICS, CATEGORY_CULTURE
        )


def test_every_rule_term_exists_in_glossary_seed():
    """용어 시드가 없으면 mock 카드가 빈 문장이 되고 S5 딥링크가 깨진다."""
    from app.adapters.llm.mock import load_glossary

    glossary = load_glossary()
    missing = sorted({r.term_id for r in RULES} - set(glossary))
    assert missing == []


def test_every_category_has_at_least_one_rule():
    """온보딩에서 어떤 관심사를 골라도 그 카테고리 카드가 존재해야 한다."""
    covered = {r.category for r in RULES}
    assert covered == {CATEGORY_BASIC, CATEGORY_PITCHING, CATEGORY_TACTICS, CATEGORY_CULTURE}


def test_once_rules_fire_only_once_per_key(feed, situations):
    """fixture엔 볼넷이 3번 있지만 설명 카드는 첫 번째 한 장뿐이다."""
    walks_in_feed = [e for e in feed.events if e.detail.get("result") == "walk"]
    assert len(walks_in_feed) >= 2
    walks = _by_rule(situations, "walk")
    assert len(walks) == 1
    assert walks[0].event_ids == [walks_in_feed[0].id]


def test_cheer_song_fires_once_per_team(situations):
    cheers = _by_rule(situations, "cheer_song")
    assert [c.state.half for c in cheers] == ["top", "bot"]
    assert all(c.state.inning == 1 for c in cheers)


# ── 타임라인 ────────────────────────────────────────────────────────────
def test_timeline_offset_shifts_the_cutoff(feed):
    plain = Timeline(0)
    shifted = Timeline(300)  # 영상 앞에 5분 프리뷰가 붙은 경우
    assert plain.to_video(100) == 100
    assert shifted.to_video(100) == 400
    assert shifted.to_relay(400) == 100

    assert len(shifted.events_until(feed.events, BALK_T + 300)) == len(
        plain.events_until(feed.events, BALK_T)
    )


def test_timeline_none_means_whole_game(feed):
    assert len(Timeline(0).events_until(feed.events, None)) == len(feed.events)
