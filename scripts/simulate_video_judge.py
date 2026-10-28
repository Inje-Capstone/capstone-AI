#!/usr/bin/env python
"""영상 판정 로직의 상한을 잰다 — 실경기 문자중계로 "완벽한 눈"을 흉내 내서.

실제 영상 없이도 판정 규칙이 맞는지 확인하려고 만든다. 중계의 경기 상태로 점수판 판독을,
중계 문장으로 장면 단서·해설 키워드를 합성해 video_judge에 넣고, 나온 판정을 같은 중계와
채점한다. 판독 오류·단서 누락을 일부러 넣어 VLM이 불완전할 때의 저하도 본다.

이 숫자는 "VLM이 점수판과 장면을 이만큼 잘 읽으면 판정은 이만큼 맞는다"는 뜻이다.
실제 정확도는 영상 실측(analyze_video.py)으로만 말할 수 있다.

사용:
    python scripts/simulate_video_judge.py
    python scripts/simulate_video_judge.py --cue-drop 0.3 --misread 0.1 --seed 7
"""

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.relay.fixture import FixtureRelaySource  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.domain.detectors import detect  # noqa: E402
from app.domain.game_state import KIND_SUB, states_by_event  # noqa: E402
from app.domain.grading import GRADED_RULES, grade  # noqa: E402
from app.domain.models import GameFeed, RelayEvent  # noqa: E402
from app.domain.scorebug import Reading, stabilize  # noqa: E402
from app.domain.video_judge import (  # noqa: E402
    Cue,
    judge,
    merge_states,
    state_events,
    to_relay_events,
)

# 중계 문장 키워드 → (장면 단서, 해설 키워드)
_TEXT_CUES = (
    ("도루", ("slide", "도루")),
    ("보크", ("pitcher_stops", "보크")),
    ("폭투", (None, "폭투")),
    ("포일", (None, "포일")),
    ("홈런", ("ball_over_fence", "홈런")),
    ("희생플라이", ("deep_fly", "희생플라이")),
    ("낫 아웃", ("catcher_miss", "낫아웃")),
    ("낫아웃", ("catcher_miss", "낫아웃")),
    ("인필드플라이", ("infield_popup", "인필드플라이")),
    ("몸에 맞는", ("hit_by_pitch", "몸에 맞는 공")),
    ("볼넷", (None, "볼넷")),
    ("삼진", ("swing_miss", "삼진")),
    ("병살", (None, "병살")),
)


def perfect_eye(
    events: Sequence[RelayEvent], away: str, home: str, reads: int = 2
) -> tuple[list[Reading], list[Cue]]:
    """중계 → 점수판 판독(이벤트마다 `reads`번) + 단서.

    영상 통째 판독은 같은 점수판을 여러 번 읽는다(2). 점수판 전용 판독(scorebug_reader)은
    점수판이 보이는 구간마다 한 장만 읽는다 — 상태마다 한 번이 최악의 경우다(1).
    """
    readings: list[Reading] = []
    cues: list[Cue] = []
    last_t = -1.0
    for e, s in zip(events, states_by_event(events, away, home)):
        # 중계는 마지막 투구와 타석 결과를 같은 초에 찍는다. 실제 점수판은 순서대로 바뀌므로
        # 같은 초의 이벤트는 0.2초씩 뒤로 민다(판독 순서가 섞이면 안정화가 둘 다 버린다).
        t = max(float(e.t), last_t + 0.2)
        last_t = t + 0.1
        r = Reading(t, s.inning, s.half, min(s.balls, 3), min(s.strikes, 2), s.outs,
                    s.bases, s.away_score, s.home_score)
        if s.outs >= 3:  # 경기 끝 3아웃 — 전광판엔 3아웃이 잠깐 보인다
            r = Reading(r.t, r.inning, r.half, 0, 0, 3, r.bases, r.away, r.home)
        readings += [r] + [Reading(r.t + 0.1 * k, *r.key()) for k in range(1, reads)]
        if e.kind == "pitch":
            res = e.detail.get("result")
            if res == "in_play":
                cues.append(Cue(t, "contact"))
            elif res == "swing_strike":
                cues.append(Cue(t, "swing_miss"))
        if e.kind == KIND_SUB and e.detail.get("sub_type") == "pitcher":
            cues.append(Cue(t, "pitching_change"))
        for word, (scene, speech) in _TEXT_CUES:
            if word in e.text:
                if scene:
                    cues.append(Cue(t, scene))
                cues.append(Cue(t + 1, "speech", speech))
    return readings, cues


def add_noise(
    readings: list[Reading], cues: list[Cue], cue_drop: float, misread: float, rng: random.Random
) -> tuple[list[Reading], list[Cue]]:
    """단서 누락 + 점수판 오독(한 번씩 튀는 값)."""
    kept = [c for c in cues if rng.random() >= cue_drop]
    noisy = []
    for r in readings:
        if rng.random() < misread:
            noisy.append(Reading(r.t, r.inning, r.half, rng.randint(0, 3), rng.randint(0, 2),
                                 rng.randint(0, 2), (rng.random() < .5, rng.random() < .5,
                                                     rng.random() < .5), r.away, r.home))
        else:
            noisy.append(r)
    return noisy, kept


def video_feed(
    feed: GameFeed, readings: list[Reading], cues: list[Cue], min_repeat: int = 2
) -> GameFeed:
    # 서비스 피드와 같게 — 판정 이벤트 + 점수판 상태(build_video_feed)
    events = merge_states(
        to_relay_events(judge(readings, cues, min_repeat=min_repeat)),
        state_events(stabilize(readings, min_repeat)),
    )
    meta = feed.meta.model_copy(update={"id": feed.meta.id + "-video"})
    return GameFeed(meta=meta, events=events)


def run(
    game_ids: Sequence[str], cue_drop: float, misread: float, seed: int,
    reads: int = 2, min_repeat: int = 2,
) -> dict:
    source = FixtureRelaySource(get_settings().truth_dir)
    rng = random.Random(seed)
    totals: dict[str, list[int]] = {r: [0, 0, 0] for r in GRADED_RULES}
    for gid in game_ids:
        feed = source.load(gid)
        readings, cues = perfect_eye(
            feed.events, feed.meta.away_team, feed.meta.home_team, reads)
        readings, cues = add_noise(readings, cues, cue_drop, misread, rng)
        vfeed = video_feed(feed, readings, cues, min_repeat)
        for s in grade(detect(feed), detect(vfeed), offset=0, tolerance=30):
            totals[s.rule_id][0] += s.expected
            totals[s.rule_id][1] += s.found
            totals[s.rule_id][2] += s.matched
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("games", nargs="*", default=[
        "20260920HHLG02026", "20260920HTNC02026", "20260920OBKT02026"])
    parser.add_argument("--cue-drop", type=float, default=0.0, help="단서 누락 비율 (0~1)")
    parser.add_argument("--misread", type=float, default=0.0, help="점수판 오독 비율 (0~1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reads", type=int, default=2,
                        help="상태마다 판독 수 (점수판 전용 판독은 최악 1)")
    parser.add_argument("--min-repeat", type=int, default=2, help="judge의 안정화 기준")
    args = parser.parse_args()

    totals = run(args.games, args.cue_drop, args.misread, args.seed, args.reads, args.min_repeat)
    print(f"경기 {len(args.games)}개 · 단서 누락 {args.cue_drop:.0%}"
          f" · 점수판 오독 {args.misread:.0%} · 판독 {args.reads}번 · 안정화 {args.min_repeat}")
    header = ("규칙", "정답", "판정", "일치", "재현율", "정밀도")
    print(f"{header[0]:<22}" + "".join(f"{h:>5}" for h in header[1:4])
          + "".join(f"{h:>8}" for h in header[4:]))
    agg = [0, 0, 0]
    for rule, (e, f, m) in totals.items():
        agg = [agg[0] + e, agg[1] + f, agg[2] + m]
        rec = f"{m / e:.0%}" if e else "-"
        pre = f"{m / f:.0%}" if f else "-"
        print(f"{rule:<22}{e:>5}{f:>5}{m:>5}{rec:>8}{pre:>8}")
    rec = agg[2] / agg[0] if agg[0] else 1.0
    pre = agg[2] / agg[1] if agg[1] else 1.0
    print(f"{'전체':<22}{agg[0]:>5}{agg[1]:>5}{agg[2]:>5}{rec:>8.0%}{pre:>8.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
