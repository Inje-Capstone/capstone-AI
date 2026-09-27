#!/usr/bin/env python
"""현재 LLM 백엔드로 카드·한 줄 요약·챗봇·퀴즈를 뽑아 규칙 검사를 돌린다.

프롬프트(app/prompts/*.md)를 고친 뒤 실제 모델로 돌려 회귀를 본다. mock이면 조립 문장이라
검사는 형식 확인 정도의 의미만 있다.

사용:
    python scripts/eval_outputs.py 20260823LGOB
    python scripts/eval_outputs.py 20260823LGOB --levels 0 2 --report out.json --max-fail 0.1
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import evals  # noqa: E402
from app.adapters.glossary.seed import load_glossary  # noqa: E402
from app.api.deps import (  # noqa: E402
    get_card_service,
    get_chat_service,
    get_moment_service,
    get_quiz_service,
    get_relay_source,
)
from app.domain.game_state import states_by_event  # noqa: E402
from app.domain.profile import profile_from_onboarding  # noqa: E402


def run(game_id: str, levels: list[int], chat_points: int = 3) -> list[evals.Result]:
    feed = get_relay_source().load(game_id)
    glossary = load_glossary()
    gloss_text = [" ".join(str(v) for v in e.values()) for e in glossary.values()]
    states = {e.id: s for e, s in zip(feed.events,
                                      states_by_event(feed.events, feed.meta.away_team,
                                                      feed.meta.home_team))}
    results: list[evals.Result] = []

    for level in levels:
        profile = profile_from_onboarding(level=level)
        situations = {s.id: s for s in get_card_service().situations(game_id)}
        for card in get_card_service().cards(game_id, None, profile):
            s = situations.get(card.situation_id)
            ctx = gloss_text + ([s.trigger_text, s.state.scoreboard_text(),
                                 s.state.runners_text()] if s else [])
            results.append(evals.Result("card", card.id, f"{card.title} | {card.body}",
                                        evals.check_card(card.title, card.body, ctx)))

        for m in get_moment_service().all(game_id, level):
            event = next(e for e in feed.events if e.id == m.event_id)
            ctx = [event.text, m.scoreboard, str(event.inning)]
            idx = feed.events.index(event)
            if idx:
                ctx.append(states[feed.events[idx - 1].id].scoreboard_text())
            results.append(evals.Result("moment", f"{m.event_id}@L{level}", m.text,
                                        evals.check_moment(m.text, ctx)))

        points = [e.t for e in feed.events if e.kind == "result"]
        step = max(1, len(points) // chat_points)
        for t in points[::step][:chat_points]:
            for q in ("방금 저게 무슨 상황이에요?",) + evals.RED_TEAM_QUESTIONS:
                ans = get_chat_service().answer(game_id, q, t, profile)
                ctx = gloss_text + [ans.context_summary] + [
                    e.text for e in feed.events if e.id in ans.used_event_ids] + [q]
                results.append(evals.Result("chat", f"t{t}@L{level}:{q[:12]}", ans.text,
                                            evals.check_chat(ans.text, ctx)))

        term_ids = [s.term_id for s in situations.values()]
        for item in get_quiz_service().build(term_ids, level=level, seed="eval"):
            results.append(evals.Result("quiz", item.id, item.question,
                                        evals.check_quiz(item.question, item.choices,
                                                         item.answer_index)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game_id")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-fail", type=float, default=0.0, help="허용 실패율 (0~1)")
    args = parser.parse_args()

    results = run(args.game_id, args.levels)
    by_kind = Counter(r.kind for r in results)
    fails = [r for r in results if not r.ok]
    failed_checks = Counter(c.name for r in fails for c in r.checks if not c.ok)
    backend = get_card_service().llm.source
    print(f"백엔드 {backend} · 검사 {len(results)}건 {dict(by_kind)} · 실패 {len(fails)}건")
    for name, n in failed_checks.most_common():
        print(f"  {name}: {n}")
    for r in fails[:10]:
        bad = ", ".join(f"{c.name}({c.detail})" for c in r.checks if not c.ok)
        print(f"  ✗ [{r.kind}] {r.key}: {bad} — {r.text[:60]}")

    if args.report:
        args.report.write_text(json.dumps([{
            "kind": r.kind, "key": r.key, "text": r.text, "ok": r.ok,
            "checks": [c.__dict__ for c in r.checks],
        } for r in results], ensure_ascii=False, indent=1), encoding="utf-8")
    rate = len(fails) / len(results) if results else 0.0
    return 0 if rate <= args.max_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
