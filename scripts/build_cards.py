#!/usr/bin/env python
"""경기 전체 설명 카드를 미리 만들어 스냅샷으로 굳힌다.

왜 사전 생성인가 (docs/RELIABILITY.md):
  · 리플레이는 실시간이 아니다 → 데모 중 카드 생성 API 호출을 0으로 만들 수 있다.
  · Batch API는 표준가의 50%다.
  · 외부 API가 죽어도 스냅샷만 있으면 카드 패널이 산다.

스냅샷에는 완성 카드만 굳히지 않고 **원본 중계 이벤트도 함께** 저장한다. 프롬프트를
고치면 같은 재료로 전량 재생성할 수 있어야 스냅샷이 마르지 않는다.

사용:
    python scripts/build_cards.py 20260823LGOB
    python scripts/build_cards.py 20260823LGOB --levels 0 1 2 --no-batch
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.llm import build_llm_client  # noqa: E402
from app.adapters.llm.base import LLMError  # noqa: E402
from app.adapters.relay.fixture import FixtureRelaySource  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.domain.detectors import detect  # noqa: E402
from app.domain.models import Card, Situation  # noqa: E402
from app.prompt_templates import (  # noqa: E402
    CARD_JSON_SCHEMA,
    card_user_prompt,
    system_prompt,
)
from app.services.card_service import _card_id, _parse_card_text  # noqa: E402

ALL_CATEGORIES_FOR_PROMPT = ("basic_rules", "pitching", "tactics", "culture")


def main() -> int:
    parser = argparse.ArgumentParser(description="설명 카드 배치 사전 생성")
    parser.add_argument("game_id")
    parser.add_argument(
        "--levels", nargs="+", type=int, default=[0, 1, 2], help="생성할 난이도"
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Batch API 대신 순차 호출 (mock 백엔드는 항상 순차)",
    )
    parser.add_argument("--generated-at", default=None, help="스냅샷에 기록할 시각")
    args = parser.parse_args()

    settings = get_settings()
    feed = FixtureRelaySource(settings.fixture_dir).load(args.game_id)
    situations = detect(feed)
    jobs = [(s, level) for s in situations for level in args.levels]
    print(f"상황 {len(situations)}건 × 난이도 {len(args.levels)}개 = 요청 {len(jobs)}건")

    backend = settings.resolved_backend
    if backend == "claude" and not args.no_batch:
        cards = _generate_batch(jobs, settings.llm_model)
    else:
        cards = _generate_sequential(jobs)

    if not cards:
        print("생성된 카드가 없다. 스냅샷을 쓰지 않는다.", file=sys.stderr)
        return 1

    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.snapshot_dir / f"{args.game_id}.json"
    payload = {
        "_note": "배치 사전 생성 스냅샷. cards는 재생성 가능한 산출물이고 relay_events가 재료다.",
        "game_id": args.game_id,
        "generated_at": args.generated_at,
        "backend": backend,
        "model": settings.llm_model if backend == "claude" else "mock",
        "levels": args.levels,
        # 재계산 재료 — 프롬프트를 고치면 이걸로 전량 다시 만든다.
        "relay_events": [e.model_dump() for e in feed.events],
        "cards": [c.model_dump() for c in cards],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"카드 {len(cards)}장 → {out_path}")
    return 0


def _build_card(situation: Situation, level: int, text: str, source: str) -> Card:
    title, body = _parse_card_text(text, situation.label)
    return Card(
        id=_card_id(situation.id, level),
        t=situation.t,
        situation_id=situation.id,
        rule_id=situation.rule_id,
        term_id=situation.term_id,
        category=situation.category,
        level=level,
        title=title,
        body=body,
        reasons=list(situation.reasons),  # 감지 근거만 — 노출 근거는 프로필마다 다르다
        source=source,
    )


def _generate_sequential(jobs: list) -> list:
    client = build_llm_client()
    cards = []
    for index, (situation, level) in enumerate(jobs, start=1):
        try:
            result = client.complete(
                system=system_prompt("card_system"),
                user=card_user_prompt(situation, level, ALL_CATEGORIES_FOR_PROMPT),
                max_tokens=600,
                schema=CARD_JSON_SCHEMA,
                effort="low",
                thinking=False,
            )
        except LLMError as exc:
            print(f"  [{index}/{len(jobs)}] 실패 {situation.id} L{level}: {exc}")
            continue
        cards.append(_build_card(situation, level, result.text, result.source))
        print(f"  [{index}/{len(jobs)}] {situation.label} L{level}")
    return cards


def _generate_batch(jobs: list, model: str) -> list:
    """Batch API — 표준가의 50%. 결과는 순서 보장이 없으므로 custom_id로 맞춘다."""
    import time

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from app.adapters.llm.claude import workspace_headers

    client = anthropic.Anthropic(
        api_key=get_settings().anthropic_api_key, default_headers=workspace_headers()
    )
    index: dict[str, Any] = {}
    requests = []
    for job_no, (situation, level) in enumerate(jobs):
        custom_id = f"c{job_no}"
        index[custom_id] = (situation, level)
        requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=600,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt("card_system"),
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": card_user_prompt(
                                situation, level, ALL_CATEGORIES_FOR_PROMPT
                            ),
                        }
                    ],
                    output_config={
                        "effort": "low",
                        "format": {"type": "json_schema", "schema": CARD_JSON_SCHEMA},
                    },
                    thinking={"type": "disabled"},
                ),
            )
        )

    batch = client.messages.batches.create(requests=requests)
    print(f"배치 생성됨: {batch.id} — 완료까지 기다린다(보통 1시간 이내)")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"  상태 {batch.processing_status} …")
        time.sleep(30)

    cards = []
    for result in client.messages.batches.results(batch.id):
        situation, level = index[result.custom_id]
        if result.result.type != "succeeded":
            print(f"  실패 {situation.id} L{level}: {result.result.type}")
            continue
        message = result.result.message
        text = "".join(b.text for b in message.content if b.type == "text")
        cards.append(_build_card(situation, level, text, "llm"))
    cards.sort(key=lambda c: (c.t, c.level))
    return cards


if __name__ == "__main__":
    raise SystemExit(main())
