"""설명 카드 생성·조회.

**카드는 타임코드의 결정적 함수다.** `cards()`는 그 시점까지 감지된 상황을 프로필로
거른 뒤 카드 문장을 붙여 돌려준다. 시킹·배속이 그냥 되는 이유이자, 배치로 미리 만들어
스냅샷에 굳혀두면 데모 중 LLM 호출이 0이 되는 이유다.

LLM이 죽으면 그 카드만 조용히 빠지고 나머지 화면(스코어보드·분석 패널)은 산다
(user-flow §5 "AI 응답 지연/실패 → 설명 카드: 실패 시 미노출").
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from app.adapters.llm.base import LLMClient, LLMError
from app.adapters.relay.base import RelaySource
from app.domain.detectors import detect
from app.domain.models import Card, ExplainProfile, GameFeed, Situation
from app.domain.profile import select
from app.domain.timeline import Timeline
from app.prompt_templates import CARD_JSON_SCHEMA, card_user_prompt, system_prompt

log = logging.getLogger(__name__)

MIN_LEVEL = 0


class CardService:
    def __init__(
        self,
        relay: RelaySource,
        llm: LLMClient,
        snapshot_dir: Optional[Path] = None,
    ) -> None:
        self.relay = relay
        self.llm = llm
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        # (situation_id, level) → Card. 같은 상황·같은 난이도면 문장이 같다.
        self._cards: dict[tuple[str, int], Card] = {}
        self._snapshots_loaded: set = set()

    # ── 조회 ────────────────────────────────────────────────────────────
    def cards(
        self, game_id: str, video_t: Optional[int], profile: ExplainProfile
    ) -> list[Card]:
        feed = self.relay.load(game_id)
        self._ensure_snapshot(game_id)
        situations = self._situations_until(feed, video_t)
        picked = select(situations, profile)

        cards: list[Card] = []
        for situation in picked:
            card = self._card_for(situation, profile)
            if card is not None:
                cards.append(card)
        # 최신 카드가 위 (와이어프레임 S4 우상단 스택)
        cards.sort(key=lambda c: c.t, reverse=True)
        return cards

    def simplify(
        self, game_id: str, card_id: str, profile: ExplainProfile
    ) -> Optional[Card]:
        """'더 쉽게 설명해줘' — 같은 상황을 한 단계 낮은 난이도로 재생성한다."""
        situation_id, level = _split_card_id(card_id)
        if level <= MIN_LEVEL:
            return None  # 이미 가장 쉬운 단계

        feed = self.relay.load(game_id)
        situation = self._find_situation(feed, situation_id)
        if situation is None:
            return None
        easier = profile.model_copy(update={"level": level - 1})
        return self._card_for(situation, easier)

    def situations(self, game_id: str, video_t: Optional[int] = None) -> list[Situation]:
        return self._situations_until(self.relay.load(game_id), video_t)

    # ── 생성 ────────────────────────────────────────────────────────────
    def _card_for(self, situation: Situation, profile: ExplainProfile) -> Optional[Card]:
        key = (situation.id, profile.level)
        cached = self._cards.get(key)
        if cached is not None:
            return _with_reasons(cached, situation.reasons)

        try:
            result = self.llm.complete(
                system=system_prompt("card_system"),
                user=card_user_prompt(situation, profile.level, profile.categories),
                max_tokens=600,
                schema=CARD_JSON_SCHEMA,
                effort="low",
                thinking=False,
            )
        except LLMError as exc:
            # 카드 하나가 실패해도 나머지는 계속 만든다. 화면은 살아야 한다.
            log.warning("카드 생성 실패 (%s): %s", situation.id, exc)
            return None

        title, body = _parse_card_text(result.text, situation.label)
        card = Card(
            id=_card_id(situation.id, profile.level),
            t=situation.t,
            situation_id=situation.id,
            rule_id=situation.rule_id,
            term_id=situation.term_id,
            category=situation.category,
            level=profile.level,
            title=title,
            body=body,
            reasons=list(situation.reasons),
            source=result.source,
        )
        self._cards[key] = card
        return card

    # ── 내부 ────────────────────────────────────────────────────────────
    def _situations_until(
        self, feed: GameFeed, video_t: Optional[int]
    ) -> list[Situation]:
        timeline = Timeline(feed.meta.relay_video_offset_sec)
        relay_t = timeline.to_relay(int(video_t)) if video_t is not None else None
        return detect(feed, until_t=relay_t)

    def _find_situation(self, feed: GameFeed, situation_id: str) -> Optional[Situation]:
        for situation in detect(feed):
            if situation.id == situation_id:
                return situation
        return None

    def _ensure_snapshot(self, game_id: str) -> None:
        """배치로 미리 만들어둔 카드가 있으면 메모리에 올린다 (RELIABILITY 폴백 계층 ②).

        스냅샷은 LLM 호출을 완전히 대체하므로, 어떤 백엔드로 만들었는지가 중요하다.
        mock으로 만든 스냅샷이 실키 환경에서 쓰이면 조립 문장이 생성물인 척 나간다 —
        그건 조용한 거짓말이라 막는다 (core-belief 2).
        """
        if self.snapshot_dir is None or game_id in self._snapshots_loaded:
            return
        self._snapshots_loaded.add(game_id)
        path = self.snapshot_dir / f"{game_id}.json"
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("스냅샷을 읽지 못했다 (%s): %s", path, exc)
            return

        snapshot_backend = raw.get("backend", "unknown")
        if snapshot_backend != "claude" and getattr(self.llm, "source", "") == "llm":
            log.warning(
                "스냅샷을 무시한다 — backend=%s로 생성됐는데 지금은 실제 모델로 돌고 있다 (%s). "
                "`python scripts/build_cards.py %s`로 다시 만들어라",
                snapshot_backend,
                path,
                game_id,
            )
            return

        loaded = 0
        for item in raw.get("cards", []):
            try:
                card = Card(**item)
            except (TypeError, ValueError) as exc:
                log.warning("스냅샷 카드 형식 오류: %s", exc)
                continue
            # 생성물이면 snapshot, 조립물이면 mock으로 출처를 유지한다.
            card.source = "snapshot" if card.source == "llm" else card.source
            self._cards[(card.situation_id, card.level)] = card
            loaded += 1
        log.info("스냅샷 카드 %d장 적재 (%s, backend=%s)", loaded, game_id, snapshot_backend)


# ── 순수 헬퍼 ───────────────────────────────────────────────────────────
def _card_id(situation_id: str, level: int) -> str:
    # 구분자로 '#'을 쓰면 URL에서 프래그먼트로 잘려 나간다 — '@L'을 쓴다.
    return f"{situation_id}@L{level}"


def _split_card_id(card_id: str) -> tuple[str, int]:
    situation_id, _, suffix = card_id.rpartition("@L")
    if not situation_id or not suffix.isdigit():
        return card_id, MIN_LEVEL
    return situation_id, int(suffix)


def _parse_card_text(text: str, fallback_label: str) -> tuple[str, str]:
    """구조화 출력을 기대하지만, 아니어도 카드가 통째로 날아가지 않게 받아낸다."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            title = str(data.get("title") or "").strip()
            body = str(data.get("body") or "").strip()
            if title and body:
                return title, body
            if body:
                return f"{fallback_label}이란?", body
        except json.JSONDecodeError:
            pass

    head, _, rest = stripped.partition("\n")
    if rest.strip():
        return head.strip(), rest.strip()
    return f"{fallback_label}이란?", stripped


def _with_reasons(card: Card, reasons: Sequence[str]) -> Card:
    """근거는 프로필마다 다르므로 캐시된 문장에 현재 근거를 다시 붙여 돌려준다."""
    clone = card.model_copy(deep=True)
    clone.reasons = list(reasons)
    return clone
