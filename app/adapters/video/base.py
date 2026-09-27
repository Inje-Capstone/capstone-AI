"""영상 분석 어댑터 경계 (VSS 방식: 영상 청크 → VLM 이벤트).

"VLM이 보고, 규칙 엔진이 판정한다." 여기서 나오는 VideoEvent는 판정이 아니라 관찰이다.
특이상황 판정은 여전히 중계 이벤트 + detectors가 하고, 영상 이벤트는 ① 영상-중계
시각 정합 ② 카드 근거 문구에만 쓴다.

⚠️ TODO(스키마 미검증): Cosmos Reason 호스팅 API·VSS LVS의 실제 응답은 아직 실측 전이다.
외부 응답 모양은 `parse_events()` 한 곳에서만 해석한다.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import BaseModel

# VLM에게 고르게 할 관찰 종류. 판정 용어(보크·인필드플라이)는 일부러 넣지 않는다.
EVENT_TYPES = (
    "pitch",  # 투구 (타격 없음)
    "hit",  # 타격 — 공이 인플레이
    "home_run",
    "strikeout",
    "walk",
    "stolen_base",
    "pitching_change",
    "mound_visit",
    "celebration",  # 득점·홈런 세리머니
    "crowd_cheer",  # 응원석 클로즈업
    "replay_review",  # 비디오 판독·느린 화면
    "other",
)

PROMPT_VERSION = "video-events-v1"


class VideoAnalyzerError(RuntimeError):
    """분석 실패. 스크립트는 그 청크만 건너뛰고 다음으로 간다."""


class VideoEvent(BaseModel):
    """영상에서 관찰한 장면 하나. 시각은 **전체 영상 기준** 초."""

    t_start: float
    t_end: float
    event_type: str
    description: str = ""
    provider: str = ""
    model: str = ""


@dataclass(frozen=True)
class VideoClip:
    """분석 단위. `path`는 잘라 둔 청크 파일, `start`는 원본 영상에서의 시작 초."""

    path: Path
    start: float
    duration: float


class VideoAnalyzer(Protocol):
    provider: str
    model: str

    def analyze(self, clip: VideoClip) -> list[VideoEvent]:
        ...


def event_prompt(scenario: str = "KBO 야구 TV 중계") -> str:
    """VLM 지시문. 관찰만 시키고, 시각은 청크 기준 초로 받는다."""
    types = ", ".join(EVENT_TYPES)
    return (
        f"This clip is from a {scenario} broadcast. List only what is visibly shown. "
        "Do not judge rules (no balk/infield-fly calls). "
        f"For each notable moment output an object with keys start, end (seconds from the "
        f"start of this clip), type (one of: {types}), description (one short sentence). "
        'Answer with JSON only: {"events": [...]}. Use an empty list if nothing happens.'
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _json_payload(text: str) -> Any:
    """<think>·<answer>·코드펜스를 벗기고 JSON을 꺼낸다. 실패하면 None."""
    body = _THINK_RE.sub("", text)
    for pattern in (_ANSWER_RE, _FENCE_RE):
        m = pattern.search(body)
        if m:
            body = m.group(1)
            break
    body = body.strip()
    start = min((i for i in (body.find("{"), body.find("[")) if i >= 0), default=-1)
    if start < 0:
        return None
    try:
        return json.loads(body[start:])
    except json.JSONDecodeError:
        end = max(body.rfind("}"), body.rfind("]"))
        try:
            return json.loads(body[start:end + 1])
        except json.JSONDecodeError:
            return None


def _seconds(value: Any) -> Optional[float]:
    """`12`, `12.5`, `"00:12"`, `"0:01:05"` → 초."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parts = value.strip().split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        total = 0.0
        for n in nums:
            total = total * 60 + n
        return total
    return None


def parse_events(
    text: str, clip: VideoClip, provider: str, model: str
) -> list[VideoEvent]:
    """모델 출력 → 전체 영상 기준 VideoEvent 목록.

    시각이 없거나 청크 밖이면 버린다(청크 길이보다 약간 넘는 건 경계값으로 자른다).
    모르는 type은 other로 접는다 — 새 라벨이 흘러들어와도 파이프라인이 죽지 않게.
    """
    payload = _json_payload(text)
    rows = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise VideoAnalyzerError(f"이벤트 JSON을 찾지 못했다: {text[:200]!r}")

    events = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _seconds(row.get("start", row.get("start_time", row.get("timestamp"))))
        if start is None or start < 0 or start > clip.duration + 1:
            continue
        end = _seconds(row.get("end", row.get("end_time")))
        end = start if end is None or end < start else min(end, clip.duration)
        kind = str(row.get("type") or row.get("event_type") or "other").strip().lower()
        events.append(
            VideoEvent(
                t_start=round(clip.start + min(start, clip.duration), 2),
                t_end=round(clip.start + end, 2),
                event_type=kind if kind in EVENT_TYPES else "other",
                description=str(row.get("description") or "")[:300],
                provider=provider,
                model=model,
            )
        )
    return sorted(events, key=lambda e: e.t_start)
