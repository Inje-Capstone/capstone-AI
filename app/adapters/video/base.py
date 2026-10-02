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
# VLM에게 고르게 할 장면 단서. video_judge.CUE_KINDS와 같은 어휘다(판정 규칙이 이 이름을 본다).
# 판정 용어(보크·인필드플라이)는 일부러 없다 — 판정은 규칙 엔진이 한다.
EVENT_TYPES = (
    "pitch",  # 투구 (타격 없음)
    "swing_miss",  # 헛스윙
    "contact",  # 배트에 맞음 — 공이 인플레이
    "slide",  # 주자 슬라이딩
    "pickoff_throw",  # 견제구
    "pitcher_stops",  # 투수가 투구·견제 동작 중 멈춤
    "catcher_miss",  # 포수가 공을 놓침
    "infield_popup",  # 내야 높은 뜬공
    "deep_fly",  # 외야 깊은 뜬공
    "ball_over_fence",  # 타구가 담장을 넘어감
    "hit_by_pitch",  # 투구가 타자 몸에 맞음
    "pitching_change",  # 새 투수 등판
    "mound_visit",  # 마운드 방문
    "replay_review",  # 비디오 판독·느린 화면
    "celebration",  # 득점·홈런 세리머니
    "crowd_cheer",  # 응원석 클로즈업
    "other",
)

PROMPT_VERSION = "video-scoreboard-v2"

# 해설에서 들어야 할 말 (Omni 모델은 음성을 같이 듣는다)
SPEECH_KEYWORDS = (
    "보크", "도루", "견제", "폭투", "포일", "낫아웃", "인필드플라이", "희생플라이", "홈런",
    "볼넷", "몸에 맞는 공", "삼진", "병살", "삼중살", "투수 교체",
)


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


@dataclass
class ClipAnalysis:
    """청크 하나의 VLM 출력. 시각은 모두 **전체 영상 기준** 초.

    - events: 장면 단서 (VideoEvent)
    - scoreboard: 점수판 판독 dict (t, inning, half, balls, strikes, outs, bases, away, home)
    - speech: 해설 키워드 dict (t, text)
    """

    events: list
    scoreboard: list
    speech: list


@dataclass(frozen=True)
class VideoClip:
    """분석 단위. `path`는 잘라 둔 청크 파일, `start`는 원본 영상에서의 시작 초."""

    path: Path
    start: float
    duration: float


class VideoAnalyzer(Protocol):
    provider: str
    model: str

    def analyze(self, clip: VideoClip) -> "ClipAnalysis":
        ...


def event_prompt(scenario: str = "KBO 야구 TV 중계") -> str:
    """VLM 지시문 v2 — 점수판 판독 + 장면 단서 + 해설 키워드. 시각은 청크 기준 초.

    판정은 시키지 않는다. 점수판은 상태의 뼈대, 장면·해설은 종류를 가르는 단서다.
    """
    types = ", ".join(EVENT_TYPES)
    words = ", ".join(SPEECH_KEYWORDS)
    return (
        f"This clip is from a {scenario} broadcast with Korean commentary. "
        "Report only what is visible or audible. Do not decide rulings yourself.\n"
        "1) scoreboard: read the on-screen score bug whenever it is visible and every time it "
        "changes. Each item: t (seconds from clip start), inning (int), half (\"top\" for 초, "
        "\"bot\" for 말), balls, strikes, outs (ints), bases (list of occupied bases among 1,2,3), "
        "away, home (runs). Skip moments where the score bug is not visible.\n"
        f"2) events: notable moments. Each item: start, end (seconds), type (one of: {types}), "
        "description (one short Korean sentence).\n"
        f"3) speech: commentary phrases about the play, especially: {words}. "
        "Each item: t (seconds), text (Korean, as heard).\n"
        'Answer with JSON only: {"scoreboard": [...], "events": [...], "speech": [...]}. '
        "Use empty lists when nothing applies."
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


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bases(value: Any) -> Optional[list[int]]:
    """[1, 3] · "1,3" · [true, false, true] → [1, 3]."""
    if isinstance(value, str):
        value = [v for v in value.replace(" ", "").split(",") if v]
    if not isinstance(value, list):
        return None
    if len(value) == 3 and all(isinstance(v, bool) for v in value):
        return [n for n, on in zip((1, 2, 3), value) if on]
    out = sorted({_int(v) for v in value if _int(v) in (1, 2, 3)})
    return out


def _half(value: Any) -> Optional[str]:
    s = str(value or "").strip().lower()
    if s in ("top", "t", "초", "away"):
        return "top"
    if s in ("bot", "bottom", "b", "말", "home"):
        return "bot"
    return None


def parse_scoreboard(rows: Any, clip: VideoClip) -> list[dict[str, Any]]:
    """점수판 판독 → 검증된 dict 목록. 필드 하나라도 못 읽은 판독은 통째로 버린다."""
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        t = _seconds(row.get("t", row.get("time")))
        vals = {k: _int(row.get(k)) for k in ("inning", "balls", "strikes", "outs", "away", "home")}
        half, bases = _half(row.get("half")), _bases(row.get("bases"))
        if t is None or t < 0 or t > clip.duration + 1 or half is None or bases is None:
            continue
        if any(v is None for v in vals.values()):
            continue
        out.append({"t": round(clip.start + min(t, clip.duration), 2), "half": half,
                    "bases": bases, **vals})
    return sorted(out, key=lambda r: r["t"])


def parse_speech(rows: Any, clip: VideoClip) -> list[dict[str, Any]]:
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not str(row.get("text") or "").strip():
            continue
        t = _seconds(row.get("t", row.get("start")))
        if t is None or t < 0 or t > clip.duration + 1:
            continue
        out.append({"t": round(clip.start + min(t, clip.duration), 2),
                    "text": str(row["text"]).strip()[:100]})
    return sorted(out, key=lambda r: r["t"])


def parse_analysis(text: str, clip: VideoClip, provider: str, model: str) -> ClipAnalysis:
    """모델 출력 → ClipAnalysis. 옛 형식(이벤트 목록만)도 받는다."""
    payload = _json_payload(text)
    if isinstance(payload, dict):
        board = parse_scoreboard(payload.get("scoreboard"), clip)
        speech = parse_speech(payload.get("speech"), clip)
    else:
        board, speech = [], []
    events = parse_events(text, clip, provider, model, payload=payload)
    return ClipAnalysis(events=events, scoreboard=board, speech=speech)


def parse_events(
    text: str, clip: VideoClip, provider: str, model: str, payload: Any = None
) -> list[VideoEvent]:
    """모델 출력 → 전체 영상 기준 VideoEvent 목록.

    시각이 없거나 청크 밖이면 버린다(청크 길이보다 약간 넘는 건 경계값으로 자른다).
    모르는 type은 other로 접는다 — 새 라벨이 흘러들어와도 파이프라인이 죽지 않게.
    """
    if payload is None:
        payload = _json_payload(text)
    rows = payload.get("events") if isinstance(payload, dict) else payload
    if isinstance(payload, dict) and rows is None and (
        "scoreboard" in payload or "speech" in payload
    ):
        rows = []
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
