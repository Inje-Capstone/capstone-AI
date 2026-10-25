"""점수판 전용 판독 — 모서리 한 곳을 잘라 키워, 바뀐 순간만 VLM에 한 장씩 읽힌다.

실측(2026-10-26 하이라이트 클립): 영상 통째로 주면 점수판을 클립당 0~4번만, 그것도 자주 틀리게
읽었다. 같은 프레임도 점수판 부분만 잘라 3배로 키워 이미지 한 장으로 주면 이닝·카운트·주자를
정확히 읽었다. 여러 장을 한 요청에 묶으면 다시 틀려서 한 장씩 보낸다.

비용을 줄이려고 '언제 읽을지'는 로컬에서 정한다(ffmpeg 썸네일 비교, API 없음).
1. 모서리 고르기: 네 모서리 이미지를 한 번에 보여 주고 점수판이 있는 곳을 묻는다 (요청 1번)
2. 그 모서리를 1초마다 작은 흑백 썸네일로 뽑아, 기준 프레임과 같은 화소(=점수판 그래픽)를 찾는다
3. 점수판이 보이는 구간을 바뀐 지점마다 나눠, 구간마다 한 장만 읽는다 (요청 = 상태 변화 수)
"""

import base64
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from app.adapters.video.base import VideoAnalyzerError, _json_payload
from app.adapters.video.http import completion_text, post_json

# 모서리 후보 (x, y, w, h — 화면 비율)
CORNERS = {
    "top_left": (0.0, 0.0, 0.45, 0.32),
    "top_right": (0.55, 0.0, 0.45, 0.32),
    "bottom_left": (0.0, 0.68, 0.45, 0.32),
    "bottom_right": (0.55, 0.68, 0.45, 0.32),
}
THUMB_W, THUMB_H = 96, 48
PIXEL_TOL = 30  # 같은 화소로 볼 밝기 차
MASK_MIN = 0.3  # 기준과 같은 값으로 이 비율 이상 나온 화소 = 점수판 그래픽
MAYBE_VISIBLE = 0.55  # 기준 화면과 이만큼 닮으면 점수판이 보일 수 있다 (배경이 바뀌어도)
SHOT_CHANGE = 0.05  # 이웃 프레임이 이만큼 달라지면 다른 장면(또는 점수판 변화)

PICK_PROMPT = (
    "These 4 images are the top-left, top-right, bottom-left and bottom-right corners of one "
    "baseball broadcast frame (images 1-4). Which image contains the score bug (team scores, "
    'inning, ball-strike count, outs)? Answer JSON only: {"image": 1-4, or 0 if none}.'
)
READ_PROMPT = (
    "This is an enlarged corner of a baseball TV broadcast. If a score bug is visible "
    "(team scores, inning, ball-strike count, outs, base runners), read it. Answer JSON only: "
    '{"visible": true|false, "inning": int, "half": "top"|"bot", "balls": int, "strikes": int, '
    '"outs": int, "bases": [occupied bases among 1,2,3], "away": int, "home": int}. '
    "Korean broadcasts: ▲ or 초 = top, ▼ or 말 = bottom; the count is balls-strikes (e.g. 3-1); "
    "outs are filled dots; occupied bases are highlighted squares in the diamond. MLB: TOP/BOT or "
    "arrows, '1 OUT', count like 2-2. The away team is listed first. "
    'If no score bug: {"visible": false}.'
)


# ── 순수 함수 (테스트 대상) ───────────────────────────────────────────────
def crop_filter(corner: str, scale: float = 3.0) -> str:
    x, y, w, h = CORNERS[corner]
    return f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale=iw*{scale}:-2:flags=lanczos"


def graphic_mask(thumbs: Sequence[bytes], ref: int) -> list[int]:
    """기준 프레임과 같은 밝기로 자주 나오는 화소 — 움직이는 화면 위에 고정된 그래픽."""
    base = thumbs[ref]
    n = len(thumbs)
    mask = []
    for i, v in enumerate(base):
        same = sum(1 for t in thumbs if abs(t[i] - v) <= PIXEL_TOL)
        if same / n >= MASK_MIN and same >= 2:
            mask.append(i)
    return mask


def _match(a: bytes, b: bytes, mask: Sequence[int]) -> float:
    if not mask:
        return 0.0
    return sum(1 for i in mask if abs(a[i] - b[i]) <= PIXEL_TOL) / len(mask)


def plan_reads(thumbs: Sequence[bytes], ref: int, max_reads: int = 10) -> list[tuple[int, int]]:
    """점수판이 보일 법한 순간들 → [(시작 프레임, 읽을 프레임)].

    하이라이트에서는 점수판이 1~2초만 비치고 사라지는 일이 많다(실측: 78초 중 9초).
    기준 화면과 닮은(MAYBE_VISIBLE 이상) 프레임을 이어진 덩어리로 묶고, 덩어리 안에서도
    크게 바뀌면 나눈다. 덩어리마다 가장 닮은 프레임 한 장을 읽는다. 상한을 넘으면
    더 닮은 덩어리를 남기되 시간순을 지킨다.
    """
    mask = graphic_mask(thumbs, ref)
    if not mask:
        return []
    vis = [_match(t, thumbs[ref], mask) for t in thumbs]
    runs: list[list[int]] = []
    for i, v in enumerate(vis):
        if v < MAYBE_VISIBLE:
            continue
        if (
            runs
            and runs[-1][-1] == i - 1
            and 1 - _match(thumbs[i - 1], thumbs[i], mask) < SHOT_CHANGE
        ):
            runs[-1].append(i)
        else:
            runs.append([i])
    picks = [(run[0], max(run, key=lambda i: vis[i]), max(vis[i] for i in run)) for run in runs]
    if len(picks) > max_reads:
        keep = sorted(picks, key=lambda p: -p[2])[:max_reads]
        picks = sorted(keep)
    return [(start, at) for start, at, _ in picks]


def normalize_reading(raw: Any, t: float) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict) or not raw.get("visible", True):
        return None
    half = str(raw.get("half", "")).lower()
    half = (
        "top"
        if half.startswith(("top", "초", "t"))
        else "bot"
        if half.startswith(("bot", "말", "b"))
        else ""
    )
    try:
        reading = {
            "t": round(t, 2),
            "inning": int(raw["inning"]),
            "half": half,
            "balls": int(raw["balls"]),
            "strikes": int(raw["strikes"]),
            "outs": int(raw["outs"]),
            "bases": sorted({int(b) for b in raw.get("bases") or [] if int(b) in (1, 2, 3)}),
            "away": int(raw["away"]),
            "home": int(raw["home"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    return reading if half and reading["inning"] >= 1 else None


# ── 실행 ────────────────────────────────────────────────────────────────
class ScorebugReader:
    def __init__(
        self, api_key: str, base_url: str, model: str, max_reads: int = 10, ffmpeg: str = "ffmpeg"
    ) -> None:
        self.api_key, self.model, self.max_reads, self.ffmpeg = api_key, model, max_reads, ffmpeg
        self.base_url = base_url.rstrip("/")
        self.requests = 0

    def _ask(self, images: Sequence[bytes], text: str, max_tokens: int) -> Any:
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()},
            }
            for b in images
        ]
        content.append({"type": "text", "text": text})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        self.requests += 1
        return _json_payload(
            completion_text(
                post_json(f"{self.base_url}/chat/completions", body, token=self.api_key)
            )
        )

    def _frame(self, video: Path, t: float, vf: str) -> bytes:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "f.jpg"
            subprocess.run(
                [
                    self.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{t:.2f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    vf,
                    "-q:v",
                    "2",
                    str(out),
                ],
                check=True,
            )
            return out.read_bytes()

    def _thumbs(self, video: Path, corner: str) -> list[bytes]:
        x, y, w, h = CORNERS[corner]
        vf = f"fps=1,crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale={THUMB_W}:{THUMB_H},format=gray"
        raw = subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vf",
                vf,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout
        size = THUMB_W * THUMB_H
        return [raw[i : i + size] for i in range(0, len(raw) - size + 1, size)]

    def pick_corner(self, video: Path, duration: float) -> tuple[Optional[str], float]:
        for t in (1.0, duration * 0.3, duration * 0.6):
            t = min(max(0.0, t), max(0.0, duration - 0.5))
            images = [self._frame(video, t, crop_filter(c, 2.0)) for c in CORNERS]
            answer = self._ask(images, PICK_PROMPT, 64) or {}
            try:
                k = int(answer.get("image", 0))
            except (TypeError, ValueError):
                k = 0
            if 1 <= k <= len(CORNERS):
                return list(CORNERS)[k - 1], t
        return None, 0.0

    def read(self, video: Path, duration: float) -> dict[str, Any]:
        """영상 → {"corner", "scoreboard": [판독…], "requests"}. 시각은 영상 시작 기준 초."""
        try:
            corner, t_ref = self.pick_corner(video, duration)
            if corner is None:
                return {"corner": None, "scoreboard": [], "requests": self.requests}
            thumbs = self._thumbs(video, corner)
            if not thumbs:
                return {"corner": corner, "scoreboard": [], "requests": self.requests}
            ref = min(int(t_ref), len(thumbs) - 1)
            readings, failed = [], 0
            for start, at in plan_reads(thumbs, ref, self.max_reads):
                try:  # 한 장 실패(503 등)로 클립 전체를 버리지 않는다
                    answer = self._ask(
                        [self._frame(video, at + 0.5, crop_filter(corner))], READ_PROMPT, 256
                    )
                except VideoAnalyzerError:
                    failed += 1
                    continue
                r = normalize_reading(answer, float(start))
                if r:
                    readings.append(r)
        except (subprocess.CalledProcessError, VideoAnalyzerError) as exc:
            raise VideoAnalyzerError(f"점수판 판독 실패: {exc}") from exc
        return {
            "corner": corner,
            "scoreboard": readings,
            "requests": self.requests,
            "failed_reads": failed,
        }
