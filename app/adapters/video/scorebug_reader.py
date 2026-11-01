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
from app.adapters.video.scorebug_layout import Layout, decode, rgb_sampler

# 모서리 후보 (x, y, w, h — 화면 비율)
CORNERS = {
    "top_left": (0.0, 0.0, 0.45, 0.32),
    "top_right": (0.55, 0.0, 0.45, 0.32),
    "bottom_left": (0.0, 0.68, 0.45, 0.32),
    "bottom_right": (0.55, 0.68, 0.45, 0.32),
}
VERSION = "scorebug-crop-v1"  # 스냅샷 캐시 키 — 판독 방식·지시문을 바꾸면 올린다
THUMB_W, THUMB_H = 96, 48
RGB_W, RGB_H = 1280, 720  # 주자·아웃 화소 읽기용 프레임 크기 (scorebug_layout)
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
READ_PROMPT_V2 = (
    "This is an enlarged corner of a baseball TV broadcast. If a score bug is visible "
    "(team scores, inning, ball-strike count, outs, base runners), read it. Answer JSON only: "
    '{"visible": true|false, "inning": int, "half": "top"|"bot", "balls": int, "strikes": int, '
    '"outs": int, "bases": [occupied bases among 1,2,3], "away": int, "home": int}. '
    "How to read it: "
    "INNING is the number next to the ▲/▼ arrow (▲ or 초 = top, ▼ or 말 = bottom). Ignore other "
    "numbers such as batting order ('5. Name'), pitch counts and averages. "
    "COUNT is balls-strikes, e.g. 1-2 means 1 ball and 2 strikes. "
    "OUTS: Korean broadcasts show two small out dots near the count. Count only the FILLED "
    "(solid colored) dots: both filled = 2 outs, one = 1, none (hollow or pale) = 0. "
    "BASES: three small diamonds placed like the infield. The TOP diamond is 2nd base, the RIGHT "
    "diamond is 1st base, the LEFT diamond is 3rd base. A solid colored diamond = runner on that "
    "base; a white, hollow or dashed diamond = empty. Decide each diamond by its position. "
    "The away team is listed first. MLB: TOP/BOT or arrows, '1 OUT'. "
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


def plan_periodic(
    thumbs: Sequence[bytes], ref: int, every: float, max_reads: int = 600
) -> list[tuple[int, int]]:
    """긴 경기 영상 — 점수판이 보이는 동안 `every`초마다 한 장, 가렸다 다시 나타나면 바로 한 장.

    리허설 실측(2026-10-28, 3분 중계): 모서리 배경(관중석)이 대부분 기준 화면과 같아 '점수판
    그래픽' 마스크가 모서리의 94%가 됐다. 카메라가 움직일 때마다 바뀜으로 잡혀 1초마다 읽었다
    (3분에 87번 — 경기 1시간이면 1,600번). 상태는 투구마다(15~25초) 바뀌니 일정 간격으로 읽고,
    같은 상태가 두 번 읽혀야 믿는다(안정화 2 — 한 장 오독을 반복으로 거른다).
    """
    mask = graphic_mask(thumbs, ref)
    if not mask:
        return []
    vis = [_match(t, thumbs[ref], mask) for t in thumbs]
    picks: list[tuple[int, int]] = []
    last, hidden = -1, 2
    for i, v in enumerate(vis):
        if v < MAYBE_VISIBLE:
            hidden += 1
            continue
        if hidden >= 2 or i - last >= every:  # 2초 넘게 가려졌다 나타남 = 새 장면
            picks.append((i, i))
            last = i
        hidden = 0
    if len(picks) > max_reads:  # 상한이면 고르게 솎는다(시간순 유지)
        step = len(picks) / max_reads
        picks = [picks[int(k * step)] for k in range(max_reads)]
    return picks


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
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_reads: int = 10,
        ffmpeg: str = "ffmpeg",
        every: Optional[float] = None,
        layout: Optional[Layout] = None,
    ) -> None:
        """`every`: 경기 영상이면 그 간격(초)마다 읽는다(plan_periodic). 없으면 하이라이트 방식.
        `layout`: 중계사 점수판 위치 — 있으면 주자·아웃은 화소 색으로 읽는다(VLM이 못 읽었다).
        """
        self.api_key, self.model, self.max_reads, self.ffmpeg = api_key, model, max_reads, ffmpeg
        self.base_url = base_url.rstrip("/")
        self.every, self.layout = every, layout
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

    def _rgb(self, video: Path, t: float) -> bytes:
        """원본 프레임을 RGB_W×RGB_H rgb24로 — 위치는 비율이라 크기를 고정해도 된다."""
        return subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t:.2f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={RGB_W}:{RGB_H}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout

    def _thumbs(self, video: Path, corner: str, start: float, duration: float) -> list[bytes]:
        """[start, start+duration) 구간을 1초마다 — i번째 썸네일은 영상 start+i초."""
        x, y, w, h = CORNERS[corner]
        vf = f"fps=1,crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale={THUMB_W}:{THUMB_H},format=gray"
        raw = subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.2f}",
                "-t",
                f"{duration:.2f}",
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

    def pick_corner(
        self, video: Path, duration: float, start: float = 0.0
    ) -> tuple[Optional[str], float]:
        for t in (1.0, duration * 0.3, duration * 0.6):
            t = start + min(max(0.0, t), max(0.0, duration - 0.5))
            images = [self._frame(video, t, crop_filter(c, 2.0)) for c in CORNERS]
            answer = self._ask(images, PICK_PROMPT, 64) or {}
            try:
                k = int(answer.get("image", 0))
            except (TypeError, ValueError):
                k = 0
            if 1 <= k <= len(CORNERS):
                return list(CORNERS)[k - 1], t
        return None, 0.0

    def read(self, video: Path, duration: float, start: float = 0.0) -> dict[str, Any]:
        """영상의 [start, start+duration) → {"corner", "scoreboard": [판독…], "requests"}.

        시각은 영상 시작 기준 초. 경기 영상은 VLM이 본 구간(--start·--end·--limit)만 읽는다.
        """
        try:
            corner, t_ref = self.pick_corner(video, duration, start)
            if corner is None:
                return {"corner": None, "scoreboard": [], "requests": self.requests}
            thumbs = self._thumbs(video, corner, start, duration)
            if not thumbs:
                return {"corner": corner, "scoreboard": [], "requests": self.requests}
            ref = min(int(t_ref - start), len(thumbs) - 1)
            readings, failed = [], 0
            plan = (
                plan_periodic(thumbs, ref, self.every, self.max_reads)
                if self.every
                else plan_reads(thumbs, ref, self.max_reads)
            )
            for first, at in plan:
                try:  # 한 장 실패(503 등)로 클립 전체를 버리지 않는다
                    answer = self._ask(
                        [self._frame(video, start + at + 0.5, crop_filter(corner))],
                        READ_PROMPT,
                        256,
                    )
                except VideoAnalyzerError:
                    failed += 1
                    continue
                r = normalize_reading(answer, start + first)
                if r and self.layout is not None:
                    px = self._rgb(video, start + at + 0.5)
                    r["bases"], r["outs"] = decode(self.layout, rgb_sampler(px, RGB_W, RGB_H))
                if r:
                    readings.append(r)
        except (subprocess.CalledProcessError, VideoAnalyzerError) as exc:
            raise VideoAnalyzerError(f"점수판 판독 실패: {exc}") from exc
        return {
            "corner": corner,
            "scoreboard": readings,
            "requests": self.requests,
            "failed_reads": failed,
            "every": self.every,  # 있으면 같은 상태를 여러 번 읽었다 → 안정화 2 (observe)
            "layout": self.layout.name if self.layout else None,
        }
