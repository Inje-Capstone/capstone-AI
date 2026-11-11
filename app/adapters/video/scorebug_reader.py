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
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

from app.adapters.video.base import VideoAnalyzerError, _json_payload
from app.adapters.video.http import completion_text, post_json
from app.adapters.video.scorebug_layout import (
    Layout,
    board_shown,
    decode,
    decode_count,
    decode_half,
    decode_mark,
    rgb_sampler,
)

# 모서리 후보 (x, y, w, h — 화면 비율)
CORNERS = {
    "top_left": (0.0, 0.0, 0.45, 0.32),
    "top_right": (0.55, 0.0, 0.45, 0.32),
    "bottom_left": (0.0, 0.68, 0.45, 0.32),
    "bottom_right": (0.55, 0.68, 0.45, 0.32),
}
# 스냅샷 캐시 키 — 판독 방식·지시문을 바꾸면 올린다.
# v2: 덩어리 판독 시각을 화소가 같아진 뒤로(_settled). v3: 기준 화면을 열두 곳 더 찾음(REF_MORE),
# 위치 파일 half_pair
VERSION = "scorebug-crop-v3"
THUMB_W, THUMB_H = 96, 48
MARK_SCAN_FPS = 2.0  # 삼진 표시 훑기 — 표시가 2~3초 떠 있어 초당 2장이면 4장 이상 걸린다
MARK_GAP = 3.0  # 이보다 떨어진 표시는 다른 삼진
RGB_W, RGB_H = 1280, 720  # 주자·아웃 화소 읽기용 프레임 크기 (scorebug_layout)
PIXEL_TOL = 30  # 같은 화소로 볼 밝기 차
MASK_MIN = 0.3  # 기준과 같은 값으로 이 비율 이상 나온 화소 = 점수판 그래픽
# 위치 파일 영역: 점수판이 보이는 때(기준 화면)를 찾아볼 곳 — 구간 비율
REF_TRIES = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8)
# 위 여섯이 모두 점수판 없는 장면이면(PO 3차전 하이라이트 — 판독 0장) 사이를 더 본다.
# 앞 여섯은 그대로라
# 예전에 찾은 기준 화면은 바뀌지 않는다
REF_MORE = (0.1, 0.15, 0.25, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.75, 0.85, 0.9)
PICK_TRIES = (0.3, 0.6, 0.45, 0.75)  # 모서리 고르기를 해 볼 곳(1초 다음) — 구간 비율
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
Region = tuple[float, float, float, float]


def region_of(corner: Union[str, Region]) -> Region:
    """모서리 이름 또는 위치 파일의 crop(x, y, w, h 비율)."""
    return CORNERS[corner] if isinstance(corner, str) else corner


def crop_filter(corner: Union[str, Region], scale: float = 3.0) -> str:
    x, y, w, h = region_of(corner)
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


def plan_both(
    thumbs: Sequence[bytes], ref: int, every: float, max_reads: int = 600
) -> list[tuple[int, int]]:
    """하이라이트 — 일정 간격(plan_periodic) + 점수판 그림이 바뀔 때(plan_reads)를 합친다.

    실측(2026-11-07, 1080p 하이라이트 3편): 바뀔 때만 읽으면 아래 띠 점수판은 숫자 변화가
    작은 썸네일에 묻혀 놓치고(36칸 중 10), 8초마다만 읽으면 짧게 비친 상태를 놓쳤다(36칸 중 21).
    """
    picks = {at: (start, at) for start, at in plan_periodic(thumbs, ref, every, max_reads)}
    for start, at in plan_reads(thumbs, ref, max_reads):
        picks.setdefault(at, (start, at))
    out = sorted(picks.values(), key=lambda p: p[1])
    if len(out) > max_reads:
        step = len(out) / max_reads
        out = [out[int(k * step)] for k in range(max_reads)]
    return out


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
        prompt: str = READ_PROMPT,
        min_gap: float = 0.0,
        changes: bool = False,
    ) -> None:
        """`every`: 경기 영상이면 그 간격(초)마다 읽는다(plan_periodic). 없으면 하이라이트 방식.
        `layout`: 중계사 점수판 위치 — 있으면 주자·아웃은 화소 색으로 읽는다(VLM이 못 읽었다).
        """
        self.api_key, self.model, self.max_reads, self.ffmpeg = api_key, model, max_reads, ffmpeg
        self.base_url = base_url.rstrip("/")
        self.every, self.layout = every, layout
        self.prompt = prompt  # READ_PROMPT_V2: 회차는 ▲/▼ 옆 숫자만(타순·투구 수를 회차로 읽음)
        if layout is not None and layout.prompt_note:  # 이 점수판만의 안내를 덧붙인다
            self.prompt = f"{prompt} {layout.prompt_note}"
        # 요청 사이 최소 간격(초) — 호스팅 API는 분당 한도를 넘으면 429로 막는다. 한 줄로 이어
        # 보내도 장당 1.3초라 한도를 넘었다(2026-11-07). 다시 보내지 않고 미리 천천히 보낸다
        self.min_gap, self._last = min_gap, -1e9
        self._pace = threading.Lock()  # 실시간은 조각을 여러 줄로 읽는다 — 간격 확인을 한 줄로
        self.changes = changes  # every와 함께: 일정 간격 + 점수판 그림이 바뀔 때도 읽는다
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
        with self._pace:  # 보내기까지 잠금 안에서 — 잠금을 푼 뒤 밀리면 두 요청이 붙어 나간다
            wait = self._last + self.min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            response = post_json(f"{self.base_url}/chat/completions", body, token=self.api_key)
        return _json_payload(completion_text(response))

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

    def _thumbs(
        self, video: Path, corner: Union[str, Region], start: float, duration: float
    ) -> list[bytes]:
        """[start, start+duration) 구간을 1초마다 — i번째 썸네일은 영상 start+i초."""
        x, y, w, h = region_of(corner)
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
        for t in (1.0, *(duration * part for part in PICK_TRIES)):
            t = start + min(max(0.0, t), max(0.0, duration - 0.5))
            images = [self._frame(video, t, crop_filter(c, 2.0)) for c in CORNERS]
            answer = self._ask(images, PICK_PROMPT, 64) or {}
            try:
                k = int(answer.get("image", 0))
            except (TypeError, ValueError):
                k = 0
            if 1 <= k <= len(CORNERS):
                # 고른 자리에서 실제로 읽히는지 본다 — 영상 첫 장면의 팀 소개 그래픽을 점수판으로
                # 고르면 그 뒤 판독이 전부 빈 곳을 읽는다(2026-11-07 하이라이트 실측: 109회 0건)
                corner = list(CORNERS)[k - 1]
                seen = self._ask([self._frame(video, t, crop_filter(corner))], self.prompt, 256)
                if normalize_reading(seen, t):
                    return corner, t
        return None, 0.0

    def _settled(self, video: Path, start: float, first: int, at: int,
                 state: tuple[list[int], int]) -> float:
        """덩어리(first..at초)에서 주자·아웃 화소가 읽은 프레임(at)과 같아진 첫 초.

        plan_reads는 덩어리 시작(first)을 판독 시각으로 찍는다 — 상태가 덩어리 내내 같다고 보고.
        점수판이 장면이 바뀌어도 이어 보이는 중계(MBC 포스트시즌 상자)는 덩어리가 수십 초라
        그 안에서
        상태가 바뀌었다: 한국시리즈 5차전 판독 108장 중 10장이 앞 판독보다 이른 시각(최대 43초)으로
        찍혀 플레이가 엉뚱한 칸에 붙었다(2026-11-10). 읽은 프레임에서 거꾸로 1초씩 되짚어 화소가
        처음 달라지는 곳 바로 뒤를 쓴다. 끝까지 같으면 예전처럼 덩어리 시작.
        """
        assert self.layout is not None
        for i in range(at - 1, first - 1, -1):
            try:  # 되짚다 프레임 하나를 못 읽어도 판독 전체를 버리지 않는다(독립 리뷰)
                px = self._rgb(video, start + i + 0.5)
            except (subprocess.CalledProcessError, OSError, VideoAnalyzerError):
                return float(i + 1)
            if len(px) < RGB_W * RGB_H * 3:
                return float(i + 1)
            bases, outs = decode(self.layout, rgb_sampler(px, RGB_W, RGB_H))
            if (bases, outs) != (list(state[0]), state[1]):
                return float(i + 1)
        return float(first)

    def read_frame(self, video: Path, t: float) -> Optional[dict[str, Any]]:
        """한 장만 읽는다(실시간 조각) — 위치 파일의 영역(crop)으로 자르고, 주자·아웃·초말은 화소로.

        점수판이 안 보이면 None. 실시간은 조각마다 한 장(5초 조각이면 분당 12회 — 요청 한도 안).
        """
        if self.layout is None or self.layout.crop is None:
            raise VideoAnalyzerError("한 장 판독은 점수판 영역(crop)이 있는 위치 파일이 필요하다")
        try:
            frame = self._frame(video, t, crop_filter(self.layout.crop, self.layout.scale))
            r = normalize_reading(self._ask([frame], self.prompt, 256), t)
            if r:
                px = self._rgb(video, t)
                if len(px) < RGB_W * RGB_H * 3:  # 조각 끝을 넘긴 시각(짧은 마지막 조각)
                    raise VideoAnalyzerError(f"프레임을 못 읽음: {video} {t:.1f}s")
                r["bases"], r["outs"] = decode(self.layout, rgb_sampler(px, RGB_W, RGB_H))
                count = decode_count(self.layout, rgb_sampler(px, RGB_W, RGB_H))
                if count:  # 점으로 그린 볼·스트라이크
                    r["balls"], r["strikes"] = count
                half = decode_half(self.layout, px, RGB_W, RGB_H)
                if half:
                    r["half"] = half
                if not _board_whole(self.layout, px):  # 전환·사라지는 중 — 표시도 믿지 않는다
                    return None
                mark = decode_mark(self.layout, px, RGB_W, RGB_H)
                if mark:  # 주자·아웃 칸이 덮였다 — 상태 판독에서 빠지고 단서가 된다
                    r["mark"] = mark
            return r
        except (subprocess.CalledProcessError, OSError) as exc:  # 출력 파일이 안 생김 등
            raise VideoAnalyzerError(f"점수판 판독 실패: {exc}") from exc

    def find_in_layout(
        self, video: Path, duration: float, start: float = 0.0
    ) -> tuple[Optional[Region], float]:
        """위치 파일에 점수판 영역이 있으면 고를 필요 없이, 보이는 때(기준 화면)만 찾는다."""
        assert self.layout is not None and self.layout.crop is not None
        for part in REF_TRIES + REF_MORE:
            t = start + min(max(0.0, duration * part), max(0.0, duration - 0.5))
            frame = self._frame(video, t, crop_filter(self.layout.crop, self.layout.scale))
            if normalize_reading(self._ask([frame], self.prompt, 256), t):
                return self.layout.crop, t
        return None, 0.0

    def read(self, video: Path, duration: float, start: float = 0.0) -> dict[str, Any]:
        """영상의 [start, start+duration) → {"corner", "scoreboard": [판독…], "requests"}.

        시각은 영상 시작 기준 초. 경기 영상은 VLM이 본 구간(--start·--end·--limit)만 읽는다.
        """
        try:
            fixed = self.layout is not None and self.layout.crop is not None
            corner: Union[str, Region, None]
            if fixed:
                corner, t_ref = self.find_in_layout(video, duration, start)
            else:
                corner, t_ref = self.pick_corner(video, duration, start)
            scale = self.layout.scale if fixed and self.layout else 3.0
            if corner is None:
                return {"corner": None, "scoreboard": [], "requests": self.requests}
            thumbs = self._thumbs(video, corner, start, duration)
            if not thumbs:
                return {"corner": "layout" if fixed else corner, "scoreboard": [],
                        "requests": self.requests}
            ref = min(int(t_ref - start), len(thumbs) - 1)
            readings, failed, dropped = [], 0, 0
            if self.every and self.changes:
                plan = plan_both(thumbs, ref, self.every, self.max_reads)
            elif self.every:
                plan = plan_periodic(thumbs, ref, self.every, self.max_reads)
            else:
                plan = plan_reads(thumbs, ref, self.max_reads)
            for first, at in plan:
                try:  # 한 장 실패(503 등)로 클립 전체를 버리지 않는다
                    answer = self._ask(
                        [self._frame(video, start + at + 0.5, crop_filter(corner, scale))],
                        self.prompt,
                        256,
                    )
                except VideoAnalyzerError:
                    failed += 1
                    continue
                r = normalize_reading(answer, start + first)
                if r and self.layout is not None:
                    px = self._rgb(video, start + at + 0.5)
                    r["bases"], r["outs"] = decode(self.layout, rgb_sampler(px, RGB_W, RGB_H))
                    count = decode_count(self.layout, rgb_sampler(px, RGB_W, RGB_H))
                    if count:  # 점으로 그린 볼·스트라이크도 화소가 VLM보다 낫다
                        r["balls"], r["strikes"] = count
                    half = decode_half(self.layout, px, RGB_W, RGB_H)
                    if half:  # 초·말도 화살표 화소가 VLM보다 믿을 만하다
                        r["half"] = half
                    mark = decode_mark(self.layout, px, RGB_W, RGB_H)
                    if not _board_whole(self.layout, px):  # 전환·사라지는 중 — 표시도 버린다
                        r, dropped = None, dropped + 1
                    elif mark:  # 주자·아웃 칸이 삼진 표시로 덮였다 — 상태 판독에서 빠진다
                        r["mark"] = mark
                    if r and at > first:  # 덩어리 안에서 주자·아웃이 바뀌었으면 바뀐 뒤로 찍는다
                        settled = self._settled(video, start, first, at, (r["bases"], r["outs"]))
                        r["t"] = round(start + settled, 2)
                if r:
                    readings.append(r)
        except (subprocess.CalledProcessError, VideoAnalyzerError) as exc:
            raise VideoAnalyzerError(f"점수판 판독 실패: {exc}") from exc
        return {
            "corner": "layout" if fixed else corner,
            "scoreboard": readings,
            "requests": self.requests,
            "failed_reads": failed,
            "dropped_reads": dropped,  # 바탕 색(panel)이 안 맞아 버린 판독
            "every": self.every,  # 있으면 같은 상태를 여러 번 읽었다 → 안정화 2 (observe)
            "layout": self.layout.name if self.layout else None,
        }


def _board_whole(layout: Layout, px: bytes) -> bool:
    """점수판이 온전히 보이나 — 바탕 고정 색(panel)이 없으면 늘 참(예전과 같다).

    전환 효과·사라지는 중인 프레임은 VLM이 숫자를 읽어도 주자·아웃 화소가 엉터리였다
    (만루·3스트라이크 같은 값, panel 있는 8편 판독 671장 중 8장). 바탕 색이 안 맞으면 그 판독을
    버린다 — 조정용 98 → 99/118, 시험용 58/61 그대로(2026-11-08).
    """
    return not layout.panel or board_shown(layout, rgb_sampler(px, RGB_W, RGB_H))


def scan_marks(layout: Optional[Layout], video: Path, start: float = 0.0,
               end: Optional[float] = None, fps: float = MARK_SCAN_FPS) -> list[float]:
    """VLM 없이 화소로 [start, end)를 훑어 삼진 표시가 뜬 시각들(표시마다 처음 본 때).

    위치 파일에 panel(바탕 색)과 mark(표시 칸)가 있을 때만 — 점수판이 보이는 장면만 센다.
    """
    if layout is None or not layout.panel or layout.mark is None:
        return []
    import cv2

    from app.adapters.video.frames import open_frames

    found: list[float] = []
    last: Optional[float] = None
    for t, img in open_frames(video, start, end, target_fps=fps, size=(RGB_W, RGB_H)).frames:
        px = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()
        if board_shown(layout, rgb_sampler(px, RGB_W, RGB_H)) and decode_mark(
                layout, px, RGB_W, RGB_H):
            if last is None or t - last > MARK_GAP:
                found.append(round(float(t), 2))
            last = t
    return found
