#!/usr/bin/env python
"""실시간 영상 분석기 — 영상을 짧은 청크로 잘라 흘리며 분석하고, 결과를 서버로 밀어 넣는다.

ffmpeg가 입력(녹화 파일은 `-re`로 실제 속도 재생, 또는 스트림 URL)을 `--seg-sec`초 조각으로
쓰고, 조각이 완성되는 대로 VLM 분석기에 보내 결과를 `POST /api/live/{id}/ingest`로 보낸다.
VLM 한 번에 15~20초가 걸려 조각 길이보다 길므로 여러 조각을 동시에 분석한다(`--workers`).
서버는 들어온 결과 전체로 다시 판정하므로 순서가 섞여도 괜찮다.

화면은 실제 장면보다 대략 (조각 길이 + VLM 응답 시간)만큼 늦다. 기본은 5초 조각 + 추론 끔:
2026-10-02 실측 VLM 응답 1.5~2.2초 → 약 5~10초 지연 (추론 켜면 3.6~65초로 들쭉날쭉).
NVIDIA 무료 API는 약관상 운영에 못 쓴다. 시연·테스트용이고, 실제 운영은 GPU NIM(`--base-url`)으로.

사용:
    ROOKIE_LIVE_TOKEN=... python scripts/live_video.py LIVE1 game.mp4 --away 한화 --home LG
    python scripts/live_video.py LIVE1 rtmp://... --server http://localhost:8000 --no-realtime
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.relay.naver import (  # noqa: E402
    NaverRelayError,
    convert_game,
    fetch_game,
    fetch_preview,
)
from app.adapters.video.base import VideoAnalyzerError, VideoClip  # noqa: E402

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from analyze_video import build_analyzer, find_tool  # noqa: E402


def segment_cmd(ffmpeg: str, source: str, out_dir: Path, seg_sec: float, height: int,
                realtime: bool) -> list[str]:
    """입력 → seg_%05d.mp4 조각. 녹화 파일이면 -re로 실제 속도(시연용 '가짜 실시간')."""
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        *(["-re"] if realtime else []), "-i", source,
        "-vf", f"scale=-2:{height}", "-c:a", "aac", "-b:a", "32k", "-ac", "1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        # 조각은 키프레임에서만 잘린다 — 키프레임을 조각 길이마다 강제해야 n번째 조각 = n×길이초
        "-force_key_frames", f"expr:gte(t,n_forced*{seg_sec})",
        "-f", "segment", "-segment_time", f"{seg_sec}", "-reset_timestamps", "1",
        str(out_dir / "seg_%05d.mp4"),
    ]


def ready_segments(out_dir: Path, done: set, producer_alive: bool) -> list[Path]:
    """완성된 조각. ffmpeg가 다음 조각을 쓰기 시작해야 앞 조각이 끝난 것이다(마지막은 종료 후)."""
    segs = sorted(out_dir.glob("seg_*.mp4"))
    complete = segs if not producer_alive else segs[:-1]
    return [s for s in complete if s.name not in done]


def seg_index(path: Path) -> int:
    return int(path.stem.split("_")[1])


class Server:
    def __init__(self, base: str, token: str) -> None:
        self.base = base.rstrip("/")
        self.token = token

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base}{path}", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Live-Token": self.token},
            method="POST")
        with urllib.request.urlopen(req, timeout=30) as res:  # noqa: S310
            return json.loads(res.read().decode("utf-8"))


def data_payload(game_id: str, fetch=fetch_game, preview=fetch_preview) -> dict[str, Any]:
    """네이버 중계 현재분 → /data 본문. 매번 전체를 받는다(이닝 수만큼 요청)."""
    info, innings = fetch(game_id)
    fx = convert_game(info, innings, preview=preview(game_id))
    return {"events": fx["events"], "context": fx["game"].get("context") or {}}


class DataPoller(threading.Thread):
    """경기 중 네이버 중계를 `every`초마다 받아 서버로. 실패해도 영상 분석은 계속된다."""

    def __init__(self, naver_id: str, server: Any, game_id: str, every: float,
                 log: Callable[[str], None] = print, payload=data_payload) -> None:
        super().__init__(daemon=True)
        self.naver_id, self.server, self.game_id = naver_id, server, game_id
        self.every, self.log, self.payload = every, log, payload
        self.stop = threading.Event()
        self.sent = 0

    def poll_once(self) -> None:
        try:
            body = self.payload(self.naver_id)
        except (NaverRelayError, OSError, ValueError) as exc:
            self.log(f"데이터 받기 실패(다음에 다시): {exc}")
            return
        res = self.server.post(f"/api/live/{self.game_id}/data", body)
        self.sent += 1
        off = res.get("offset")
        self.log(f"데이터 {len(body['events'])}건 → 서버"
                 + (f" (영상↔데이터 {off:+.0f}s)" if off is not None else " (시간 맞추는 중)"))

    def run(self) -> None:
        while not self.stop.is_set():
            self.poll_once()
            self.stop.wait(self.every)


def analyze_segment(analyzer: Any, path: Path, start: float, length: float) -> dict[str, Any]:
    result = analyzer.analyze(VideoClip(path, start, length))
    return {"scoreboard": result.scoreboard, "speech": result.speech,
            "events": [e.model_dump() for e in result.events]}


def run(
    game_id: str, analyzer: Any, server: Server, out_dir: Path, seg_sec: float,
    producer: Optional[subprocess.Popen], workers: int = 3,
    log: Callable[[str], None] = print,
) -> int:
    """조각이 생기는 대로 분석 → 서버로. 생산자(ffmpeg)가 끝나고 남은 조각까지 처리하면 종료."""
    done: set = set()
    pending: dict[Future, Path] = {}
    sent = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            alive = producer is not None and producer.poll() is None
            for seg in ready_segments(out_dir, done, alive):
                done.add(seg.name)
                start = seg_index(seg) * seg_sec
                pending[pool.submit(analyze_segment, analyzer, seg, start, seg_sec)] = seg
            for fut in [f for f in pending if f.done()]:
                seg = pending.pop(fut)
                try:
                    body = fut.result()
                except VideoAnalyzerError as exc:
                    log(f"{seg.name} 분석 실패: {exc}")
                    continue
                res = server.post(f"/api/live/{game_id}/ingest", body)
                sent += 1
                new = res.get("new_situations") or []
                tail = f" | 새 상황: {', '.join(new)}" if new else ""
                log(f"{seg.name} → 점수판 {len(body['scoreboard'])} 장면 {len(body['events'])}"
                    f" 해설 {len(body['speech'])}{tail}")
            if not alive and not pending and not ready_segments(out_dir, done, False):
                break
            time.sleep(0.5)
    return sent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game_id")
    parser.add_argument("source", help="영상 파일 경로 또는 스트림 URL")
    parser.add_argument("--away", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--date", default="")
    parser.add_argument("--stadium", default="")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--seg-sec", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--no-realtime", action="store_true",
                        help="파일을 실제 속도가 아니라 최대 속도로")
    parser.add_argument("--naver-id", default=None,
                        help="네이버 경기 ID — 주면 선수·구종·기록을 데이터로 붙인다")
    parser.add_argument("--data-sec", type=float, default=20.0, help="데이터 갱신 주기(초)")
    parser.add_argument("--backend", choices=("cosmos", "vss"), default="cosmos")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--think", dest="no_think", action="store_false",
                        help="VLM 추론 켜기 (느림 — 기본은 꺼서 지연을 줄인다)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vss-url", default=None)
    parser.add_argument("--clip-base-url", default=None)
    parser.add_argument("--ffmpeg", default=None)
    args = parser.parse_args()

    token = os.getenv("ROOKIE_LIVE_TOKEN", "").strip()
    if not token:
        raise SystemExit("ROOKIE_LIVE_TOKEN이 없다 — 서버와 같은 값을 준다")
    server = Server(args.server, token)
    server.post(f"/api/live/{args.game_id}", {"away_team": args.away, "home_team": args.home,
                                               "date": args.date, "stadium": args.stadium})
    analyzer = build_analyzer(args)
    ffmpeg = find_tool("ffmpeg", args.ffmpeg)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        producer = subprocess.Popen(segment_cmd(
            ffmpeg, args.source, out_dir, args.seg_sec, args.height, not args.no_realtime))
        print(f"실시간 분석 시작: {args.game_id} ({args.seg_sec:.0f}초 조각, 동시 {args.workers})")
        poller = None
        if args.naver_id:
            poller = DataPoller(args.naver_id, server, args.game_id, args.data_sec)
            poller.start()
        try:
            sent = run(args.game_id, analyzer, server, out_dir, args.seg_sec, producer,
                       args.workers)
        finally:
            if producer.poll() is None:
                producer.terminate()
            if poller is not None:
                poller.stop.set()
                poller.poll_once()  # 마지막 상태로 한 번 더
    server.post(f"/api/live/{args.game_id}/end", {})
    print(f"종료: 조각 {sent}개 전송")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
