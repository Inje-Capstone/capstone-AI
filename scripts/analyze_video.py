#!/usr/bin/env python
"""경기 영상을 VSS 방식(청크 → VLM 관찰)으로 분석하고 영상-중계 오프셋을 추정한다.

흐름: ffmpeg로 청크 분할(저해상도·무음) → 청크마다 VLM 관찰 → `data/video/{game}.json`에
청크 단위로 누적 저장(중단돼도 이어서 돈다) → 중계 이벤트와 정합해 오프셋 추정 →
`--apply`면 fixture의 `relay_video_offset_sec`를 갱신.

서버는 이 결과 파일만 읽는다. 영상과 VLM 호출은 여기서 한 번뿐이다.

백엔드:
  cosmos  NVIDIA 호스팅 Cosmos Reason (VSS 기본 VLM). 환경변수 NVIDIA_API_KEY 필요.
  vss     VSS LVS 서버 (--vss-url, --clip-base-url: VSS가 청크를 읽어 갈 주소)

사용:
    python scripts/analyze_video.py 20260920HHLG02026 game.mp4 --limit 3        # 앞 3청크만 시험
    python scripts/analyze_video.py 20260920HHLG02026 game.mp4 --apply
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.relay.fixture import FixtureRelaySource  # noqa: E402
from app.adapters.video.base import (  # noqa: E402
    PROMPT_VERSION,
    VideoAnalyzer,
    VideoAnalyzerError,
    VideoClip,
)
from app.config import get_settings  # noqa: E402
from app.domain.video_sync import Anchor, OffsetEstimate, estimate_offset  # noqa: E402


# ── 순수 함수 (테스트 대상) ───────────────────────────────────────────────
def plan_clips(
    duration: float, clip_sec: float, start: float = 0.0, end: Optional[float] = None
) -> list[tuple[float, float]]:
    """[(청크 시작, 길이)]. 마지막 청크는 남은 길이만큼. 1초 미만 꼬리는 버린다."""
    stop = min(duration, end) if end is not None else duration
    clips, t = [], max(0.0, start)
    while stop - t >= 1.0:
        length = min(clip_sec, stop - t)
        clips.append((round(t, 3), round(length, 3)))
        t += clip_sec
    return clips


def ffmpeg_cut_cmd(
    ffmpeg: str, video: Path, start: float, length: float, out: Path, height: int = 360
) -> list[str]:
    """분석용 청크: 저해상도·무음·짧은 GOP. VLM 토큰과 업로드 크기를 줄인다."""
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(video),
        "-vf", f"scale=-2:{height}", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        str(out),
    ]


def anchors_from(events: list[dict[str, Any]]) -> list[Anchor]:
    return [Anchor(float(e["t_start"]), e["event_type"]) for e in events]


# ── 실행 ────────────────────────────────────────────────────────────────
def find_tool(name: str, explicit: Optional[str]) -> str:
    candidates = [explicit, os.getenv(f"ROOKIE_{name.upper()}"), shutil.which(name)]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit(f"{name}를 찾지 못했다 — PATH에 넣거나 --{name}로 경로를 준다")


def probe_duration(ffprobe: str, video: Path) -> float:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def load_snapshot(path: Path, game_id: str, analyzer: VideoAnalyzer) -> dict[str, Any]:
    if path.exists():
        snap = json.loads(path.read_text(encoding="utf-8"))
        same = (snap.get("provider"), snap.get("model"), snap.get("prompt_version")) == (
            analyzer.provider, analyzer.model, PROMPT_VERSION)
        if same:
            return snap
        print("모델·프롬프트가 바뀌어 스냅샷을 새로 만든다", file=sys.stderr)
    return {
        "_note": "영상 VLM 관찰 결과(판정 아님). scripts/analyze_video.py로 재생성한다.",
        "game_id": game_id, "provider": analyzer.provider, "model": analyzer.model,
        "prompt_version": PROMPT_VERSION, "done": [], "failed": [], "events": [],
    }


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def run(
    game_id: str,
    video: Path,
    analyzer: VideoAnalyzer,
    clips: list[tuple[float, float]],
    cut: Any,
    out_path: Path,
) -> dict[str, Any]:
    """청크를 차례로 분석해 스냅샷에 누적한다. `cut(start, length, dest)`가 청크 파일을 만든다."""
    snap = load_snapshot(out_path, game_id, analyzer)
    snap["video"] = video.name
    done = set(snap["done"])
    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, length) in enumerate(clips, 1):
            if start in done:
                continue
            dest = Path(tmp) / f"clip_{int(start):06d}.mp4"
            cut(start, length, dest)
            try:
                events = analyzer.analyze(VideoClip(dest, start, length))
            except VideoAnalyzerError as exc:
                print(f"[{i}/{len(clips)}] {start:.0f}s 실패: {exc}", file=sys.stderr)
                snap["failed"] = sorted(set(snap.get("failed", [])) | {start})
                save(out_path, snap)
                continue
            snap["events"] = sorted(
                snap["events"] + [e.model_dump() for e in events], key=lambda e: e["t_start"])
            snap["done"] = sorted(done | {start})
            snap["failed"] = [s for s in snap.get("failed", []) if s != start]
            done.add(start)
            save(out_path, snap)
            print(f"[{i}/{len(clips)}] {start:.0f}s +{len(events)}개")
    return snap


def apply_offset(fixture_dir: Path, game_id: str, offset: float) -> Path:
    for path in sorted(fixture_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (raw.get("game") or {}).get("id") == game_id:
            raw["game"]["relay_video_offset_sec"] = int(round(offset))
            save(path, raw)
            return path
    raise SystemExit(f"fixture에서 경기를 찾지 못했다: {game_id}")


def build_analyzer(args: argparse.Namespace) -> VideoAnalyzer:
    if args.backend == "cosmos":
        from app.adapters.video.cosmos import DEFAULT_BASE_URL, DEFAULT_MODEL, CosmosAnalyzer

        key = os.getenv("NVIDIA_API_KEY", "").strip()
        if not key:
            raise SystemExit("NVIDIA_API_KEY가 없다 (.env 또는 환경변수)")
        return CosmosAnalyzer(
            key, base_url=args.base_url or DEFAULT_BASE_URL,
            model=args.model or DEFAULT_MODEL, fps=args.fps,
        )
    from app.adapters.video.vss import VssAnalyzer

    if not (args.vss_url and args.clip_base_url):
        raise SystemExit("vss 백엔드는 --vss-url과 --clip-base-url이 필요하다")
    base = args.clip_base_url.rstrip("/")
    return VssAnalyzer(
        args.vss_url, url_for=lambda c: f"{base}/{c.path.name}",
        token=os.getenv("VSS_API_TOKEN") or None, model=args.model or "",
    )


def report(estimate: Optional[OffsetEstimate]) -> None:
    if estimate is None:
        print("오프셋 추정 불가 — 중계와 짝지을 영상 관찰이 없다")
        return
    flag = "신뢰" if estimate.confident else "표 부족 — 수동 확인 필요"
    print(
        f"오프셋 {estimate.offset:+.1f}s (짝 {estimate.matched}/{estimate.anchors}, "
        f"오차 중앙값 {estimate.residual}s, {flag})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game_id")
    parser.add_argument("video", type=Path)
    parser.add_argument("--backend", choices=("cosmos", "vss"), default="cosmos")
    parser.add_argument("--clip-sec", type=float, default=60.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N청크만 (시험용)")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None, help="cosmos: 로컬 NIM이면 http://host:8000/v1")
    parser.add_argument("--vss-url", default=None)
    parser.add_argument("--clip-base-url", default=None)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="기본: ROOKIE_VIDEO_DIR")
    parser.add_argument("--apply", action="store_true", help="신뢰할 만하면 fixture 오프셋 갱신")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"영상이 없다: {args.video}")
    settings = get_settings()
    feed = FixtureRelaySource(settings.fixture_dir).load(args.game_id)  # 경기부터 확인

    ffmpeg = find_tool("ffmpeg", args.ffmpeg)
    ffprobe = find_tool("ffprobe", args.ffprobe)
    analyzer = build_analyzer(args)
    clips = plan_clips(probe_duration(ffprobe, args.video), args.clip_sec, args.start, args.end)
    if args.limit:
        clips = clips[: args.limit]

    def cut(start: float, length: float, dest: Path) -> None:
        subprocess.run(
            ffmpeg_cut_cmd(ffmpeg, args.video, start, length, dest, args.height), check=True)

    out_path = (args.out_dir or settings.video_dir) / f"{args.game_id}.json"
    snap = run(args.game_id, args.video, analyzer, clips, cut, out_path)

    estimate = estimate_offset(feed.events, anchors_from(snap["events"]))
    report(estimate)
    if estimate is not None:
        snap["offset"] = {
            "offset": estimate.offset, "matched": estimate.matched,
            "anchors": estimate.anchors, "residual": estimate.residual,
            "confident": estimate.confident,
        }
        save(out_path, snap)
    print(f"저장: {out_path} (관찰 {len(snap['events'])}개)")

    if args.apply:
        if estimate is None or not estimate.confident:
            print("--apply 생략: 추정을 신뢰할 수 없다", file=sys.stderr)
            return 2
        path = apply_offset(settings.fixture_dir, args.game_id, estimate.offset)
        print(f"fixture 오프셋 갱신: {path.name} → {int(round(estimate.offset))}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
