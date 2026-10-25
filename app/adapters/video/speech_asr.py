"""해설 받아쓰기 — NVIDIA 호스팅 Whisper large-v3 (Riva gRPC). 선택 설치: pip install -e ".[asr]"

실측(2026-10-26 KBO 하이라이트): 영상 VLM(Nemotron Omni)은 한국어 해설을 거의 못 알아듣고
지어낸 문장을 적었다. 같은 소리를 Whisper에 주자 "믿을 수 없는 트리플 플레이"처럼 정확히 적었다.
판정 단서로 쓰는 해설은 이 경로로 받는다. 무료 API 키(NVIDIA_API_KEY)를 그대로 쓴다.

Whisper 응답에는 단어 시각이 없어, 소리를 `piece_sec` 조각으로 잘라 조각 시작 시각을 붙인다.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.adapters.video.base import VideoAnalyzerError

RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
WHISPER_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"  # build.nvidia.com whisper-large-v3


def plan_pieces(duration: float, piece_sec: float) -> list[tuple[float, float]]:
    """[(시작, 길이)] — 1초 미만 꼬리는 버린다."""
    pieces, t = [], 0.0
    while duration - t >= 1.0:
        pieces.append((round(t, 3), round(min(piece_sec, duration - t), 3)))
        t += piece_sec
    return pieces


class WhisperTranscriber:
    """영상 파일 → [{"t": 초, "text": 해설}] (시각은 영상 시작 기준)."""

    def __init__(
        self, api_key: str, ffmpeg: str = "ffmpeg", language: str = "ko", piece_sec: float = 30.0
    ) -> None:
        try:
            import riva.client
        except ImportError as exc:  # 선택 설치
            raise VideoAnalyzerError('해설 받아쓰기는 pip install -e ".[asr]"가 필요하다') from exc
        self._riva = riva.client
        auth = riva.client.Auth(
            uri=RIVA_SERVER,
            use_ssl=True,
            metadata_args=[
                ["function-id", WHISPER_FUNCTION_ID],
                ["authorization", f"Bearer {api_key}"],
            ],
        )
        self._asr = riva.client.ASRService(auth)
        self.ffmpeg, self.language, self.piece_sec = ffmpeg, language, piece_sec
        self.requests = 0

    def _wav(self, video: Path, start: float, length: float, out: Path) -> None:
        subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.2f}",
                "-t",
                f"{length:.2f}",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                str(out),
            ],
            check=True,
        )

    def transcribe(self, video: Path, duration: float) -> list[dict[str, Any]]:
        config = self._riva.RecognitionConfig(
            language_code=self.language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        )
        segments = []
        with tempfile.TemporaryDirectory() as tmp:
            for start, length in plan_pieces(duration, self.piece_sec):
                wav = Path(tmp) / f"a_{int(start):05d}.wav"
                try:
                    self._wav(video, start, length, wav)
                    self.requests += 1
                    resp = self._asr.offline_recognize(wav.read_bytes(), config)
                except Exception as exc:  # noqa: BLE001 — gRPC·ffmpeg 오류 모두 조각 하나만 버린다
                    segments.append({"t": start, "text": "", "error": str(exc)[:200]})
                    continue
                text = " ".join(
                    r.alternatives[0].transcript.strip() for r in resp.results if r.alternatives
                ).strip()
                if text:
                    segments.append({"t": start, "text": text})
        return segments
