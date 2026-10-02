"""NVIDIA VSS 계열 VLM을 호스팅 API(NIM)로 직접 부른다(GPU 불필요).

조사서 권고안: VSS 풀스택을 배포하지 않고 "VSS 방식"을 엔진 안에서 가볍게 돌린다.
OpenAI 호환 chat completions에 청크 영상을 base64 `video_url`로 싣는다.

2026-09-28 호스팅 엔드포인트 실측:
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (VSS 3.2의 Omni 모델) — 영상 입력 OK,
  `media_io_kwargs`(fps) 수용, 추론은 `reasoning_content`로 분리되고 `content`엔 답만 온다.
  영상+**음성**을 함께 받아 해설("보크!")까지 단서로 쓴다 → 기본값.
- `nvidia/cosmos-reason2-8b` — 모델 목록엔 있으나 이 계정에서 404(function not found).
- `nvidia/cosmos3-nano-reasoner` — 호스팅 안 됨(로컬 NIM 전용). `--base-url`로 로컬 NIM이면 사용.
- 호스팅은 수용량이 차면 503(ResourceExhausted)을 준다 → post_json이 대기 후 재시도.
트라이얼 약관상 운영 금지 — 사전 배치 분석에만 쓴다.
"""

import base64
from typing import Optional

from app.adapters.video.base import ClipAnalysis, VideoClip, event_prompt, parse_analysis
from app.adapters.video.http import completion_text, post_json

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


class CosmosAnalyzer:
    provider = "nvidia-nim"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        fps: float = 2.0,
        max_tokens: int = 1024,
        scenario: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fps = fps
        self.max_tokens = max_tokens
        self.prompt = event_prompt(scenario) if scenario else event_prompt()

    def request_body(self, clip: VideoClip) -> dict:
        video_b64 = base64.b64encode(clip.path.read_bytes()).decode("ascii")
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url",
                         "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "media_io_kwargs": {"video": {"fps": self.fps}},
        }

    def analyze(self, clip: VideoClip) -> ClipAnalysis:
        response = post_json(
            f"{self.base_url}/chat/completions", self.request_body(clip), token=self.api_key
        )
        return parse_analysis(completion_text(response), clip, self.provider, self.model)
