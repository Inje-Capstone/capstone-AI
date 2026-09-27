"""NVIDIA Cosmos Reason — VSS의 기본 VLM을 호스팅 API로 직접 부른다(GPU 불필요).

조사서 권고안: VSS 풀스택을 배포하지 않고 "VSS 방식"을 엔진 안에서 가볍게 돌린다.
OpenAI 호환 chat completions에 청크 영상을 base64 `video_url`로 싣는다.

⚠️ TODO(스키마 미검증): 호스팅 엔드포인트(integrate.api.nvidia.com)가 이 모델의 영상 입력과
`media_io_kwargs`를 그대로 받는지 실측 전이다. 로컬 NIM(`http://127.0.0.1:8000/v1`)이면
문서 예시와 같은 형식이다. 트라이얼 약관상 운영 금지 — 사전 배치 분석에만 쓴다.
"""

import base64
from typing import Optional

from app.adapters.video.base import VideoClip, VideoEvent, event_prompt, parse_events
from app.adapters.video.http import completion_text, post_json

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/cosmos3-nano-reasoner"


class CosmosAnalyzer:
    provider = "nvidia-cosmos"

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

    def analyze(self, clip: VideoClip) -> list[VideoEvent]:
        response = post_json(
            f"{self.base_url}/chat/completions", self.request_body(clip), token=self.api_key
        )
        return parse_events(completion_text(response), clip, self.provider, self.model)
