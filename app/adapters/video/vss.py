"""NVIDIA VSS LVS(Long Video Summarization) — VSS 서버가 있을 때의 교체 구현.

`POST {base}/v1/summarize` 에 청크 URL과 시나리오·이벤트 목록을 보내고, 구조화 출력
(`{"video_summary", "events":[{start_time,end_time,type,description}]}`)을 받는다.
요청 필드 이름은 VSS CLI(libs/vss/cli, 2026-09 main)의 SummarizeInput을 따랐다.

⚠️ TODO(스키마 미검증): VSS 3.x 실제 배포에서 응답을 실측하지 않았다. VSS는 파일을
직접 받지 않고 `url`(HTTP/S3)이나 사전 업로드 `id`를 받으므로, 청크를 VSS가 읽을 수 있는
위치에 올려 두는 건 호출자 몫이다(`url_for`).
"""

from typing import Callable, Optional

from app.adapters.video.base import EVENT_TYPES, VideoClip, VideoEvent, parse_events
from app.adapters.video.http import completion_text, post_json


class VssAnalyzer:
    provider = "nvidia-vss"

    def __init__(
        self,
        base_url: str,
        url_for: Callable[[VideoClip], str],
        token: Optional[str] = None,
        model: str = "",
        chunk_duration: int = 10,
        scenario: str = "KBO baseball TV broadcast",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.url_for = url_for
        self.token = token
        self.model = model  # 비우면 VSS가 배포된 기본 VLM을 쓴다
        self.chunk_duration = chunk_duration
        self.scenario = scenario

    def request_body(self, clip: VideoClip) -> dict:
        body = {
            "url": self.url_for(clip),
            "scenario": self.scenario,
            "events": [t for t in EVENT_TYPES if t != "other"],
            "chunk_duration": self.chunk_duration,
            "enable_vlm_structured_output": True,
        }
        if self.model:
            body["model"] = self.model
        return body

    def analyze(self, clip: VideoClip) -> list[VideoEvent]:
        response = post_json(
            f"{self.base_url}/v1/summarize", self.request_body(clip), token=self.token
        )
        return parse_events(
            completion_text(response), clip, self.provider, self.model or "vss-default"
        )
