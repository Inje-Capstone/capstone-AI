"""영상 타임코드 ↔ 중계 타임코드 정렬. 순수 로직(I/O 없음).

지금은 상수 오프셋 하나짜리 선형 모델이다. 캡스톤2에서 중계 음성 STT로 자동
정렬하게 되면 이 모듈의 `to_relay`만 구간별 매핑으로 바뀌고, 호출부는 그대로다.
"""

from collections.abc import Sequence
from typing import Optional, TypeVar

from app.domain.models import RelayEvent

_T = TypeVar("_T", bound=RelayEvent)


class Timeline:
    """`offset_sec` = 영상 t − 중계 t.

    예) 영상 앞에 5분짜리 프리뷰가 붙어 있으면 offset_sec = 300.
    """

    def __init__(self, offset_sec: int = 0) -> None:
        self.offset_sec = int(offset_sec)

    def to_video(self, relay_t: int) -> int:
        return relay_t + self.offset_sec

    def to_relay(self, video_t: int) -> int:
        return video_t - self.offset_sec

    def events_until(
        self, events: Sequence[_T], video_t: Optional[int]
    ) -> list[_T]:
        """영상 타임코드 기준으로 그 시점까지(포함) 일어난 이벤트."""
        if video_t is None:
            return list(events)
        cutoff = self.to_relay(int(video_t))
        return [e for e in events if e.t <= cutoff]
